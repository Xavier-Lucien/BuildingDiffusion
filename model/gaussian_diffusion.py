"""扩散过程数学核心（只负责 DDPM 数学，不含网络）。

迁移自 BuildingBlock/scene_synthesis/networks/diffusion_ddpm.py
(GaussianDiffusion / DiffusionPoint)。

数据通道约定：[translation(3), size(3), angle(2), class(class_dim)]，
point_dim = bbox_dim + class_dim。class one-hot 的最后一维是 end 标记：
归一化到 [-1,1] 后，真实构件 = -1，padding/空槽 = +1。
"""
from __future__ import annotations

import json
import math
from collections import namedtuple
from functools import partial
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .iou import axis_aligned_bbox_overlaps_3d

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def identity(t, *args, **kwargs):
    return t


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == "linear":
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type.startswith("warm"):
        frac = float(schedule_type[len("warm"):])
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * frac)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == "cosine":
        def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
            betas = []
            for i in range(num_diffusion_timesteps):
                t1 = i / num_diffusion_timesteps
                t2 = (i + 1) / num_diffusion_timesteps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            return np.array(betas).astype(np.float64)

        betas = betas_for_alpha_bar(
            time_num, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
    else:
        raise NotImplementedError(schedule_type)
    return betas


class GaussianDiffusion:
    """线性/warm/cosine beta 时间表下的 DDPM。只负责数学，不含网络。"""

    def __init__(
        self,
        config: Dict[str, Any],
        betas: np.ndarray,
        loss_type: str,
        model_mean_type: str,
        model_var_type: str,
        loss_separate: bool,
        loss_iou: bool,
        train_stats_file: Optional[str],
    ):
        self.class_dim = config.get("class_dim", 14)
        self.translation_dim = config.get("translation_dim", 3)
        self.size_dim = config.get("size_dim", 3)
        self.angle_dim = config.get("angle_dim", 2)
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim

        self.loss_separate = loss_separate
        self.loss_iou = loss_iou
        self.loss_iou_mode = config.get("loss_iou_mode", "iou")
        self.size_half = config.get("size_half", True)
        # split-heads 下 class CE / objectness BCE 相对 bbox 扩散 loss 的权重
        self.split_heads = config.get("split_heads", False)
        self.lambda_class = config.get("lambda_class", 1.0)
        self.lambda_obj = config.get("lambda_obj", 1.0)
        # size/angle 有效性正则（作用在还原的 x0 上）：默认关，保持现有行为
        self.loss_validity = config.get("loss_validity", False)
        self.lambda_size_valid = config.get("lambda_size_valid", 1.0)
        self.lambda_angle_norm = config.get("lambda_angle_norm", 1.0)

        if self.loss_iou:
            if train_stats_file is None:
                raise ValueError("loss_iou=True 需要提供 train_stats_file")
            with open(train_stats_file, "r", encoding="utf-8") as f:
                train_stats = json.load(f)
            c = train_stats["bounds_translations"]
            self._centroids_min = torch.from_numpy(np.array(c[:3])).float()
            self._centroids_max = torch.from_numpy(np.array(c[3:])).float()
            s = train_stats["bounds_sizes"]
            self._sizes_min = torch.from_numpy(np.array(s[:3])).float()
            self._sizes_max = torch.from_numpy(np.array(s[3:])).float()

        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type

        assert isinstance(betas, np.ndarray)
        betas = betas.astype(np.float64)  # float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1.0, alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        alphas_t = torch.from_numpy(alphas).float()
        posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance))
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas_t) / (1.0 - alphas_cumprod)
        )

        snr = alphas_cumprod / (1 - alphas_cumprod)
        if model_mean_type == "eps":
            self.loss_weight = torch.ones_like(snr)
        elif model_mean_type == "x0":
            self.loss_weight = snr
        elif model_mean_type == "v":
            self.loss_weight = snr / (snr + 1)
        else:
            raise NotImplementedError(model_mean_type)

    @staticmethod
    def _extract(a, t, x_shape):
        (bs,) = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    # --------------------------------------------------- target conversions
    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    def _predict_eps_from_start(self, x_t, t, x0):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - x0
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape)

    def _predict_v(self, x0, t, eps):
        return (
            self._extract(self.sqrt_alphas_cumprod.to(x0.device), t, x0.shape) * eps
            - self._extract(self.sqrt_one_minus_alphas_cumprod.to(x0.device), t, x0.shape) * x0
        )

    def _predict_start_from_v(self, x_t, t, v):
        return (
            self._extract(self.sqrt_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t
            - self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_t.device), t, x_t.shape) * v
        )

    def model_predictions(self, denoise_fn, x_t, t, condition, condition_cross,
                          clip_x_start=False, rederive_pred_noise=False):
        model_output = denoise_fn(x_t, t, condition, condition_cross)
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity

        if self.model_mean_type == "eps":
            pred_noise = model_output
            x_start = maybe_clip(self._predict_xstart_from_eps(x_t, t, pred_noise))
            if clip_x_start and rederive_pred_noise:
                pred_noise = self._predict_eps_from_start(x_t, t, x_start)
        elif self.model_mean_type == "x0":
            x_start = maybe_clip(model_output)
            pred_noise = self._predict_eps_from_start(x_t, t, x_start)
        elif self.model_mean_type == "v":
            x_start = maybe_clip(self._predict_start_from_v(x_t, t, model_output))
            pred_noise = self._predict_eps_from_start(x_t, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    # --------------------------------------------------- forward / posterior
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        return (
            self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, data, t, condition, condition_cross,
                        clip_denoised: bool, return_pred_xstart: bool):
        preds = self.model_predictions(denoise_fn, data, t, condition, condition_cross)
        x_recon = preds.pred_x_start
        if clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)

        if self.model_var_type in ["fixedsmall", "fixedlarge"]:
            model_variance, model_log_variance = {
                "fixedlarge": (
                    self.betas.to(data.device),
                    torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device),
                ),
                "fixedsmall": (
                    self.posterior_variance.to(data.device),
                    self.posterior_log_variance_clipped.to(data.device),
                ),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        return model_mean, model_variance, model_log_variance

    # --------------------------------------------------- sampling
    def p_sample(self, denoise_fn, data, t, condition, condition_cross, noise_fn,
                 clip_denoised=False, return_pred_xstart=False):
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data, t, condition, condition_cross,
            clip_denoised=clip_denoised, return_pred_xstart=True,
        )
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        # no noise when t == 0
        nonzero_mask = torch.reshape(
            1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1)
        )
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_loop(self, denoise_fn, shape, device, condition, condition_cross,
                      noise_fn=torch.randn, clip_denoised=True):
        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps)):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(
                denoise_fn=denoise_fn, data=img_t, t=t_, condition=condition,
                condition_cross=condition_cross, noise_fn=noise_fn,
                clip_denoised=clip_denoised, return_pred_xstart=False,
            )
        return img_t

    # --------------------------------------------------- training loss
    def descale_to_origin(self, x, minimum, maximum):
        x = (x + 1) / 2
        return x * (maximum - minimum)[None, None, :] + minimum[None, None, :]

    def p_losses(self, denoise_fn, data_start, t, noise=None, condition=None, condition_cross=None):
        B = data_start.shape[0]
        if noise is None:
            noise = torch.randn(data_start.shape, dtype=data_start.dtype, device=data_start.device)
        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type != "mse":
            raise NotImplementedError(self.loss_type)

        if self.model_mean_type == "eps":
            target = noise
        elif self.model_mean_type == "x0":
            target = data_start
        elif self.model_mean_type == "v":
            target = self._predict_v(data_start, t, noise)
        else:
            raise NotImplementedError

        denoise_out = denoise_fn(data_t, t, condition, condition_cross)
        assert denoise_out.shape == data_start.shape

        if data_start.shape[-1] != self.class_dim + self.bbox_dim:
            raise NotImplementedError(f"unimplemented point dim: {data_start.shape[-1]}")

        td, sd, bd, cd = self.translation_dim, self.size_dim, self.bbox_dim, self.class_dim
        reduce_dims = list(range(1, len(data_start.shape)))

        loss_trans = ((target[:, :, 0:td] - denoise_out[:, :, 0:td]) ** 2).mean(dim=reduce_dims)
        loss_size = ((target[:, :, td:td + sd] - denoise_out[:, :, td:td + sd]) ** 2).mean(dim=reduce_dims)
        loss_angle = ((target[:, :, td + sd:bd] - denoise_out[:, :, td + sd:bd]) ** 2).mean(dim=reduce_dims)
        loss_bbox = ((target[:, :, 0:bd] - denoise_out[:, :, 0:bd]) ** 2).mean(dim=reduce_dims)
        loss_class = ((target[:, :, bd:bd + cd] - denoise_out[:, :, bd:bd + cd]) ** 2).mean(dim=reduce_dims)

        if self.loss_separate:
            losses = loss_bbox + loss_class
        else:
            losses = ((target - denoise_out) ** 2).mean(dim=reduce_dims)

        losses_weight = losses * self._extract(self.loss_weight.to(losses.device), t, losses.shape)

        if self.loss_iou:
            if self.model_mean_type == "eps":
                x_recon = self._predict_xstart_from_eps(data_t, t, eps=denoise_out)
            elif self.model_mean_type == "x0":
                x_recon = denoise_out
            elif self.model_mean_type == "v":
                x_recon = self._predict_start_from_v(data_t, t, v=denoise_out)
            x_recon = torch.clamp(x_recon, -1.0, 1.0)

            trans_recon = x_recon[:, :, 0:td]
            sizes_recon = x_recon[:, :, td:td + sd]
            # end-label（class one-hot 最后一维）作为有效性掩码：<= 0 表示真实构件
            obj_recon = x_recon[:, :, bd + cd - 1: bd + cd]
            valid_mask = (obj_recon <= 0).float().squeeze(2)

            descale_trans = self.descale_to_origin(
                trans_recon, self._centroids_min.to(data_start.device), self._centroids_max.to(data_start.device)
            )
            descale_sizes = self.descale_to_origin(
                sizes_recon, self._sizes_min.to(data_start.device), self._sizes_max.to(data_start.device)
            )
            if self.size_half:
                bbox_corn = torch.cat([descale_trans - descale_sizes, descale_trans + descale_sizes], dim=-1)
            else:
                bbox_corn = torch.cat([descale_trans - descale_sizes / 2, descale_trans + descale_sizes / 2], dim=-1)
            assert bbox_corn.shape[-1] == 6

            bbox_iou = axis_aligned_bbox_overlaps_3d(bbox_corn, bbox_corn, mode=self.loss_iou_mode)
            bbox_iou_mask = valid_mask[:, :, None] * valid_mask[:, None, :]
            pair_mask = 1.0 - torch.eye(
                bbox_iou.shape[-1], dtype=bbox_iou.dtype, device=bbox_iou.device
            )[None, :, :]
            bbox_iou_mask = bbox_iou_mask * pair_mask
            bbox_iou_valid = bbox_iou * bbox_iou_mask
            iou_reduce = list(range(1, len(bbox_iou_valid.shape)))
            bbox_iou_valid_avg = bbox_iou_valid.sum(dim=iou_reduce) / (
                bbox_iou_mask.sum(dim=iou_reduce) + 1e-6
            )
            w_iou = self._extract(self.alphas_cumprod.to(data_start.device), t, bbox_iou.shape)
            loss_iou_valid_avg = (w_iou * 0.1 * bbox_iou_valid).sum(dim=iou_reduce) / (
                bbox_iou_mask.sum(dim=iou_reduce) + 1e-6
            )
            losses_weight = losses_weight + loss_iou_valid_avg
        else:
            loss_iou_valid_avg = torch.zeros(B, device=data_start.device)
            bbox_iou_valid_avg = torch.zeros(B, device=data_start.device)

        if self.loss_validity:
            x0_valid = self._recover_x0(data_t, t, denoise_out, clamp=False)
            size_valid, angle_norm = self._validity_loss(x0_valid)
            losses_weight = (
                losses_weight
                + self.lambda_size_valid * size_valid
                + self.lambda_angle_norm * angle_norm
            )
        else:
            size_valid = torch.zeros(B, device=data_start.device)
            angle_norm = torch.zeros(B, device=data_start.device)

        return losses_weight, {
            "loss.bbox": loss_bbox.mean(),
            "loss.trans": loss_trans.mean(),
            "loss.size": loss_size.mean(),
            "loss.angle": loss_angle.mean(),
            "loss.class": loss_class.mean(),
            "loss.size_valid": size_valid.mean(),
            "loss.angle_norm": angle_norm.mean(),
            "loss.liou": loss_iou_valid_avg.mean(),
            "loss.bbox_iou": bbox_iou_valid_avg.mean(),
        }

    # --------------------------------------------------- validity regularizers
    def _recover_x0(self, x_t, t, model_out, clamp=True):
        """按 model_mean_type 从网络输出还原 x0。

        clamp=True 用于 iou（几何需要合法区间）；validity 正则需 clamp=False，
        否则越界量被 clamp 抹掉，惩罚恒为 0、没有梯度。
        """
        if self.model_mean_type == "eps":
            x0 = self._predict_xstart_from_eps(x_t, t, eps=model_out)
        elif self.model_mean_type == "x0":
            x0 = model_out
        else:  # v
            x0 = self._predict_start_from_v(x_t, t, v=model_out)
        return torch.clamp(x0, -1.0, 1.0) if clamp else x0

    def _validity_loss(self, x0_bbox):
        """size/angle 有效性正则，返回 (size_valid[B], angle_norm[B])。

        - size：归一化空间合法区间是 [-1,1]（descale 后非负有界），惩罚越界量。
        - angle：(cos, sin) 应在单位圆上，惩罚模长偏离 1。
        x0_bbox 既可是完整 point_dim（bbox 在前），也可是 split 的 bbox_dim。
        """
        td, sd, bd = self.translation_dim, self.size_dim, self.bbox_dim
        reduce = list(range(1, x0_bbox.dim()))

        size_seg = x0_bbox[:, :, td:td + sd]
        over = torch.relu(size_seg - 1.0) + torch.relu(-1.0 - size_seg)
        size_valid = (over ** 2).mean(dim=reduce)

        ang = x0_bbox[:, :, td + sd:bd]            # [B, N, 2] = (cos, sin)
        norm = torch.sqrt((ang ** 2).sum(dim=-1) + 1e-8)  # [B, N]
        angle_norm = ((norm - 1.0) ** 2).mean(dim=list(range(1, norm.dim())))
        return size_valid, angle_norm

    # --------------------------------------------------- split-heads loss
    def _collision_loss(self, x0_bbox, valid_mask, t):
        """对 x0 的 bbox 计算重叠惩罚，valid_mask=[B,N] 标记真实构件。

        返回 (loss_iou_valid_avg[B], bbox_iou_valid_avg[B])。复用 axis_aligned 重叠逻辑，
        只统计不同的真实构件对（i != j）。
        """
        td, sd = self.translation_dim, self.size_dim
        device = x0_bbox.device
        trans_recon = x0_bbox[:, :, 0:td]
        sizes_recon = x0_bbox[:, :, td:td + sd]
        descale_trans = self.descale_to_origin(
            trans_recon, self._centroids_min.to(device), self._centroids_max.to(device)
        )
        descale_sizes = self.descale_to_origin(
            sizes_recon, self._sizes_min.to(device), self._sizes_max.to(device)
        )
        if self.size_half:
            bbox_corn = torch.cat([descale_trans - descale_sizes, descale_trans + descale_sizes], dim=-1)
        else:
            bbox_corn = torch.cat([descale_trans - descale_sizes / 2, descale_trans + descale_sizes / 2], dim=-1)

        bbox_iou = axis_aligned_bbox_overlaps_3d(bbox_corn, bbox_corn, mode=self.loss_iou_mode)
        mask = valid_mask[:, :, None] * valid_mask[:, None, :]
        pair_mask = 1.0 - torch.eye(bbox_iou.shape[-1], dtype=bbox_iou.dtype, device=device)[None]
        mask = mask * pair_mask
        reduce = list(range(1, len(bbox_iou.shape)))
        denom = mask.sum(dim=reduce) + 1e-6
        bbox_iou_valid_avg = (bbox_iou * mask).sum(dim=reduce) / denom
        w_iou = self._extract(self.alphas_cumprod.to(device), t, bbox_iou.shape)
        loss_iou_valid_avg = (w_iou * 0.1 * bbox_iou * mask).sum(dim=reduce) / denom
        return loss_iou_valid_avg, bbox_iou_valid_avg

    def p_losses_split(self, denoise_fn, bbox_start, t, class_target, obj_target, noise=None):
        """split-heads 训练损失：bbox 扩散(MSE) + class(CE) + objectness(BCE) [+ iou]。

        bbox_start:   [B, N, bbox_dim]
        class_target: [B, N] long，材质索引（仅真实槽有效）
        obj_target:   [B, N] float，1=真实构件 0=空槽
        denoise_fn 返回 (out_bbox[B,N,bbox_dim], class_logits[B,N,M], obj_logit[B,N,1])。
        """
        B = bbox_start.shape[0]
        if noise is None:
            noise = torch.randn(bbox_start.shape, dtype=bbox_start.dtype, device=bbox_start.device)
        bbox_t = self.q_sample(x_start=bbox_start, t=t, noise=noise)

        if self.loss_type != "mse":
            raise NotImplementedError(self.loss_type)
        if self.model_mean_type == "eps":
            target = noise
        elif self.model_mean_type == "x0":
            target = bbox_start
        elif self.model_mean_type == "v":
            target = self._predict_v(bbox_start, t, noise)
        else:
            raise NotImplementedError

        out_bbox, class_logits, obj_logit = denoise_fn(bbox_t, t, None, None)
        assert out_bbox.shape == bbox_start.shape

        td, sd, bd = self.translation_dim, self.size_dim, self.bbox_dim
        reduce_dims = list(range(1, len(bbox_start.shape)))
        loss_trans = ((target[:, :, 0:td] - out_bbox[:, :, 0:td]) ** 2).mean(dim=reduce_dims)
        loss_size = ((target[:, :, td:td + sd] - out_bbox[:, :, td:td + sd]) ** 2).mean(dim=reduce_dims)
        loss_angle = ((target[:, :, td + sd:bd] - out_bbox[:, :, td + sd:bd]) ** 2).mean(dim=reduce_dims)
        loss_bbox = ((target - out_bbox) ** 2).mean(dim=reduce_dims)
        bbox_weighted = (loss_bbox * self._extract(self.loss_weight.to(loss_bbox.device), t, loss_bbox.shape)).mean()

        # class CE：只在真实槽上
        real = obj_target > 0.5  # [B, N] bool
        if real.any():
            loss_class = F.cross_entropy(class_logits[real], class_target[real])
        else:
            loss_class = class_logits.sum() * 0.0
        # objectness BCE
        loss_obj = F.binary_cross_entropy_with_logits(obj_logit[:, :, 0], obj_target)

        total = bbox_weighted + self.lambda_class * loss_class + self.lambda_obj * loss_obj

        # iou / validity 都基于还原的 x0：iou 用 clamp 版（几何合法），validity 用未 clamp 版（保留越界梯度）
        if self.loss_iou or self.loss_validity:
            x0_raw = self._recover_x0(bbox_t, t, out_bbox, clamp=False)

        if self.loss_iou:
            loss_iou_valid_avg, bbox_iou_valid_avg = self._collision_loss(
                torch.clamp(x0_raw, -1.0, 1.0), obj_target, t
            )
            total = total + loss_iou_valid_avg.mean()
        else:
            loss_iou_valid_avg = torch.zeros(B, device=bbox_start.device)
            bbox_iou_valid_avg = torch.zeros(B, device=bbox_start.device)

        if self.loss_validity:
            size_valid, angle_norm = self._validity_loss(x0_raw)
            total = (
                total
                + self.lambda_size_valid * size_valid.mean()
                + self.lambda_angle_norm * angle_norm.mean()
            )
        else:
            size_valid = torch.zeros(B, device=bbox_start.device)
            angle_norm = torch.zeros(B, device=bbox_start.device)

        return total, {
            "loss.bbox": loss_bbox.mean(),
            "loss.trans": loss_trans.mean(),
            "loss.size": loss_size.mean(),
            "loss.angle": loss_angle.mean(),
            "loss.class": loss_class.detach() if isinstance(loss_class, torch.Tensor) else torch.tensor(0.0),
            "loss.obj": loss_obj.detach(),
            "loss.size_valid": size_valid.mean(),
            "loss.angle_norm": angle_norm.mean(),
            "loss.liou": loss_iou_valid_avg.mean(),
            "loss.bbox_iou": bbox_iou_valid_avg.mean(),
        }
