"""顶层布局扩散模型。

迁移自 BuildingBlock/scene_synthesis/networks/diffusion_scene_layout_ddpm.py
（数据打包 / 采样后处理）。

数据通道约定：[translation(3), size(3), angle(2), class(class_dim)]，
point_dim = bbox_dim + class_dim。class one-hot 的最后一维是 end 标记：
归一化到 [-1,1] 后，真实构件 = -1，padding/空槽 = +1。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .denoiser import build_denoiser
from .gaussian_diffusion import GaussianDiffusion, get_betas


class DiffusionBuildingBlock(nn.Module):
    """把数据集样本 -> [B, N, point_dim] 张量 -> 扩散训练 / 采样的壳。"""

    def __init__(
        self,
        n_classes: int,
        net_config: Dict[str, Any],
        train_stats_file: Optional[str] = None,
        device: Any = "cpu",
    ):
        super().__init__()
        self.n_classes = n_classes
        self.class_dim = net_config.get("class_dim", 14)
        self.translation_dim = net_config.get("translation_dim", 3)
        self.size_dim = net_config.get("size_dim", 3)
        self.angle_dim = net_config.get("angle_dim", 2)
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.point_dim = net_config["point_dim"]
        self.sample_num_points = net_config.get("sample_num_points", 128)
        self.split_heads = net_config.get("split_heads", False)
        self.n_materials = self.class_dim - 1  # 去掉 end 后的真实材质数

        self.denoiser = build_denoiser(net_config, device=device)

        diff_kwargs = dict(net_config["diffusion_kwargs"])
        betas = get_betas(
            diff_kwargs.get("schedule_type", "linear"),
            diff_kwargs.get("beta_start", 1e-4),
            diff_kwargs.get("beta_end", 2e-2),
            diff_kwargs.get("time_num", 1000),
        )
        self.diffusion = GaussianDiffusion(
            config=net_config,
            betas=betas,
            loss_type=diff_kwargs.get("loss_type", "mse"),
            model_mean_type=diff_kwargs.get("model_mean_type", "v"),
            model_var_type=diff_kwargs.get("model_var_type", "fixedsmall"),
            loss_separate=diff_kwargs.get("loss_separate", True),
            loss_iou=diff_kwargs.get("loss_iou", False),
            train_stats_file=diff_kwargs.get("train_stats_file", train_stats_file),
        )

    def _denoise(self, x, t, condition, condition_cross):
        return self.denoiser(x, t, condition, condition_cross)

    def _pack(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        """dict -> [B, N, point_dim]，顺序 [trans, size, angle, class]。"""
        return torch.cat(
            [sample["translations"], sample["sizes"], sample["angles"], sample["class_labels"]],
            dim=-1,
        ).contiguous().float()

    # --------------------------------------------------- train
    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.split_heads:
            return self._forward_split(sample)
        x0 = self._pack(sample)  # [B, N, point_dim]
        B = x0.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=x0.device)
        losses, loss_dict = self.diffusion.p_losses(self._denoise, x0, t)
        loss = losses.mean()
        return {"loss": loss, **loss_dict}

    def _forward_split(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """split-heads 训练：bbox 走扩散，class/objectness 从 class_labels 推导监督目标。"""
        bbox = torch.cat(
            [sample["translations"], sample["sizes"], sample["angles"]], dim=-1
        ).contiguous().float()  # [B, N, bbox_dim]
        cls = sample["class_labels"].float()  # [B, N, class_dim]，{-1,+1}
        class_target = cls[:, :, : self.n_materials].argmax(dim=-1)  # [B, N] 材质索引
        obj_target = (cls[:, :, -1] < 0).float()                     # [B, N] 真实=1（end<0）

        B = bbox.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=bbox.device)
        total, loss_dict = self.diffusion.p_losses_split(
            self._denoise, bbox, t, class_target, obj_target
        )
        return {"loss": total, **loss_dict}

    # --------------------------------------------------- inference
    @torch.no_grad()
    def generate_layout(self, batch_size, num_points, point_dim, device, clip_denoised=False, **unused):
        if self.split_heads:
            return self._generate_split(batch_size, num_points, device, clip_denoised)
        shape = (batch_size, num_points, point_dim)
        samples = self.diffusion.p_sample_loop(
            self._denoise, shape=shape, device=device, condition=None,
            condition_cross=None, clip_denoised=clip_denoised,
        )
        return self._split_samples(samples, device=device)

    @torch.no_grad()
    def _generate_split(self, batch_size, num_points, device, clip_denoised=False):
        """只对 bbox 迭代去噪；末步从 x0_bbox 预测 class/objectness。"""
        shape = (batch_size, num_points, self.bbox_dim)
        bbox_denoise = lambda x, t, c, cc: self.denoiser(x, t, c, cc)[0]  # noqa: E731
        x0_bbox = self.diffusion.p_sample_loop(
            bbox_denoise, shape=shape, device=device, condition=None,
            condition_cross=None, clip_denoised=clip_denoised,
        )
        t0 = torch.zeros(batch_size, dtype=torch.int64, device=device)
        _, class_logits, obj_logit = self.denoiser(x0_bbox, t0, None, None)
        return self._split_samples_split(x0_bbox, class_logits, obj_logit, device=device)

    def _split_samples_split(self, x0_bbox, class_logits, obj_logit, device):
        """按 objectness>0.5 过滤空槽，class_labels 直接给 13 维材质 logit（供 post_process argmax）。"""
        td, sd, bd = self.translation_dim, self.size_dim, self.bbox_dim
        out: List[Dict[str, torch.Tensor]] = []
        for b in range(x0_bbox.shape[0]):
            s = x0_bbox[b : b + 1]      # [1, N, bbox_dim]
            keep = torch.sigmoid(obj_logit[b, :, 0]) > 0.5
            cl = class_logits[b : b + 1]  # [1, N, n_materials]
            out.append({
                "translations": s[:, keep, 0:td].to(device),
                "sizes": s[:, keep, td : td + sd].to(device),
                "angles": s[:, keep, td + sd : bd].to(device),
                "class_labels": cl[:, keep, :].to(device),
            })
        return out

    def _split_samples(self, samples: torch.Tensor, device: str) -> List[Dict[str, torch.Tensor]]:
        """把 [B, N, point_dim] 拆成每个样本的 dict，并按 end-label 过滤空槽。"""
        td, sd, bd, cd = self.translation_dim, self.size_dim, self.bbox_dim, self.class_dim
        n_obj_classes = self.n_classes - 2  # 去掉 start/end 的真实类别数

        out: List[Dict[str, torch.Tensor]] = []
        for b in range(samples.shape[0]):
            s = samples[b : b + 1]  # [1, N, point_dim]
            # objectness: end-label > 0 表示空槽
            end_label = s[:, :, bd + cd - 1 : bd + cd]  # [1, N, 1]
            keep = (end_label[0, :, 0] < 0)  # 真实构件
            class_logits = s[:, :, bd : bd + n_obj_classes]  # 13 维材质 logit
            sample_dict = {
                "translations": s[:, keep, 0:td].to(device),
                "sizes": s[:, keep, td : td + sd].to(device),
                "angles": s[:, keep, td + sd : bd].to(device),
                "class_labels": class_logits[:, keep, :].to(device),
            }
            out.append(sample_dict)
        return out
