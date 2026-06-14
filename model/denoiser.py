"""DiT (Transformer + AdaLN) 去噪网络。

迁移自 BuildingBlock/scene_synthesis/networks/denoise_net_transformer_adaln.py。
只保留默认 building 无条件配置实际走的路径：
    - seperate_all：class / bbox 两路独立 MLP 编解码
    - pos_embeding_way="learned"：用 bbox 角点学习式位置编码
砍掉了原文件里的 window / hilbert 序列化 / cross-attention 文本条件 /
surface_loc / gradient-checkpoint 等死路径。

数据通道约定：[translation(3), size(3), angle(2), class(class_dim)]，
即 bbox 在前、class 在后，与 GaussianDiffusion 的切片保持一致。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------- helpers
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """标准正弦时间步嵌入，返回 [N, dim]。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def init_linear(l: nn.Linear, stddev: float) -> None:
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN 调制： x * (1 + scale) + shift。shift/scale 形如 [B, W]。"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------- modules
class LearnablePositionalEncoding(nn.Module):
    """用 MLP 把 bbox 角点 (6 维) 映射成位置编码。"""

    def __init__(self, input_dim: int = 6, hidden_dims: List[int] = [512], output_dim: int = 512):
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, device, dtype, heads: int, n_ctx: int):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3  # 3 for q,k,v
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return out.permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)


class MultiheadAttention(nn.Module):
    def __init__(self, *, device, dtype, n_ctx: int, width: int, heads: int, init_scale: float):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, *, device, dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    """DiT block：自注意力 + MLP，均带 AdaLN 调制（zero-init 的 8-way modulation）。"""

    def __init__(self, *, device, dtype, n_ctx: int, width: int, heads: int,
                 init_scale: float = 1.0, gate_shifting: bool = True):
        super().__init__()
        self.gate_shifting = gate_shifting
        self.attn = MultiheadAttention(
            device=device, dtype=dtype, n_ctx=n_ctx, width=width, heads=heads, init_scale=init_scale
        )
        self.ln_1 = nn.LayerNorm(width, elementwise_affine=False, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, elementwise_affine=False, device=device, dtype=dtype)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 8 * width, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (
            shift_msa, scale_msa, shift_after_msa, scale_after_msa,
            shift_mlp, scale_mlp, shift_after_mlp, scale_after_mlp,
        ) = self.adaLN_modulation(c).chunk(8, dim=1)
        if not self.gate_shifting:
            shift_after_mlp = shift_after_mlp * 0
            shift_after_msa = shift_after_msa * 0

        x = x + modulate(
            self.attn(modulate(self.ln_1(x), shift_msa, scale_msa)),
            shift_after_msa, scale_after_msa,
        )
        x = x + modulate(
            self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp)),
            shift_after_mlp, scale_after_mlp,
        )
        return x


class Transformer(nn.Module):
    def __init__(self, *, device, dtype, n_ctx: int, width: int, layers: int, heads: int,
                 init_scale: float = 0.25, gate_shifting: bool = True):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device, dtype=dtype, n_ctx=n_ctx, width=width, heads=heads,
                    init_scale=init_scale, gate_shifting=gate_shifting,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, c)
        return x


class PointTransformer1D(nn.Module):
    """DiT 去噪器。forward(x, t) -> 与 x 同形的预测（eps / x0 / v，由扩散侧解释）。

    x: [B, N, channels]，t: [B]。
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 128,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
        dim: int = 512,
        channels: int = 22,
        class_dim: int = 14,
        translation_dim: int = 3,
        size_dim: int = 3,
        angle_dim: int = 2,
        context_dim: int = 0,
        instanclass_dim: int = 0,
        seperate_all: bool = True,
        pos_embeding_way: str = "learned",
        size_half: bool = False,
        gate_shifting: bool = True,
        split_heads: bool = False,
        **unused,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.channels = channels
        self.seperate_all = seperate_all
        self.split_heads = split_heads
        self.class_dim = class_dim
        # split-heads 下 class 不进扩散：只预测 n_materials 个材质 logit + 1 个 objectness logit
        self.n_materials = class_dim - 1
        self.translation_dim = translation_dim
        self.size_dim = size_dim
        self.angle_dim = angle_dim
        self.bbox_dim = translation_dim + size_dim + angle_dim
        self.pos_embeding_way = pos_embeding_way
        self.size_half = size_half

        if "learned" in self.pos_embeding_way:
            self.pos_embed = LearnablePositionalEncoding(6, [width], width)

        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )

        context_dim += instanclass_dim
        self.context_embed = (
            nn.Sequential(nn.Linear(context_dim, width, bias=True)) if context_dim else None
        )

        self.ln_pre = nn.LayerNorm(width, elementwise_affine=False, device=device, dtype=dtype)
        self.backbone = Transformer(
            device=device, dtype=dtype, n_ctx=n_ctx + int(time_token_cond), width=width,
            layers=layers, heads=heads, init_scale=init_scale, gate_shifting=gate_shifting,
        )
        self.ln_post = nn.LayerNorm(width, elementwise_affine=False, device=device, dtype=dtype)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 4 * width, bias=True))

        if self.split_heads:
            # 只编码 bbox（class 不作为输入）
            self.bbox_embedf = PointTransformer1D._encoder_mlp(dim, self.bbox_dim)
            input_channels = dim
        elif self.seperate_all:
            self.class_embedf = PointTransformer1D._encoder_mlp(dim, self.class_dim)
            self.bbox_embedf = PointTransformer1D._encoder_mlp(dim, self.bbox_dim)
            input_channels = dim
        else:
            input_channels = channels

        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, input_channels, device=device, dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

        if self.split_heads:
            # 三头：bbox 走扩散(v)，class/obj 是分类 logit（不加噪）
            self.bbox_hidden2output = PointTransformer1D._decoder_mlp(dim, self.bbox_dim)
            self.class_hidden2output = PointTransformer1D._decoder_mlp(dim, self.n_materials)
            self.obj_hidden2output = PointTransformer1D._decoder_mlp(dim, 1)
        elif self.seperate_all:
            self.class_hidden2output = PointTransformer1D._decoder_mlp(dim, self.class_dim)
            self.bbox_hidden2output = PointTransformer1D._decoder_mlp(dim, self.bbox_dim)

        # Zero-out adaLN modulation layers（DiT 初始化为恒等）
        for block in self.backbone.resblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None, context_cross=None) -> torch.Tensor:
        # x: [B, N, C]
        pos = 0
        if "learned" in self.pos_embeding_way:
            x_trans = x[:, :, : self.translation_dim].clamp(0, 1)
            x_sizes = x[:, :, self.translation_dim : self.translation_dim + self.size_dim]
            if self.size_half:
                xyzxyz = torch.cat((x_trans - x_sizes, x_trans + x_sizes), dim=-1)
            else:
                xyzxyz = torch.cat((x_trans - 0.5 * x_sizes, x_trans + 0.5 * x_sizes), dim=-1)
            pos = torch.permute(self.pos_embed(xyzxyz), (0, 2, 1)).contiguous()  # [B, W, N]

        x = torch.permute(x, (0, 2, 1)).contiguous()  # [B, C, N]

        if self.split_heads:
            # split 模式输入只含 bbox（[B, bbox_dim, N]）
            x = self.bbox_embedf(x[:, 0 : self.bbox_dim, :])  # [B, dim, N]
        elif self.seperate_all:
            x_class = self.class_embedf(x[:, self.bbox_dim : self.bbox_dim + self.class_dim, :])
            x_bbox = self.bbox_embedf(x[:, 0 : self.bbox_dim, :])
            x = x_class + x_bbox  # [B, dim, N]

        x = x + pos

        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))

        context_emb = (
            self.context_embed(context).permute(0, 2, 1) if self.context_embed is not None else 0
        )
        x = x + context_emb

        x = self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

        if self.split_heads:
            out_bbox = torch.permute(self.bbox_hidden2output(x), (0, 2, 1)).contiguous()   # [B,N,bbox_dim]
            out_class = torch.permute(self.class_hidden2output(x), (0, 2, 1)).contiguous()  # [B,N,n_materials]
            out_obj = torch.permute(self.obj_hidden2output(x), (0, 2, 1)).contiguous()      # [B,N,1]
            return out_bbox, out_class, out_obj

        if self.seperate_all:
            out_bbox = self.bbox_hidden2output(x)
            out_class = self.class_hidden2output(x)
            out = torch.cat([out_bbox, out_class], dim=1).contiguous()
        else:
            # 非 seperate_all 时 output_proj 已映射回 channels，直接返回
            out = x

        out = torch.permute(out, (0, 2, 1)).contiguous()  # [B, N, C]
        return out

    def _forward_with_cond(self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # [B, C, N] -> [B, N, W]
        c = torch.zeros_like(cond_as_token[0][0])
        for emb, _ in cond_as_token:
            if emb is not None:
                c = c + emb
        shift_pre, scale_pre, shift_post, scale_post = self.adaLN_modulation(c).chunk(4, dim=1)
        h = modulate(self.ln_pre(h), shift_pre, scale_pre)
        h = self.backbone(h, c)
        h = modulate(self.ln_post(h), shift_post, scale_post)
        h = self.output_proj(h)
        return h.permute(0, 2, 1)  # [B, input_channels, N]

    @staticmethod
    def _encoder_mlp(hidden_size: int, input_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size * 2, hidden_size, 1),
        )

    @staticmethod
    def _decoder_mlp(hidden_size: int, output_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size * 2, hidden_size, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size, output_size, 1),
        )


def build_denoiser(net_config: Dict[str, Any], device: Any = "cpu") -> nn.Module:
    """根据配置构造去噪器。默认 building 配置走 dit_adaln (PointTransformer1D)。"""
    net_type = net_config.get("net_type", "dit_adaln")
    kwargs = dict(net_config.get("net_kwargs", {}))
    device = torch.device(device)

    if net_type == "dit_adaln":
        return PointTransformer1D(device=device, dtype=torch.float32, **kwargs)
    raise NotImplementedError(f"denoiser net_type {net_type} is not supported")
