"""数据编码 / 后处理。

迁移自 BuildingBlock/scene_synthesis/datasets/threed_front_dataset.py 中
`Scale_CosinAngle` + `Diffusion` 围绕 `cached_diffusion_cosin_angle_wocm_prmAll`
的编码逻辑。一条样本从 numpy -> 归一化张量（pad 到 max_length），以及反向后处理。

编码后通道（每个字段都已 pad 到 ``max_length``）：
    - translations : [L, 3]  归一化到 [-1, 1]
    - sizes        : [L, 3]  归一化到 [-1, 1]
    - angles       : [L, 2]  (cos, sin)
    - class_labels : [L, C]  删除 start 标签后 (= 材质数 + end)，one-hot 经 *2-1 映射到 {-1, 1}

class one-hot 的最后一维是 end 标记：真实构件 = -1，padding 槽 = +1。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

Bounds = Dict[str, Tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------- utils
def _scale(x: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    """clip 到 [min, max] 后线性映射到 [-1, 1]。"""
    X = x.astype(np.float32)
    X = np.clip(X, minimum, maximum)
    X = (X - minimum) / (maximum - minimum)
    return 2.0 * X - 1.0


def _descale(x: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    """反向：[-1, 1] -> [min, max]。"""
    x = (x + 1.0) / 2.0
    return x * (maximum - minimum) + minimum


# ---------------------------------------------------------------------- encoding
def encode_sample(
    sample: Dict[str, np.ndarray],
    bounds: Bounds,
    max_length: int,
    encoding_type: str,
    pad_zero: bool = False,
) -> Dict[str, np.ndarray]:
    """把原始 numpy 样本编码成模型输入格式（pad 到 ``max_length``）。

    输入 sample:
        - class_labels : [L, n_classes]  one-hot（含 start/end 两列，真实构件这两位为 0）
        - translations : [L, 3]
        - sizes        : [L, 3]
        - angles       : [L, 1]
    """
    translations = sample["translations"].astype(np.float32)
    sizes = sample["sizes"].astype(np.float32)
    angles = sample["angles"].astype(np.float32).reshape(-1, 1)
    class_labels = sample["class_labels"].astype(np.float32)

    L = class_labels.shape[0]
    if L > max_length:
        raise ValueError(f"Sample length {L} exceeds max_length {max_length}")

    # --- Scale_CosinAngle: trans/size 归一化，angle -> cos/sin
    lo, hi = bounds["translations"]
    translations = _scale(translations, lo, hi)
    lo, hi = bounds["sizes"]
    sizes = _scale(sizes, lo, hi)
    angles = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)

    # --- class: 删除 start 标签（倒数第二列），保留 [材质..., end]，再 pad end + *2-1
    new_class_labels = np.concatenate(
        [class_labels[:, :-2], class_labels[:, -1:]], axis=-1
    )
    C = new_class_labels.shape[1]
    end_label = np.eye(C)[-1]  # [0, ..., 0, 1]
    pad_n = max_length - L
    class_out = np.vstack(
        [new_class_labels, np.tile(end_label[None, :], [pad_n, 1])]
    ).astype(np.float32) * 2.0 - 1.0

    # --- bbox 属性 padding
    def _pad_attr(p: np.ndarray, inds) -> np.ndarray:
        if pad_zero:
            pad = np.zeros((pad_n, p.shape[1]), dtype=np.float32)
        else:
            pad = p[inds]
        return np.vstack([p, pad]).astype(np.float32)

    inds = None if (pad_zero or pad_n == 0) else np.random.choice(L, pad_n)
    return {
        "translations": _pad_attr(translations, inds),
        "sizes": _pad_attr(sizes, inds),
        "angles": _pad_attr(angles, inds),
        "class_labels": class_out,
    }


# ---------------------------------------------------------------------- post process
def post_process(pred: Dict[str, Any], bounds: Bounds) -> Dict[str, Any]:
    """encode_sample 的反向过程，把采样结果转回原始坐标系。

    pred: {translations, sizes, angles, class_labels}，每个 shape [1, L, D]，
          torch.Tensor 或 numpy。
    """
    import torch

    def _to_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    translations = _to_np(pred["translations"])
    sizes = _to_np(pred["sizes"])
    angles = _to_np(pred["angles"])

    lo, hi = bounds["translations"]
    translations = _descale(translations, lo, hi)
    lo, hi = bounds["sizes"]
    sizes = _descale(sizes, lo, hi)
    # size 兜底：反归一化后裁掉非物理的负尺寸（合法样本不受影响）
    sizes = np.clip(sizes, 0.0, None)
    # angles: [..., 2] (cos, sin) -> 单个角度（arctan2 对模长不敏感，已等效归一化）
    theta = np.arctan2(angles[..., 1:2], angles[..., 0:1])

    return {
        "class_labels": torch.as_tensor(_to_np(pred["class_labels"])),
        "translations": torch.as_tensor(translations),
        "sizes": torch.as_tensor(sizes),
        "angles": torch.as_tensor(theta),
    }
