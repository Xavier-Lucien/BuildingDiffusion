"""生成结果的关系诊断指标（纯 numpy，不参与训练）。

对应 CLAUDE.md「Algorithm Roadmap -> Recommended First Step: Relation Diagnostics」。
目的：在不改训练的前提下，为生成场景给出可比较的结构一致性度量，定位
「窗户看似合理但没贴墙 / 屋顶相对墙体错位」这类 cross-object 失败。

约定（来自 parse_original_data._rotation_matrix_z：增强绕 z 轴旋转）：
    - 坐标轴 (x, y, z)，z 为竖直方向（高度）。
    - 水平 footprint = x-y 平面投影。
    - bbox 用轴对齐近似（与现有 iou loss 一致），暂忽略构件自身朝向。

所有「贴附」判定基于 AABB 之间的最小间隙 gap（gap=0 表示相交/接触），
阈值相对场景包围盒对角线，避免依赖绝对坐标尺度。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

UP_AXIS = 2
FOOTPRINT_AXES = (0, 1)


# ---------------------------------------------------------------------- 几何工具
def _corners(centers: np.ndarray, sizes_full: np.ndarray) -> np.ndarray:
    """[N,3] 中心 + [N,3] 全尺寸 -> [N,6] 角点 <xmin,ymin,zmin,xmax,ymax,zmax>。"""
    half = sizes_full / 2.0
    return np.concatenate([centers - half, centers + half], axis=-1)


def _pairwise_gap(a_corners: np.ndarray, b_corners: np.ndarray) -> np.ndarray:
    """两组 AABB 的逐对最小间隙，返回 [m, n]；相交/接触为 0。"""
    if a_corners.shape[0] == 0 or b_corners.shape[0] == 0:
        return np.zeros((a_corners.shape[0], b_corners.shape[0]), dtype=np.float32)
    a_lo, a_hi = a_corners[:, None, :3], a_corners[:, None, 3:]
    b_lo, b_hi = b_corners[None, :, :3], b_corners[None, :, 3:]
    sep = np.maximum.reduce([a_lo - b_hi, b_lo - a_hi, np.zeros_like(a_lo - b_hi)])
    return np.sqrt((sep ** 2).sum(axis=-1))


def _scene_diag(corners: np.ndarray) -> float:
    """场景整体包围盒对角线长度，用作相对阈值的尺度。"""
    if corners.shape[0] == 0:
        return 0.0
    lo = corners[:, :3].min(axis=0)
    hi = corners[:, 3:].max(axis=0)
    return float(np.linalg.norm(hi - lo))


def _footprint_rect(corners: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """一组 AABB 在 x-y 平面的并集包围矩形 (xmin, ymin, xmax, ymax)。"""
    if corners.shape[0] == 0:
        return None
    ax, ay = FOOTPRINT_AXES
    xmin = float(corners[:, ax].min())
    ymin = float(corners[:, ay].min())
    xmax = float(corners[:, ax + 3].max())
    ymax = float(corners[:, ay + 3].max())
    return xmin, ymin, xmax, ymax


def _rect_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


# ---------------------------------------------------------------------- 输入适配
def scene_from_post_process(
    processed: Dict[str, Any],
    class_names: Sequence[str],
    size_half: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """把 dataset.post_process 的输出转成 (centers[N,3], sizes_full[N,3], labels[N])。

    processed: {translations, sizes, angles, class_labels}，前三者 [1, N, *]，
               class_labels 为 [1, N, C] 的 logits；size_half=True 表示 sizes 已是半尺寸。
    """
    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    trans = _np(processed["translations"])
    sizes = _np(processed["sizes"])
    labels_logits = _np(processed["class_labels"])
    # 去掉 batch 维
    trans = trans[0] if trans.ndim == 3 else trans
    sizes = sizes[0] if sizes.ndim == 3 else sizes
    labels_logits = labels_logits[0] if labels_logits.ndim == 3 else labels_logits

    sizes_full = sizes * 2.0 if size_half else sizes
    if labels_logits.shape[0] == 0:
        labels: List[str] = []
    else:
        idx = labels_logits.argmax(axis=-1)
        labels = [class_names[int(i)] for i in idx]
    return trans.astype(np.float32), sizes_full.astype(np.float32), labels


# ---------------------------------------------------------------------- 单场景诊断
def diagnose_scene(
    centers: np.ndarray,
    sizes_full: np.ndarray,
    labels: Sequence[str],
    rel_attach_tol: float = 0.03,
) -> Dict[str, Any]:
    """对单个生成场景计算关系诊断的原始计数（聚合在 aggregate_scene_metrics）。"""
    centers = np.asarray(centers, np.float32).reshape(-1, 3)
    sizes_full = np.asarray(sizes_full, np.float32).reshape(-1, 3)
    labels = list(labels)
    n = len(labels)

    corners = _corners(centers, sizes_full) if n else np.zeros((0, 6), np.float32)
    diag = _scene_diag(corners)
    tol = rel_attach_tol * diag

    def _mask(name: str) -> np.ndarray:
        return np.array([lab == name for lab in labels], dtype=bool)

    wall_c = corners[_mask("wall")]
    roof_c = corners[_mask("roof")]

    def _attach(child: str) -> Tuple[int, int]:
        """返回 (该类构件总数, 贴墙数)。"""
        child_c = corners[_mask(child)]
        total = int(child_c.shape[0])
        if total == 0 or wall_c.shape[0] == 0:
            return total, 0
        gap = _pairwise_gap(child_c, wall_c)          # [total, n_wall]
        attached = int((gap.min(axis=1) <= tol).sum())
        return total, attached

    n_window, window_attached = _attach("window")
    n_door, door_attached = _attach("door")

    # invalid size：任一维 <= 0
    invalid_size = int((sizes_full <= 0).any(axis=1).sum()) if n else 0

    # roof-wall 关系
    roof_wall_iou: Optional[float] = None
    roof_wall_lateral: Optional[float] = None
    roof_wall_vgap: Optional[float] = None
    if roof_c.shape[0] and wall_c.shape[0]:
        r_rect = _footprint_rect(roof_c)
        w_rect = _footprint_rect(wall_c)
        roof_wall_iou = _rect_iou(r_rect, w_rect)
        # 水平中心偏移，按墙 footprint 尺度归一化
        rc = np.array([(r_rect[0] + r_rect[2]) / 2, (r_rect[1] + r_rect[3]) / 2])
        wc = np.array([(w_rect[0] + w_rect[2]) / 2, (w_rect[1] + w_rect[3]) / 2])
        w_diag = float(np.hypot(w_rect[2] - w_rect[0], w_rect[3] - w_rect[1]))
        roof_wall_lateral = float(np.linalg.norm(rc - wc) / w_diag) if w_diag > 1e-9 else None
        # 屋顶底面 z 与墙顶面 z 的间距，按场景对角线归一化
        roof_bottom = float(roof_c[:, UP_AXIS].min())
        wall_top = float(wall_c[:, UP_AXIS + 3].max())
        roof_wall_vgap = float(abs(roof_bottom - wall_top) / diag) if diag > 1e-9 else None

    return {
        "n_objects": n,
        "n_wall": int(wall_c.shape[0]),
        "n_window": n_window,
        "n_door": n_door,
        "n_roof": int(roof_c.shape[0]),
        "window_attached": window_attached,
        "door_attached": door_attached,
        "floating_window": n_window - window_attached,
        "floating_door": n_door - door_attached,
        "invalid_size": invalid_size,
        "roof_wall_footprint_iou": roof_wall_iou,
        "roof_wall_alignment_error": roof_wall_lateral,
        "roof_wall_vgap": roof_wall_vgap,
    }


# ---------------------------------------------------------------------- 跨场景聚合
def _safe_mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def aggregate_scene_metrics(scene_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把多场景的诊断计数聚合成可比较的总指标。"""
    n_scenes = len(scene_metrics)
    if n_scenes == 0:
        return {"n_scenes": 0}

    def _sum(key: str) -> int:
        return int(sum(m.get(key, 0) for m in scene_metrics))

    tot_window = _sum("n_window")
    tot_door = _sum("n_door")

    lateral = [m["roof_wall_alignment_error"] for m in scene_metrics
               if m.get("roof_wall_alignment_error") is not None]
    iou = [m["roof_wall_footprint_iou"] for m in scene_metrics
           if m.get("roof_wall_footprint_iou") is not None]
    vgap = [m["roof_wall_vgap"] for m in scene_metrics if m.get("roof_wall_vgap") is not None]

    return {
        "n_scenes": n_scenes,
        "empty_generation_rate": float(np.mean([m["n_objects"] == 0 for m in scene_metrics])),
        "mean_objects": float(np.mean([m["n_objects"] for m in scene_metrics])),
        "window_attach_rate": (_sum("window_attached") / tot_window) if tot_window else None,
        "door_attach_rate": (_sum("door_attached") / tot_door) if tot_door else None,
        "floating_window_count": _sum("floating_window"),
        "floating_door_count": _sum("floating_door"),
        "mean_floating_window": float(np.mean([m["floating_window"] for m in scene_metrics])),
        "mean_floating_door": float(np.mean([m["floating_door"] for m in scene_metrics])),
        "invalid_size_count": _sum("invalid_size"),
        "roof_wall_footprint_iou": _safe_mean(iou),
        "roof_wall_alignment_error": _safe_mean(lateral),
        "roof_wall_vgap": _safe_mean(vgap),
    }


def diagnose_post_process(
    processed: Dict[str, Any],
    class_names: Sequence[str],
    size_half: bool = False,
    rel_attach_tol: float = 0.03,
) -> Dict[str, Any]:
    """便捷入口：直接对 post_process 输出做单场景诊断。"""
    centers, sizes_full, labels = scene_from_post_process(processed, class_names, size_half)
    return diagnose_scene(centers, sizes_full, labels, rel_attach_tol=rel_attach_tol)
