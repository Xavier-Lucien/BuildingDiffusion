"""建筑 bbox 结果可视化与 json 导出。

对应原 BuildingBlock/scripts/generate_diffusion_building.py 中
`draw_scene` + `save_json` 的精简版本，去掉了 nms/merge/3dfront 等非必要逻辑。
"""
from __future__ import annotations

import json
from typing import Any, Dict, Sequence

import numpy as np
import torch


# 建筑各类别的默认颜色（可按需替换成调色板）。
_CLASS_COLOR = {
    "accessory": (0.58, 0.00, 0.90),
    "awning":    (0.63, 0.42, 0.90),
    "balcony":   (0.39, 0.21, 0.90),
    "chimney":   (0.00, 0.32, 0.90),
    "door":      (0.90, 0.77, 0.00),
    "floor":     (0.00, 0.43, 0.90),
    "pillar":    (0.39, 0.90, 0.00),
    "pipe":      (0.90, 0.40, 0.00),
    "railing":   (0.90, 0.00, 0.29),
    "roof":      (0.90, 0.00, 0.00),
    "stair":     (0.90, 0.00, 0.70),
    "wall":      (0.00, 0.15, 0.90),
    "window":    (0.00, 0.90, 0.00),
}


def _iter_boxes(boxes: Dict[str, Any], classes: Sequence[str], size_half: bool):
    """把 [1, L, D] 形式的预测展开为 (cls_name, (xmin...zmax)) 生成器。"""
    labels = boxes["class_labels"]
    translations = boxes["translations"]
    sizes = boxes["sizes"]
    L = labels.shape[1]
    for i in range(L):
        cls_idx = int(torch.argmax(labels[0][i]))
        cls_name = classes[cls_idx]
        t = translations[0][i]
        s = sizes[0][i]
        half = s if size_half else s / 2
        xmin, xmax = float(t[0] - half[0]), float(t[0] + half[0])
        ymin, ymax = float(t[1] - half[1]), float(t[1] + half[1])
        zmin, zmax = float(t[2] - half[2]), float(t[2] + half[2])
        yield cls_name, (xmin, ymin, zmin, xmax, ymax, zmax)


def draw_scene(
    boxes: Dict[str, Any],
    classes: Sequence[str],
    save_path: str,
    size_half: bool = False,
) -> None:
    """把一组 bbox 画成 3D 缩略图。"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for cls_name, bbox in _iter_boxes(boxes, classes, size_half):
        color = _CLASS_COLOR.get(cls_name)
        if color is None:
            continue
        _draw_box(ax, bbox, color=color, poly_cls=Poly3DCollection)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[viz] saved {save_path}")


def _draw_box(ax, bbox, color, poly_cls):
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    faces = [
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin)],
        [(xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax), (xmin, ymin, zmax)],
        [(xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmin, ymax, zmax)],
        [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmin, ymax, zmax), (xmin, ymin, zmax)],
        [(xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax)],
    ]
    for face in faces:
        poly = poly_cls([face], color=color, linewidths=1, edgecolors="k", alpha=0.3)
        ax.add_collection3d(poly)


def save_scene_json(
    boxes: Dict[str, Any],
    classes: Sequence[str],
    save_path: str,
    size_half: bool = False,
) -> None:
    """把一组 bbox 保存为下游引擎可消费的 json。"""
    buildings = []
    labels = boxes["class_labels"]
    translations = boxes["translations"]
    sizes = boxes["sizes"]
    L = labels.shape[1]
    for i in range(L):
        cls_idx = int(torch.argmax(labels[0][i]))
        cls_name = classes[cls_idx]
        s = sizes[0][i]
        t = translations[0][i]
        buildings.append({
            "actor_label": f"Cube{i}",
            "materials": [f"{cls_name}Material"],
            "actor_size": s.tolist(),
            "actor_location": t.tolist(),
        })
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(buildings, f, indent=2)
    print(f"[viz] saved {save_path}")
