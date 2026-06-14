"""测试公用工具：把 BuildingDiffusion 加入 sys.path，并提供最小配置/数据。"""
import json
import os
import sys
import tempfile
from typing import Any, Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # BuildingDiffusion 目录
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 数据契约常量（与 config/default.yaml 一致）
N_CLASSES = 15        # 13 材质 + start + end
CLASS_DIM = 14        # 删除 start 后 = 材质 + end
POINT_DIM = 22        # bbox(8) + class(14)
N_CTX = 16            # 去噪器要求输入序列长度 == n_ctx；测试用小值加速


def tiny_net_config(loss_iou: bool = False, split_heads: bool = False,
                    loss_validity: bool = False) -> Dict[str, Any]:
    """构造一个最小的 network 配置，用于快速跑通前向 / 采样。"""
    return {
        "type": "diffusion_scene_layout_ddpm",
        "net_type": "dit_adaln",
        "point_dim": POINT_DIM,
        "class_dim": CLASS_DIM,
        "translation_dim": 3,
        "size_dim": 3,
        "angle_dim": 2,
        "sample_num_points": 128,
        "loss_iou_mode": "overlap_ratio",
        "size_half": False,
        "split_heads": split_heads,
        "lambda_class": 1.0,
        "lambda_obj": 1.0,
        "loss_validity": loss_validity,
        "lambda_size_valid": 1.0,
        "lambda_angle_norm": 1.0,
        "diffusion_kwargs": {
            "schedule_type": "linear",
            "beta_start": 1e-4,
            "beta_end": 2e-2,
            "time_num": 4,
            "loss_type": "mse",
            "model_mean_type": "v",
            "model_var_type": "fixedsmall",
            "loss_iou": loss_iou,
            "loss_separate": True,
        },
        "net_kwargs": {
            "dim": 64,
            "width": 64,
            "n_ctx": N_CTX,
            "channels": POINT_DIM,
            "class_dim": CLASS_DIM,
            "translation_dim": 3,
            "size_dim": 3,
            "angle_dim": 2,
            "context_dim": 0,
            "layers": 1,
            "seperate_all": True,
            "pos_embeding_way": "learned",
            "size_half": False,
        },
    }


def tiny_data_config() -> Dict[str, Any]:
    """最小 data 配置：dataset_directory 仅用于推导 stats 路径（loss_iou 关时不会被读取）。"""
    return {"dataset_directory": ".", "max_length": 128, "train_stats": "dataset_stats.txt"}


def write_stats_file(path: str) -> None:
    """写一个最小的 dataset_stats.txt（bounds 单位立方体），供 loss_iou 使用。"""
    stats = {
        "bounds_translations": [0, 0, 0, 1, 1, 1],
        "bounds_sizes": [0, 0, 0, 1, 1, 1],
        "bounds_angles": [-np.pi, np.pi],
        "class_labels": [f"c{i}" for i in range(N_CLASSES)],
        "object_types": [f"c{i}" for i in range(N_CLASSES - 2)],
        "class_frequencies": {f"c{i}": 1.0 / N_CLASSES for i in range(N_CLASSES)},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f)


def make_tempdir() -> str:
    return tempfile.mkdtemp(prefix="bd_test_")
