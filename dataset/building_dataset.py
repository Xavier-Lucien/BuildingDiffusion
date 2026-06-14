import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from .splits import CSVSplitsBuilder
from .encoding import encode_sample, post_process


class CachedBuildingDataset(Dataset):
    """读取预处理好的 `boxes.npz` (class_labels / translations / sizes / angles)。

    目录约定 (来自原项目 BoxCenterSizeLabelNp):
        <base_dir>/<scene_tag>/boxes.npz
        <base_dir>/<train_stats_file>            # 全局统计 json
    """

    def __init__(
        self,
        base_dir: str,
        scene_ids: Sequence[str],
        train_stats: str = "dataset_stats.txt",
        max_length: int = 128,
        encoding_type: str = "cached_diffusion_cosin_angle_wocm_prmAll",
        pad_zero: bool = False,
    ):
        self.base_dir = base_dir
        self.max_length = max_length
        self.encoding_type = encoding_type
        self.pad_zero = pad_zero

        self._parse_train_stats(os.path.join(base_dir, train_stats))

        all_tags = sorted(os.listdir(base_dir))
        id_set = set(scene_ids)
        candidate_tags = [t for t in all_tags if t.split("_A")[0] in id_set]
        self._tags: List[str] = []
        self._paths: List[str] = []
        self.skipped_too_long = 0
        for tag in candidate_tags:
            path = os.path.join(base_dir, tag, "boxes.npz")
            if not os.path.isfile(path):
                continue
            with np.load(path) as raw:
                if raw["class_labels"].shape[0] > max_length:
                    self.skipped_too_long += 1
                    continue
            self._tags.append(tag)
            self._paths.append(path)

        if not self._paths:
            raise RuntimeError(
                f"No usable boxes.npz files found in {base_dir} for splits {list(scene_ids)}"
            )

    # ------------------------------------------------------------------ 元数据
    def _parse_train_stats(self, stats_path: str) -> None:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        self._class_labels: List[str] = stats["class_labels"]
        self._object_types: List[str] = stats["object_types"]
        self._class_frequencies: Dict[str, float] = stats["class_frequencies"]
        # 用于归一化/反归一化的边界
        self.bounds = {
            "translations": (
                np.array(stats["bounds_translations"][:3]),
                np.array(stats["bounds_translations"][3:]),
            ),
            "sizes": (
                np.array(stats["bounds_sizes"][:3]),
                np.array(stats["bounds_sizes"][3:]),
            ),
            "angles": (
                np.array(stats["bounds_angles"][0]),
                np.array(stats["bounds_angles"][1]),
            ),
        }

    @property
    def class_labels(self) -> List[str]:
        return self._class_labels

    @property
    def object_types(self) -> List[str]:
        return self._object_types

    @property
    def n_classes(self) -> int:
        return len(self._class_labels)

    @property
    def n_object_types(self) -> int:
        return len(self._object_types)

    @property
    def feature_size(self) -> int:
        # 编码会删除 start 标签：class_dim = n_classes - 1
        # point_dim = class_dim + translations(3) + sizes(3) + angles(cos/sin=2)
        return (self.n_classes - 1) + 3 + 3 + 2

    # ------------------------------------------------------------------ pytorch api
    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        with np.load(self._paths[idx]) as raw:
            sample = {
                "class_labels": raw["class_labels"],
                "translations": raw["translations"],
                "sizes": raw["sizes"],
                "angles": raw["angles"],
            }
        return encode_sample(
            sample, self.bounds, self.max_length, self.encoding_type, pad_zero=self.pad_zero
        )

    # ------------------------------------------------------------------ 推理后处理
    def post_process(self, pred: Dict[str, Any]) -> Dict[str, Any]:
        """把模型输出的归一化张量反归一化回原始坐标系。"""
        return post_process(pred, self.bounds)

    # ------------------------------------------------------------------ collate
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """默认 stack；因为 encode_sample 已把每个样本 pad 到 max_length。"""
        import torch

        out: Dict[str, Any] = {}
        for k in batch[0].keys():
            vals = [b[k] for b in batch]
            if isinstance(vals[0], np.ndarray):
                out[k] = torch.from_numpy(np.stack(vals, axis=0)).float()
            else:
                out[k] = vals
        return out


# ---------------------------------------------------------------------- builder
def build_dataset(
    data_cfg: Dict[str, Any],
    splits: Sequence[str],
) -> CachedBuildingDataset:
    """根据配置构建数据集。"""
    splits_builder = CSVSplitsBuilder(data_cfg["annotation_file"])
    scene_ids = splits_builder.get_splits(splits)

    return CachedBuildingDataset(
        base_dir=data_cfg["dataset_directory"],
        scene_ids=scene_ids,
        train_stats=data_cfg.get("train_stats", "dataset_stats.txt"),
        max_length=data_cfg.get("max_length", 128),
        encoding_type=data_cfg.get("encoding_type", "cached_diffusion_cosin_angle_wocm_prmAll"),
        pad_zero=data_cfg.get("pad_zero", False),
    )
