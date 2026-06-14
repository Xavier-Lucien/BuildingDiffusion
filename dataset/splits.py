"""按 CSV 定义的 train/val/test 切分工具。

对应原 BuildingBlock/scene_synthesis/datasets/splits_builder.py 的精简版。
"""
from __future__ import annotations

import csv
from typing import List, Sequence


class CSVSplitsBuilder:
    """从 CSV 读取 scene_id -> split 的对应关系。

    CSV 头含 `id` 和 `split` 两列即可。
    """

    def __init__(self, annotation_file: str):
        self.annotation_file = annotation_file
        self._rows = self._load()

    def _load(self):
        rows = []
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows

    def get_splits(self, splits: Sequence[str]) -> List[str]:
        """返回属于给定 split 的所有 scene id。"""
        target = set(splits)
        return [r["id"] for r in self._rows if r.get("split") in target]
