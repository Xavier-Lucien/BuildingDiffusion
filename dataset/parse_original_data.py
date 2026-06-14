"""将原始 UE 导出的 BoxCenterSizeLabel json 处理成训练用的 boxes.npz。

把 BuildingBlock/ 里的 `1-json_rotate_augment.py` / `2-normUeJson.py` /
`3-json2boxnp.py` 三步合并进来，用法：

    python -m dataset.parse_original_data \\
        --input  /path/to/BoxCenterSizeLabel_all \\
        --output /path/to/BoxCenterSizeLabelNp

执行后会在 ``--output`` 下生成:
    <output>/<scene>_A{angle}_mirror{bool}/boxes.npz   # 每个样本
    <output>/dataset_stats.txt                          # 全局统计
    <output>/building_train.lst                         # 场景名列表

保持与原项目目录约定一致，可直接被 `CachedBuildingDataset` 读取。
"""

import argparse
import json
import os
import shutil
from typing import Dict, List, Sequence

import numpy as np


# ---------------------------------------------------------------------- 常量
MATERIAL_CLASS: Dict[str, int] = {
    "accessoryMaterial": 0,
    "awningMaterial": 1,
    "awaingMaterial": 1,  # 原项目里的拼写错误，兼容保留
    "balconyMaterial": 2,
    "chimneyMaterial": 3,
    "doorMaterial": 4,
    "floorMaterial": 5,
    "pillarMaterial": 6,
    "pipeMaterial": 7,
    "railingMaterial": 8,
    "roofMaterial": 9,
    "stairMaterial": 10,
    "wallMaterial": 11,
    "wallWithManyWindowMaterial": 11,
    "windowMaterial": 12,
}

CLASS_LABELS: List[str] = [
    "accessory", "awning", "balcony", "chimney", "door", "floor",
    "pillar", "pipe", "railing", "roof", "stair", "wall", "window",
    "start", "end",
]
OBJECT_TYPES: List[str] = CLASS_LABELS[:-2]

DEFAULT_ANGLES: Sequence[int] = (90, 180, 270, 360)
DEFAULT_MIRRORS: Sequence[bool] = (False, True)


# ---------------------------------------------------------------------- 工具
def _rotation_matrix_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _to_one_hot(idx: int, num_classes: int = len(CLASS_LABELS)) -> np.ndarray:
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[idx] = 1.0
    return one_hot


# ---------------------------------------------------------------------- step1
def augment(src_dir: str, dst_dir: str,
            angles: Sequence[int] = DEFAULT_ANGLES,
            mirrors: Sequence[bool] = DEFAULT_MIRRORS) -> None:
    """对每个 json 进行绕 z 轴旋转 + 镜像增强。

    每一步旋转都是增量 90°，和原脚本保持一致。
    """
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".json"))
    radians = [np.radians(90) for _ in angles]  # 每步都转 90°

    for file in files:
        with open(os.path.join(src_dir, file), "r", encoding="utf-8") as f:
            base_boxes = json.load(f)

        for whether_mirror in mirrors:
            # 每个 mirror 分支独立拷贝，避免跨分支累积
            boxes = json.loads(json.dumps(base_boxes))
            for angle, theta in zip(angles, radians):
                rot_mat = _rotation_matrix_z(theta)
                for actor in boxes:
                    if whether_mirror and angle == 90:
                        actor["actor_location"][0] = -actor["actor_location"][0]
                    loc = np.array(actor["actor_location"])
                    actor["actor_location"] = rot_mat.dot(loc).tolist()

                    size = np.array(actor["actor_size"])
                    if angle != 0:
                        actor["actor_size"][:2] = [float(size[1]), float(size[0])]

                out_name = file.replace(
                    ".json", f"_A{angle}_mirror{whether_mirror}.json"
                )
                with open(os.path.join(dst_dir, out_name), "w",
                          encoding="utf-8") as f:
                    json.dump(boxes, f, indent=4)


# ---------------------------------------------------------------------- step2
def normalize(src_dir: str, dst_dir: str) -> None:
    """把每栋楼整体归一化到 [0, 1]。"""
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".json"))

    for file in files:
        with open(os.path.join(src_dir, file), "r", encoding="utf-8") as f:
            boxes = json.load(f)

        min_coords = [float("inf")] * 3
        max_coords = [float("-inf")] * 3
        for item in boxes:
            loc, size = item["actor_location"], item["actor_size"]
            for i in range(3):
                min_coords[i] = min(min_coords[i], loc[i] - size[i] / 2)
                max_coords[i] = max(max_coords[i], loc[i] + size[i] / 2)

        range_max = max(max_coords[i] - min_coords[i] for i in range(3))
        if range_max <= 0:
            continue

        for item in boxes:
            loc, size = item["actor_location"], item["actor_size"]
            item["actor_location"] = [
                (loc[i] - min_coords[i]) / range_max
                + 0.5 * (range_max - (max_coords[i] - min_coords[i])) / range_max
                for i in range(3)
            ]
            item["actor_size"] = [s / range_max for s in size]

        with open(os.path.join(dst_dir, file), "w", encoding="utf-8") as f:
            json.dump(boxes, f, indent=4)


# ---------------------------------------------------------------------- step3
def json_to_npz(src_dir: str, dst_dir: str, data_half: bool = False) -> List[str]:
    """把归一化后的 json 转成 `<scene>/boxes.npz`。返回场景 tag 列表。"""
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".json"))

    tags: List[str] = []
    for file in files:
        with open(os.path.join(src_dir, file), "r", encoding="utf-8") as f:
            boxes = json.load(f)

        class_rows, size_rows, loc_rows = [], [], []
        for box in boxes:
            mat = box["materials"][0]
            if mat not in MATERIAL_CLASS:
                raise KeyError(f"未知材质 `{mat}` (文件 {file})")
            class_rows.append(_to_one_hot(MATERIAL_CLASS[mat]))
            size_rows.append(np.array(box["actor_size"], dtype=np.float32))
            loc_rows.append(np.array(box["actor_location"], dtype=np.float32))

        if not class_rows:
            continue

        class_np = np.stack(class_rows, axis=0)
        size_np = np.stack(size_rows, axis=0)
        if data_half:
            size_np = size_np / 2.0
        loc_np = np.stack(loc_rows, axis=0)
        angle_np = np.zeros([class_np.shape[0], 1], dtype=np.float32)

        tag = file[:-5]
        tags.append(tag)
        scene_dir = os.path.join(dst_dir, tag)
        os.makedirs(scene_dir, exist_ok=True)
        np.savez(
            os.path.join(scene_dir, "boxes.npz"),
            class_labels=class_np,
            translations=loc_np,
            sizes=size_np,
            angles=angle_np,
        )
    return tags


# ---------------------------------------------------------------------- stats
def write_default_stats(dst_dir: str, filename: str = "dataset_stats.txt") -> None:
    """写入和原项目一致的 dataset_stats.txt（归一化边界 [0,1]）。"""
    stats = {
        "bounds_translations": [0, 0, 0, 1, 1, 1],
        "bounds_sizes": [0, 0, 0, 1, 1, 1],
        "bounds_angles": [-np.pi, np.pi],
        "class_labels": CLASS_LABELS,
        "object_types": OBJECT_TYPES,
        "class_frequencies": {},
        "class_order": {},
        "count_furniture": {},
    }
    with open(os.path.join(dst_dir, filename), "w", encoding="utf-8") as f:
        json.dump(stats, f)


def write_scene_list(tags: Sequence[str], dst_path: str) -> None:
    """把所有场景 tag 写到 lst 文件，方便 splits 划分。"""
    with open(dst_path, "w", encoding="utf-8") as f:
        for t in tags:
            f.write(t + "\n")


# ---------------------------------------------------------------------- main
def run(
    input_dir: str,
    output_dir: str,
    tmp_dir: str = "",
    data_half: bool = False,
    keep_tmp: bool = False,
) -> None:
    """一键完成 augment → normalize → to_npz 三步。"""
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = tmp_dir or os.path.join(output_dir, "_tmp_preprocess")
    aug_dir = os.path.join(tmp_dir, "augment")
    norm_dir = os.path.join(tmp_dir, "normalized")
    os.makedirs(aug_dir, exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)

    print(f"[1/3] augment  : {input_dir} -> {aug_dir}")
    augment(input_dir, aug_dir)

    print(f"[2/3] normalize: {aug_dir} -> {norm_dir}")
    normalize(aug_dir, norm_dir)

    print(f"[3/3] to npz   : {norm_dir} -> {output_dir}")
    tags = json_to_npz(norm_dir, output_dir, data_half=data_half)

    write_default_stats(output_dir)
    write_scene_list(tags, os.path.join(output_dir, "building_train.lst"))
    print(f"done. {len(tags)} samples, stats -> {output_dir}/dataset_stats.txt")

    if not keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="原始 UE 导出 json -> 训练用 boxes.npz"
    )
    p.add_argument("--input", required=True,
                   help="原始 BoxCenterSizeLabel json 所在目录")
    p.add_argument("--output", required=True,
                   help="输出目录，对应 dataset.dataset_directory")
    p.add_argument("--tmp", default="",
                   help="中间文件目录，默认 <output>/_tmp_preprocess")
    p.add_argument("--data-half", action="store_true",
                   help="尺寸除以 2（原项目 data_half 选项）")
    p.add_argument("--keep-tmp", action="store_true",
                   help="保留中间 augment / normalized json")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run(
        input_dir=args.input,
        output_dir=args.output,
        tmp_dir=args.tmp,
        data_half=args.data_half,
        keep_tmp=args.keep_tmp,
    )


if __name__ == "__main__":
    main()
