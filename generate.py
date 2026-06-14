"""生成脚本。对应原 scripts/generate_diffusion_building.py 的精简版（去掉 mesh / 3dfront / nms 等死代码）。

用法:
    python generate.py config/default.yaml \\
        --weight_file runs/exp1/model_01000 \\
        --output_directory samples/exp1 \\
        --n_sequences 10
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from config.loader import load_config
from dataset import build_dataset
from model.builder import build_model
from utils.checkpoint import load_weights
from utils.visualize import draw_scene, save_scene_json
from utils.relation_metrics import diagnose_post_process, aggregate_scene_metrics


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate building layouts with a trained diffusion model"
    )
    parser.add_argument(
        "config_file",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--weight_file",
        required=True,
        help="Path to the trained model weights",
    )
    parser.add_argument(
        "--allow_partial_weights",
        action="store_true",
        help="Allow loading checkpoints with missing or unexpected keys",
    )
    parser.add_argument(
        "--output_directory",
        default="./samples",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--n_sequences",
        default=10,
        type=int,
        help="Number of layouts to generate",
    )
    parser.add_argument(
        "--clip_denoised",
        action="store_true",
        help="If set, clip the denoised x0 to [-1, 1] during sampling",
    )
    args = parser.parse_args(argv)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    os.makedirs(args.output_directory, exist_ok=True)
    config = load_config(args.config_file)

    # ----- dataset (only used to get bounds / class_labels for post-processing)
    dataset = build_dataset(
        config["data"], splits=config["validation"].get("splits", ["test"])
    )
    classes = np.array(dataset.class_labels)
    print(f"[data] n_classes={dataset.n_classes}  classes={classes}")

    # ----- model
    model = build_model(config, dataset.n_classes, device=device)
    load_weights(
        model,
        args.weight_file,
        device=device,
        strict=not args.allow_partial_weights,
    )
    model.eval()

    num_points = config["network"]["sample_num_points"]
    point_dim = config["network"]["point_dim"]
    size_half = config["network"].get("size_half", False)

    # ----- sampling loop
    scene_metrics = []
    for i in range(args.n_sequences):
        with torch.no_grad():
            bbox_params = model.generate_layout(
                batch_size=1,
                num_points=num_points,
                point_dim=point_dim,
                device=device,
                clip_denoised=args.clip_denoised,
            )

        if not isinstance(bbox_params, list):
            bbox_params = [bbox_params]

        for b_idx, raw in enumerate(bbox_params):
            processed = dataset.post_process(raw)
            scene_metrics.append(
                diagnose_post_process(processed, classes, size_half=size_half)
            )
            tag = f"{i:04d}_{b_idx}"
            save_scene_json(
                processed, classes,
                save_path=os.path.join(args.output_directory, f"{tag}.json"),
            )
            draw_scene(
                processed, classes,
                save_path=os.path.join(args.output_directory, f"{tag}.png"),
                size_half=size_half,
            )

    # ----- relation diagnostics（纯度量，便于横向比较算法改动）
    summary = aggregate_scene_metrics(scene_metrics)
    metrics_path = os.path.join(args.output_directory, "relation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_scene": scene_metrics}, f,
                  ensure_ascii=False, indent=2)
    print("[metrics] relation diagnostics:")
    for k, v in summary.items():
        print(f"    {k}: {v}")
    print(f"[done] generated {args.n_sequences} sequences into {args.output_directory}")


if __name__ == "__main__":
    main(sys.argv[1:])
