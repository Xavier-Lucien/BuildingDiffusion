import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.loader import load_config
from dataset import build_dataset
from model.builder import build_model, build_optimizer, build_lr_scheduler, apply_lr
from utils.checkpoint import save_checkpoint, load_latest_checkpoint, load_weights


def _append_metrics(path: str, record: dict) -> None:
    """把一条训练/验证指标以 JSON Lines 形式追加到 metrics.jsonl。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _scalar_losses(out: dict) -> dict:
    """从 model forward 输出里抽取所有标量损失（含 relation losses，若有）。"""
    metrics = {}
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and v.dim() == 0:
            metrics[k] = float(v.detach())
    return metrics


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a diffusion model on building bounding boxes"
    )
    parser.add_argument(
        "config_file",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "output_directory",
        help="Path to the output directory (weights / logs / config copy)",
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="Path to a pretrained model to initialize from",
    )
    parser.add_argument(
        "--allow_partial_weights",
        action="store_true",
        help="Allow loading checkpoints with missing or unexpected keys",
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)",
    )
    parser.add_argument(
        "--seed", type=int, default=27, help="Seed for the PRNG"
    )
    parser.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help="Cap total optimizer steps (overrides training.max_steps); "
             "useful for smoke validation",
    )
    args = parser.parse_args(argv)

    # ----- setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    os.makedirs(args.output_directory, exist_ok=True)
    shutil.copy(args.config_file, os.path.join(args.output_directory, "config.yaml"))

    config = load_config(args.config_file)

    # ----- datasets
    train_set = build_dataset(
        config["data"], splits=config["training"].get("splits", ["train", "val"])
    )
    val_set = build_dataset(
        config["data"], splits=config["validation"].get("splits", ["test"])
    )
    print(
        f"[data] train={len(train_set)}  val={len(val_set)}  "
        f"n_classes={train_set.n_classes}"
    )
    if train_set.skipped_too_long or val_set.skipped_too_long:
        print(
            f"[data] skipped too long: "
            f"train={train_set.skipped_too_long}  val={val_set.skipped_too_long}"
        )

    train_loader = DataLoader(
        train_set,
        batch_size=config["training"].get("batch_size", 64),
        shuffle=True,
        num_workers=0,
        collate_fn=train_set.collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["validation"].get("batch_size", 1),
        shuffle=False,
        num_workers=0,
        collate_fn=val_set.collate_fn,
    )

    # ----- model / optimizer
    model = build_model(config, train_set.n_classes, device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params = {n_params / 1e6:.2f} M")

    optimizer = build_optimizer(config, model.parameters())
    lr_scheduler = build_lr_scheduler(config)

    # ----- resume / init weights
    if args.weight_file is not None:
        load_weights(
            model,
            args.weight_file,
            device=device,
            strict=not args.allow_partial_weights,
        )

    start_epoch, _ = load_latest_checkpoint(
        model,
        optimizer,
        args.output_directory,
        device=device,
        strict=not args.allow_partial_weights,
    )
    if args.continue_from_epoch > start_epoch:
        start_epoch = args.continue_from_epoch

    # ----- training loop
    epochs = config["training"].get("epochs", 100)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)
    max_grad_norm = config["training"].get("max_grad_norm", 0)
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = config["training"].get("max_steps")

    metrics_path = os.path.join(args.output_directory, "metrics.jsonl")
    global_step = 0
    stop = False

    for epoch in range(start_epoch, epochs):
        apply_lr(lr_scheduler, optimizer, epoch)
        lr = optimizer.param_groups[0]["lr"]
        model.train()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            out = model(batch)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            global_step += 1

            if step % 50 == 0:
                print(
                    f"[epoch {epoch:5d}][step {step:5d}] "
                    f"loss={loss.item():.4f}  time={time.time() - t0:.1f}s"
                )
                t0 = time.time()
                record = {
                    "phase": "train",
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "lr": lr,
                    **_scalar_losses(out),
                }
                _append_metrics(metrics_path, record)

            if max_steps is not None and global_step >= max_steps:
                stop = True
                break

        if not stop and epoch > 0 and epoch % save_every == 0:
            save_checkpoint(model, optimizer, epoch, args.output_directory)

        if not stop and epoch > 0 and epoch % val_every == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_loader:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)
                    losses.append(model(batch)["loss"].item())
            if losses:
                mean_loss = sum(losses) / len(losses)
                print(f"[val   {epoch:5d}] mean loss={mean_loss:.4f}")
                _append_metrics(
                    metrics_path,
                    {"phase": "val", "epoch": epoch, "global_step": global_step,
                     "loss": mean_loss},
                )

        if stop:
            print(f"[done] reached max_steps={max_steps}, stop training")
            break


if __name__ == "__main__":
    main(sys.argv[1:])
