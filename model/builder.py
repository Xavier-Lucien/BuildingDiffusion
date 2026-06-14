import math
import os
from typing import Any, Dict, Iterable
import torch
from .diffusion import DiffusionBuildingBlock

# ---------------------------------------------------------------- dimension checks
def validate_config_dims(config: Dict[str, Any], n_classes: int) -> None:
    """启动时校验配置维度自洽，尽早暴露通道顺序/数据契约不一致。

    校验项（对应 CLAUDE.md 数据契约）：
      - point_dim == translation_dim + size_dim + angle_dim + class_dim
      - class_dim == n_classes - 1   （编码删除 start 标签）
      - sample_num_points == data.max_length
    """
    net_cfg = config["network"]
    td = net_cfg.get("translation_dim", 3)
    sd = net_cfg.get("size_dim", 3)
    ad = net_cfg.get("angle_dim", 2)
    cd = net_cfg.get("class_dim", 14)
    point_dim = net_cfg["point_dim"]

    expected_point_dim = td + sd + ad + cd
    if point_dim != expected_point_dim:
        raise ValueError(
            f"point_dim={point_dim} 与 translation_dim+size_dim+angle_dim+class_dim"
            f"={td}+{sd}+{ad}+{cd}={expected_point_dim} 不一致"
        )

    if cd != n_classes - 1:
        raise ValueError(
            f"class_dim={cd} 应等于 n_classes-1={n_classes - 1}"
            f"（编码删除 start 标签，保留材质 + end）"
        )

    if net_cfg.get("split_heads", False) and cd - 1 < 1:
        raise ValueError(
            f"split_heads=True 需要材质数 n_materials=class_dim-1={cd - 1} >= 1"
        )

    data_cfg = config.get("data")
    if isinstance(data_cfg, dict) and "max_length" in data_cfg:
        max_length = data_cfg["max_length"]
        sample_num_points = net_cfg.get("sample_num_points", max_length)
        if sample_num_points != max_length:
            raise ValueError(
                f"sample_num_points={sample_num_points} 应等于 data.max_length={max_length}"
            )


# ---------------------------------------------------------------- model
def build_model(config: Dict[str, Any], n_classes: int, device: str = "cpu") -> DiffusionBuildingBlock:
    """根据配置构建扩散模型。"""
    net_cfg = config["network"]
    if net_cfg["type"] != "diffusion_scene_layout_ddpm":
        raise NotImplementedError(f"network type {net_cfg['type']} is not supported")

    validate_config_dims(config, n_classes)

    # split-heads 开关注入到 denoiser 的 net_kwargs，保持单一来源
    net_cfg.setdefault("net_kwargs", {})["split_heads"] = net_cfg.get("split_heads", False)

    # iou loss 需要 train_stats（含 bounds）。若 diffusion_kwargs 未显式给出，
    # 则从 data 配置推导出 <dataset_directory>/<train_stats> 路径。
    train_stats_file = net_cfg.get("diffusion_kwargs", {}).get("train_stats_file")
    if train_stats_file is None and "data" in config:
        data_cfg = config["data"]
        train_stats_file = os.path.join(
            data_cfg["dataset_directory"], data_cfg.get("train_stats", "dataset_stats.txt")
        )

    model = DiffusionBuildingBlock(
        n_classes=n_classes,
        net_config=net_cfg,
        train_stats_file=train_stats_file,
        device=device,
    )
    return model.to(device)


# ---------------------------------------------------------------- optimizer
def build_optimizer(config: Dict[str, Any], parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    name = train_cfg.get("optimizer", "Adam")
    lr = train_cfg.get("lr", 2e-4)
    wd = train_cfg.get("weight_decay", 0.0)

    if name == "Adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    if name == "SGD":
        return torch.optim.SGD(
            parameters, lr=lr, momentum=train_cfg.get("momentum", 0.9), weight_decay=wd
        )
    raise NotImplementedError(f"optimizer {name} is not supported")


# ---------------------------------------------------------------- lr schedule
class _StepLR:
    def __init__(self, lr: float, step: int, decay: float):
        self.lr, self.step, self.decay = lr, step, decay

    def __call__(self, epoch: int) -> float:
        return self.lr * (self.decay ** (epoch // self.step))


class _WarmupCosineLR:
    def __init__(self, lr: float, min_lr: float, warmup: int, total: int):
        self.lr, self.min_lr, self.warmup, self.total = lr, min_lr, warmup, total

    def __call__(self, epoch: int) -> float:
        if epoch <= self.warmup:
            return self.lr
        ratio = (epoch - self.warmup) / max(self.total - self.warmup, 1)
        return self.min_lr + (self.lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * ratio))


def build_lr_scheduler(config: Dict[str, Any]):
    """返回一个 epoch -> lr 的可调用对象。"""
    train_cfg = config["training"]
    name = train_cfg.get("schedule", "step").lower()

    if name == "step":
        return _StepLR(
            lr=train_cfg.get("lr", 2e-4),
            step=train_cfg.get("lr_step", 10000),
            decay=train_cfg.get("lr_decay", 0.5),
        )
    if name in ("warmupcosine", "warmup_cosine"):
        return _WarmupCosineLR(
            lr=train_cfg.get("lr", 2e-4),
            min_lr=train_cfg.get("min_lr", 1e-6),
            warmup=train_cfg.get("warmup_epochs", 10),
            total=train_cfg.get("total_epochs", 2000),
        )
    raise NotImplementedError(f"schedule {name} is not supported")


def apply_lr(scheduler, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    """把 scheduler 计算出的 lr 写入 optimizer。"""
    lr = scheduler(epoch)
    for g in optimizer.param_groups:
        g["lr"] = lr
