"""YAML 配置加载/保存工具。"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml

try:
    from yaml import CLoader as _YamlLoader  # 更快的 C 实现
except ImportError:  # pragma: no cover
    from yaml import Loader as _YamlLoader  # type: ignore


def load_config(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件并返回 dict。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=_YamlLoader)

    _resolve_paths(config, os.path.dirname(os.path.abspath(path)))
    return config


def _resolve_paths(config: Dict[str, Any], config_dir: str) -> None:
    """Resolve project paths relative to the YAML file, not the shell cwd."""

    def _resolve(value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return value
        if os.path.isabs(value):
            return value
        return os.path.normpath(os.path.join(config_dir, value))

    data_cfg = config.get("data")
    if isinstance(data_cfg, dict):
        for key in ("dataset_directory", "annotation_file"):
            if key in data_cfg:
                data_cfg[key] = _resolve(data_cfg[key])

    net_cfg = config.get("network")
    if isinstance(net_cfg, dict):
        diff_cfg = net_cfg.get("diffusion_kwargs")
        if isinstance(diff_cfg, dict) and diff_cfg.get("train_stats_file"):
            diff_cfg["train_stats_file"] = _resolve(diff_cfg["train_stats_file"])


def save_config(config: Dict[str, Any], path: str) -> None:
    """把配置 dict 另存为 YAML，用于保留实验现场。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
