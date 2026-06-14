"""config 包: 负责 YAML 配置的读写与合并。"""
from .loader import load_config, save_config

__all__ = ["load_config", "save_config"]
