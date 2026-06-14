"""兼容层：扩散模型已拆分为两个文件，本模块保留旧的导入路径。

- 扩散数学核心 -> :mod:`model.gaussian_diffusion`（``GaussianDiffusion``）
- 顶层布局模型 -> :mod:`model.layout_diffusion`（``DiffusionBuildingBlock``）

数据通道约定见对应模块。新代码请直接从拆分后的模块导入。
"""
from __future__ import annotations

from .gaussian_diffusion import (
    GaussianDiffusion,
    ModelPrediction,
    get_betas,
    identity,
)
from .layout_diffusion import DiffusionBuildingBlock

__all__ = [
    "GaussianDiffusion",
    "ModelPrediction",
    "get_betas",
    "identity",
    "DiffusionBuildingBlock",
]
