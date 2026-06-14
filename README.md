
# BuildingDiffusion

这是对原 `BuildingBlock` 项目的**结构重构版**，目标是把一个堆满死代码的研究代码库，
整理成一个**目录清晰、单一职责**的最小骨架，方便阅读与二次开发。

整体组织沿用原项目 `scripts/` 里**扁平脚本式**的命令行风格：
`train.py` 和 `generate.py` 直接是 `main(argv)` 入口，线性走流程，没有多余的 class 包装。

## 目录结构

```
BuildingDiffusion/
├── train.py              # 训练入口脚本： python train.py <config> <output_dir> [...]
├── generate.py           # 生成入口脚本： python generate.py <config> --weight_file ...
├── config/               # YAML 配置 + 加载器
├── dataset/              # 数据集（原始 json 预处理 + box npz 加载 + 编码）
├── model/                # Diffusion 模型（Transformer 去噪器 + GaussianDiffusion）
└── utils/                # 通用工具： checkpoint 保存/恢复、可视化
```

## 模块职责

| 模块 | 负责什么 | 不负责什么 |
|------|----------|-----------|
| `config`  | 读写 YAML | 业务逻辑 |
| `dataset` | 原始 json 预处理、读取缓存 `boxes.npz`、转为张量 | — |
| `model`   | 扩散过程、去噪网络 | 训练循环 |
| `utils`   | checkpoint I/O、可视化/JSON 导出 | 训练 / 生成流程本身 |

训练循环本身**直接写在 [train.py](train.py) 的 `main()` 里**，不再单独抽一个 `Trainer` 类；
生成循环同理，直接写在 [generate.py](generate.py) 的 `main()` 里。

## 快速开始

```bash
# 数据预处理：UE 原始 json -> boxes.npz
python -m dataset.parse_original_data \
    --input  /path/to/BoxCenterSizeLabel_all \
    --output /path/to/BoxCenterSizeLabelNp

# 训练
python train.py config/default.yaml runs/exp1

# 从已有权重继续训练
python train.py config/default.yaml runs/exp1 --weight_file runs/exp1/model_01000

# 生成
python generate.py config/default.yaml \
    --weight_file runs/exp1/model_01000 \
    --output_directory samples/exp1 \
    --n_sequences 10
```

## 重构与原项目的映射

| 原路径 | 新路径 |
|-------|--------|
| `config/uncond/diffusion_building_DIT.yaml` | `config/default.yaml` |
| `scene_synthesis/datasets/building_blocks.py` | `dataset/building_dataset.py` |
| `1-json_rotate_augment.py` + `2-normUeJson.py` + `3-json2boxnp.py` | `dataset/parse_original_data.py` |
| `scene_synthesis/datasets/splits_builder.py` | `dataset/splits.py` |
| `scene_synthesis/datasets/threed_front_dataset.py`（编码部分） | `dataset/encoding.py` |
| `scene_synthesis/networks/diffusion_scene_layout_ddpm.py` | `model/diffusion.py` + `model/builder.py` |
| `scene_synthesis/networks/denoise_net_transformer_adaln.py` | `model/denoiser.py` |
| `scripts/train_diffusion_building_ddp.py` + `scripts/training_utils.py` | `train.py` + `utils/checkpoint.py` |
| `scripts/generate_diffusion_building.py` | `generate.py` + `utils/visualize.py` |

## 相比原项目，砍掉了什么

- `DataParallel` / DDP 多卡旁路
- `wandb` / `swanlab` / `StatsLogger` 等日志旁路（留 `print`）
- `multiprocessing.Manager` 数据 cache
- 硬编码的英文 text prompt、无条件/有条件两套采样分支
- `nms` / mesh 导出 / `3D-FRONT` / `simple_3dviz` 的场景渲染
- `scripts/utils.py` 里 22KB 的 floor plan / textured object 相关代码

保留了**训练-评估-采样-导出**最短闭环所需的最小代码。

## 当前状态

本仓库仅提供**骨架**。核心算法实现保留接口与 TODO 注释，
可按需从原 `BuildingBlock/` 中迁移具体实现代码。
