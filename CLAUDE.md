# CLAUDE.md - BuildingDiffusion

Project-specific guidance for future work in `BuildingDiffusion`.
Global user rules still apply; this file only covers this repository.

## Goal

`BuildingDiffusion` is a standalone, simplified research project derived from the
original `BuildingBlock/` codebase.

Current target:

```text
preprocess -> train -> sample -> post-process -> visualize/export
```

Keep the code easy to change. Future work is expected to modify the algorithm.

## Data Contract

The model tensor channel order is fixed:

```text
[translations(3), sizes(3), angles(cos, sin)(2), class_labels(C)]
```

Key assumptions:

- `class_labels` removes the original `start` column and keeps the final `end`
  column.
- With 15 original labels (`13 material classes + start + end`), `class_dim`
  is 14.
- Default `point_dim` is `3 + 3 + 2 + 14 = 22`.
- The final class channel is the empty-slot/end indicator.
- `sample_num_points` must match `data.max_length`.

Do not silently change this representation. If it changes, update packing,
splitting, dataset encoding, post-processing, dimension checks, and loss slicing
together.

## Independence

This repository should not import the parent `BuildingBlock` package at runtime.

Important paths:

- `config/default.yaml`
- `config/smoke.yaml`
- `config/split_heads.yaml`
- `dataset/building_splits.csv`
- `dataset/parse_original_data.py`
- `dataset/BoxCenterSizeLabel_all/` raw input, if present

`config.loader.load_config()` resolves relative paths from the YAML file
location, not from the shell working directory.

The default preprocessed data path is still configured as:

```text
data.dataset_directory: ../BoxCenterSizeLabelNp
```

If making the project fully self-contained, prefer moving this to a project-local
path such as:

```text
data/BoxCenterSizeLabelNp
```

## Current Implementation

Implemented and expected to stay working:

- `config/smoke.yaml` for tiny training runs.
- Stdlib `unittest` tests under `tests/`.
- Startup dimension checks in `model/builder.py::validate_config_dims`.
- Split model files:
  - `model/gaussian_diffusion.py`
  - `model/layout_diffusion.py`
  - `model/diffusion.py` backward-compatible re-export shim
- Structured training logs at `<output>/metrics.jsonl`.
- Strict checkpoint loading by default in `utils/checkpoint.py`; use
  `--allow_partial_weights` only for intentional architecture changes.
- UTF-8 text in new or touched files.

Implemented algorithm options:

- `network.split_heads`
  - bbox diffusion target
  - class CE over 13 material classes
  - objectness BCE over real vs empty slots
- `network.loss_validity`
  - size validity loss
  - angle-vector norm regularization
  - post-processing clips descaled sizes to `>= 0`
- Relation diagnostics in `utils/relation_metrics.py`; `generate.py` writes
  `<output>/relation_metrics.json`.

## Known Modeling Issue

The model can produce locally plausible objects but globally inconsistent
structures, for example windows that look plausible while walls or roofs are
misaligned.

This points to weak cross-object dependency learning. Prefer improving relation
signals before making broad architecture rewrites.

## Algorithm Priorities

Recommended order for future experiments:

1. Add window-wall and door-wall attachment losses.
2. Add roof-wall alignment/support losses.
3. Replace naive overlap averaging with a collision loss:

   ```text
   loss_collision = mean(max(overlap_ratio - tolerance, 0))
   ```

   Count each distinct object pair once, preferably with `i < j`.

4. Add slot permutation augmentation or enforce a stable slot order.
5. Add DDIM sampling for faster generation.
6. Add lightweight conditioning if unconditional generation plateaus:
   - footprint
   - total height or floor count
   - class-count histogram
   - coarse occupancy grid
7. Consider two-stage generation:

   ```text
   stage 1: structural shell (wall / roof / floor / pillar)
   stage 2: attachments conditioned on shell (window / door / railing / pipe)
   ```

8. Consider parent-anchor representation for attached objects only after simpler
   relation losses fail.

## Relation Diagnostics

`utils/relation_metrics.py` is numpy-only and does not affect training.

Metrics include:

```text
window_attach_rate
door_attach_rate
floating_window_count
floating_door_count
roof_wall_footprint_iou
roof_wall_alignment_error
roof_wall_vgap
invalid_size_count
empty_generation_rate
mean_objects
```

Conventions:

- `z` is vertical.
- Footprint is the `x-y` plane.
- Attachment uses axis-aligned AABB gap and ignores object yaw.
- Treat diagnostics as relative signals across runs, not absolute physical truth.

## Validation

From the repository root:

```powershell
python -m compileall -f .
python -m unittest discover -s tests -p "test_*.py"
```

Minimal training smoke run, when data/config paths are available:

```powershell
python train.py config/smoke.yaml runs/smoke --max_steps 1
```

Minimal model-only smoke check:

```powershell
python -c "import torch; from config.loader import load_config; from model.builder import build_model; cfg=load_config('config/default.yaml'); cfg['network']['net_kwargs']['layers']=1; cfg['network']['net_kwargs']['width']=64; cfg['network']['net_kwargs']['dim']=64; cfg['network']['net_kwargs']['heads']=4; cfg['network']['diffusion_kwargs']['time_num']=4; cfg['network']['diffusion_kwargs']['loss_iou']=False; m=build_model(cfg, n_classes=15, device='cpu'); B,N=2,128; batch={'translations':torch.rand(B,N,3)*2-1,'sizes':torch.rand(B,N,3)*2-1,'angles':torch.rand(B,N,2)*2-1,'class_labels':torch.rand(B,N,14)*2-1}; out=m(batch); print(out['loss'].detach().shape, float(out['loss'].detach()))"
```
