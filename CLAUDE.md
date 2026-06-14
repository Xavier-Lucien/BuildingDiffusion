# CLAUDE.md - BuildingDiffusion

This file records project-specific guidance for future work in `BuildingDiffusion`.
Global user rules still apply; this file only covers this subproject.

## Project Goal

`BuildingDiffusion` is intended to become an independent, simplified research project derived from the original `BuildingBlock/` codebase.

The short-term goal is a runnable unconditional diffusion pipeline for building layout bounding boxes:

```text
preprocess -> train -> sample -> post-process -> visualize/export
```

Keep the project easy to modify. Future work is expected to change the algorithm.

## Current Data Contract

The model tensor channel order is:

```text
[translations(3), sizes(3), angles(cos, sin)(2), class_labels(C)]
```

Important details:

- `class_labels` removes the original `start` column and keeps the final `end` column.
- With 15 original labels = 13 material classes + `start` + `end`, the model `class_dim` is 14.
- Therefore default `point_dim` is `3 + 3 + 2 + 14 = 22`.
- The last class channel is used as the empty-slot/end indicator.
- `sample_num_points` should match `data.max_length`.

Do not silently change this order. If the representation changes, update `pack`, `split`, dataset encoding, post-processing, and loss slicing together.

## Independence Notes

`BuildingDiffusion` should not depend on importing the parent `BuildingBlock` package at runtime.

Current independent resources:

- `config/default.yaml`
- `dataset/building_splits.csv`
- `dataset/parse_original_data.py`
- `dataset/BoxCenterSizeLabel_all/` as raw input, if present

`config.loader.load_config()` resolves relative paths relative to the YAML file location, not the shell current working directory.

Default preprocessed data path is still:

```text
config/default.yaml -> data.dataset_directory: ../BoxCenterSizeLabelNp
```

If making the project fully self-contained, prefer moving this to a project-local path such as:

```text
data/BoxCenterSizeLabelNp
```

## Recommended Engineering Improvements

Prioritize these before large algorithm rewrites. Status as of the current branch:

1. [done] `config/smoke.yaml`.
   `layers=1`, `width=64`, `time_num=4`, `batch_size=4`, `max_steps=1`. Run with
   `python train.py config/smoke.yaml runs/smoke --max_steps 1`. `train.py` honors
   `--max_steps` (CLI) and `training.max_steps` (config) to cap optimizer steps.

2. [done] Basic tests under `tests/` (stdlib `unittest`, no pytest dependency):
   - `test_config.py`   : config path resolution
   - `test_encoding.py` : encoding shape + channel order + post_process roundtrip
   - `test_model.py`    : model forward returns scalar loss + dimension checks
   - `test_sampling.py` : sampling output passes `post_process`
   Run: `python -m unittest discover -s tests -p "test_*.py"`.

3. [done] Startup dimension checks in `model/builder.py::validate_config_dims`
   (called from `build_model`). Validates:
   - `point_dim == translation_dim + size_dim + angle_dim + class_dim`
   - `class_dim == n_classes - 1`
   - `sample_num_points == data.max_length`

4. [partial] Split large model files.
   Done: `model/gaussian_diffusion.py` (math core) + `model/layout_diffusion.py`
   (top model); `model/diffusion.py` is now a backward-compat re-export shim.
   Deferred: `model/losses.py`, `model/relations.py` (no separable unit yet —
   add when relation losses land, see Algorithm Roadmap).

5. [done] Structured training logs: `train.py` writes `<output>/metrics.jsonl`
   with phase, epoch, step, global_step, lr, and all scalar losses returned by
   the model (`loss`, `loss.bbox`, `loss.class`, `loss.liou`, ...). Relation
   losses appear automatically once the model returns them.

6. [done] Checkpoint loading is strict by default (`utils/checkpoint.py`).
   Use `--allow_partial_weights` only when intentionally changing architecture.

7. [done] New/touched files are UTF-8 with no mojibake comments.

## Algorithm Improvement Roadmap

The current model can produce locally plausible parts but globally inconsistent structures. A known failure mode is:

```text
window location looks plausible, but walls/roof are wrong or shifted,
so windows are not actually attached to walls.
```

This suggests the model is learning object-level distributions better than cross-object dependency.

### Highest-Value Algorithm Changes

1. [done] Split bbox, class, and objectness heads (`network.split_heads`, default false).

   ```text
   bbox: continuous diffusion target (v), MSE  — unchanged
   class: CE over 13 material classes          — denoiser class head
   objectness: BCE over real vs empty slot     — denoiser obj head
   ```

   Implementation:
   - `model/denoiser.py`: split_heads builds bbox-only encoder + three decoders
     (bbox / class_logits[13] / obj_logit[1]); class is NOT a network input.
   - `model/gaussian_diffusion.py::p_losses_split`: diffuses bbox only, CE on real
     slots, BCE on objectness, collision/iou uses GT objectness mask.
   - `model/layout_diffusion.py`: forward derives CE/BCE targets from the existing
     14-dim class_labels (no encoding change); sampling denoises bbox then reads
     class/obj at the final step and filters slots by `sigmoid(obj) > 0.5`.
   - Enable via `config/split_heads.yaml`; weights `lambda_class` / `lambda_obj`.
   - Sanity at init: `loss.class ≈ ln(13)=2.56`, `loss.obj ≈ ln(2)=0.69`.

2. [done] Explicit objectness — folded into #1 (obj_logit head + threshold filter),
   replaces the brittle end-label regression (`valid_mask = obj_recon <= 0`).

3. [done] Size validity (`network.loss_validity`, default false).

   `GaussianDiffusion._validity_loss`: penalize normalized size outside `[-1,1]`
   (`relu(s-1)+relu(-1-s)`, squared) so descaled sizes stay non-negative/bounded.
   - Computed on the **un-clamped** recovered x0 (`_recover_x0(..., clamp=False)`),
     otherwise the clamp hides the violation and the penalty/gradient is always 0.
   - Backstop: `dataset/encoding.py::post_process` clips descaled sizes to `>= 0`.
   - Wired into both `p_losses` (old) and `p_losses_split`. Weight `lambda_size_valid`.

4. [done] Angle vector normalization — folded into `loss_validity`.

   `loss_angle_norm = (sqrt(cos^2 + sin^2) - 1)^2` on the recovered x0; weight
   `lambda_angle_norm`. Note `post_process` already uses `arctan2`, so the final
   angle is magnitude-invariant — this term is a training regularizer, not a fix
   for the output.

5. Replace naive overlap average with collision loss.

   Current IoU-style overlap should be reshaped into:

   ```text
   loss_collision = mean(max(overlap_ratio - tolerance, 0))
   ```

   Count only distinct object pairs. Prefer `i < j` to avoid duplicate pair weighting.

6. Add slot permutation augmentation.

   Building components are a set, but Transformer slots see an order.
   Randomly permute real objects before padding during training, or enforce a stable sort.

7. Add DDIM sampling.

   Standard DDPM sampling with 1000 steps is slow. DDIM with 50-100 steps is the first sampling optimization to try.

8. Add simple conditions if unconditional generation plateaus.

   Useful low-cost conditions:
   - building footprint
   - total height or floor count
   - class count histogram
   - coarse occupancy grid

## Structural Priors / Dependency Plan

Introducing dependency is appropriate for this project. It is not against a "large model" approach.

For building generation, structural priors are useful inductive bias. The goal is not to replace the model with hard rules, but to add training signals and representations that make architectural consistency easier to learn.

### Recommended First Step: Relation Diagnostics [done]

Implemented in `utils/relation_metrics.py` (pure numpy, does not touch training).
`generate.py` now diagnoses every sampled scene and writes
`<output>/relation_metrics.json` (`summary` + `per_scene`) plus a console summary.

Metrics produced:

```text
window_attach_rate          door_attach_rate
floating_window_count       floating_door_count
roof_wall_footprint_iou     roof_wall_alignment_error (horizontal center shift)
roof_wall_vgap (roof-bottom vs wall-top, extra)
invalid_size_count          empty_generation_rate     mean_objects
```

Conventions / approximations (keep in mind when reading numbers):
- z is the vertical axis (augmentation rotates around z); footprint = x-y plane.
- Attach uses axis-aligned AABB gap (ignores per-object yaw), threshold relative
  to the scene bbox diagonal (`rel_attach_tol`, default 0.03).

These metrics make algorithm experiments comparable; treat them as relative
signals across runs, not absolute physical truth. Note: distinct from the future
`model/relations.py` (torch relation *losses* on normalized x0) — diagnostics are
numpy and run on post-processed real-coordinate scenes.

### Recommended Second Step: Relation Losses

Add relation losses on predicted `x0` during training. Start with only the most important dependencies:

```text
window attached_to wall
door attached_to wall
roof aligned_with / supported_by wall
floor below wall
```

Possible geometry losses:

- Window-wall attach:
  - window center distance to nearest wall plane should be small
  - window projected rectangle should lie inside wall extent
  - window should not float away from all walls

- Door-wall attach:
  - door center should be near a wall plane
  - door bottom should be near floor level
  - door extent should lie inside wall extent

- Roof-wall alignment:
  - roof footprint should cover or align with wall footprint
  - roof bottom or eaves should be near wall top
  - roof should not be laterally shifted far from the wall shell

Do not start with a full scene graph unless the simple losses fail.

### Recommended Third Step: Two-Stage Generation

If relation losses are not enough, use staged generation:

```text
stage 1: structural shell
  wall / roof / floor / pillar

stage 2: attachments conditioned on shell
  window / door / railing / pipe / accessory
```

This directly targets the observed failure where attachments are plausible but their parent structure is wrong.

### More Invasive Option: Parent-Anchor Representation

For attached objects such as windows and doors, predict:

```text
parent wall id
local position on wall
local size
```

Then decode absolute bbox from the parent wall.

This can strongly improve attachment consistency, but it changes the representation and requires parent assignment during preprocessing/training.

Use this only after trying relation diagnostics and relation losses.

## Suggested Experiment Order

Use this order for practical iteration:

1. [done] Add smoke config and metrics.
2. [done] Add relation diagnostics without changing training.
3. [done] Add explicit objectness + class CE while keeping bbox diffusion (`split_heads`).
4. [done] Add size and angle validity losses (`loss_validity`).
5. Add window-wall and door-wall attach losses.
6. Add roof-wall alignment loss.
7. Add slot permutation augmentation.
8. Add DDIM sampling.
9. Try two-stage shell -> attachment generation.
10. Consider parent-anchor representation for windows/doors.

## Validation Commands

After code edits:

```powershell
python -m compileall -f BuildingDiffusion
```

Run the unit tests (no real data needed, uses synthetic fixtures):

```powershell
cd BuildingDiffusion
python -m unittest discover -s tests -p "test_*.py"
```

Minimal model smoke test pattern:

```powershell
python -c "import sys, torch; sys.path.insert(0, 'BuildingDiffusion'); from config.loader import load_config; from model.builder import build_model; cfg=load_config('BuildingDiffusion/config/default.yaml'); cfg['network']['net_kwargs']['layers']=1; cfg['network']['net_kwargs']['width']=64; cfg['network']['net_kwargs']['dim']=64; cfg['network']['net_kwargs']['heads']=4; cfg['network']['diffusion_kwargs']['time_num']=4; cfg['network']['diffusion_kwargs']['loss_iou']=False; m=build_model(cfg, n_classes=15, device='cpu'); B,N=2,128; batch={'translations':torch.rand(B,N,3)*2-1,'sizes':torch.rand(B,N,3)*2-1,'angles':torch.rand(B,N,2)*2-1,'class_labels':torch.rand(B,N,14)*2-1}; out=m(batch); print(out['loss'].detach().shape, float(out['loss'].detach()))"
```

When real data exists, prefer adding and running `config/smoke.yaml` instead of manually patching config in one-liners.
