# Task 07 - Repository Cleanup and Spec Refresh

## Context

The active CNN segmentation direction has changed from the earlier 5-class workflow to the latest `segformer_b0_run3` 7-class workflow.

The current active CNN model predicts:

```txt
0 - background
1 - floor
2 - wall
3 - window
4 - door_arc
5 - door_leaf
6 - door_origin
```

Older CNN runs and their related generated artifacts are no longer active. However, their checkpoint folders should remain available so previous models can be reused if necessary.

The vectorization output currently stored under:

```txt
C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\outputs\vectorization\v008
```

was generated from previous run1/run2 model outputs. Those failed vectorization outputs are still useful for inspection and comparison and must be kept.

## Objective

Clean up obsolete CNN-related repository artifacts and refresh the project specs so the spec files are a reliable blueprint for the current repository state.

This task has two goals:

1. Remove inactive CNN artifacts that are no longer useful.
2. Update active spec files so they reflect the current 7-class CNN pipeline and the current vector-to-raster mask requirements.

## Cleanup Scope

Previous CNN models are no longer active.

Remove obsolete CNN-related files for inactive model runs when they are not needed for reuse, including:

```txt
cached features
training histories outside preserved checkpoint folders
old preview images
old run logs
intermediate CNN output artifacts
inactive generated files tied only to previous CNN experiments
```

Do not remove the latest active run3 files.

The latest active CNN run is:

```txt
segformer_b0_run3
```

The latest active run3 artifacts include:

```txt
configs/train_segformer_b0_run3.yaml
checkpoints/segformer_b0_run3
runs/segformer_b0_run3
features/segformer_b0_run3
splits/train.json
splits/val.json
splits/test.json
```

If a file is required to train, evaluate, inspect, or reuse `segformer_b0_run3`, keep it.

## Checkpoint Preservation Rule

Keep the whole checkpoint folder for each previous CNN model.

Do not trim old checkpoint folders down to only `.pt` files.

Preserve old checkpoint folders including associated metadata such as:

```txt
best.pt
latest.pt
training_summary.json
class_weights.json
training_history.csv
```

This allows previous models to be reused, inspected, or compared later.

## Vectorization Output Preservation Rule

Keep all current vectorization outputs under:

```txt
outputs/vectorization/v008
```

This includes failed outputs.

Do not delete failed run1/run2 vectorization outputs. They are still useful as historical evidence for the vectorization stage.

## Spec Refresh Requirements

Update active spec files so they describe the current repository accurately.

The spec files should be the true blueprint of the repository. After this task, a developer should be able to read the active specs and understand the current intended pipeline without being misled by obsolete 5-class assumptions.

At minimum, review and update:

```txt
specs/spec_v003_semantic_mask_generation.md
specs/spec_v005_segformer_train.md
specs/spec_v007_component_primitives.md
specs/spec_v008_mask_to_vector.md
workflow.md
readme.md
```

Only update `workflow.md` and `readme.md` where needed to keep them consistent with the current active specs.

## 7-Class Mask Generation Requirements

The vector-to-raster semantic mask generation instructions must reflect the current 7-class target:

```txt
background
floor
wall
window
door_arc
door_leaf
door_origin
```

Remove or supersede active instructions that describe the current model as the older 5-class mapping:

```txt
background
wall
opening
room
icon
```

The older 5-class mapping may remain only in explicitly outdated or historical documents.

The active mask-generation spec must clearly explain:

1. `model.svg` is the semantic source of truth.
2. `model_clean.png` is the SVG-rendered clean raster input.
3. `F1_scaled.png` is the original CubiCasa raster input.
4. Both `F1_scaled.png` and `model_clean.png` may be separate training rows sharing the same target mask.
5. Door-related SVG geometry should be rasterized into separate door classes where available:

```txt
door_arc
door_leaf
door_origin
```

6. Windows should be distinct from door classes.
7. Floor evidence should be a semantic class, not the old room/floor ambiguity.

## SegFormer Training Spec Requirements

The active SegFormer training spec must clearly identify `segformer_b0_run3` as the current active CNN model.

It must describe:

```txt
7 output classes
frozen SegFormer-B0 backbone
custom decoder
cached features in features/segformer_b0_run3
training outputs in checkpoints/segformer_b0_run3
preview outputs in runs/segformer_b0_run3
grouped metrics for svg_rendered_clean and original_raster inputs
```

Do not describe run1 or run2 as active.

Run1 and run2 may be mentioned only as historical or inactive model runs.

## Vectorization Spec Requirements

Review vectorization specs for assumptions tied to the older 5-class model.

The vectorization specs may still describe previous failed outputs under `outputs/vectorization/v008`, but active forward-looking instructions must account for the 7-class CNN output.

At minimum, the vectorization direction should acknowledge that the active CNN now distinguishes:

```txt
window
door_arc
door_leaf
door_origin
```

instead of a single generic `opening` class.

If the current vectorization implementation still expects older 5-class masks, document that mismatch explicitly as technical debt or a required follow-up. Do not silently pretend the vectorization stage is already fully aligned if it is not.

## Outdated Spec Preservation Rule

Keep outdated spec files for now.

Do not delete:

```txt
specs/spec_v005_segformer_train_outdated.md
```

or any other outdated spec file unless explicitly instructed.

Outdated specs should remain available as historical context until the project is over.

If an outdated spec conflicts with the current pipeline, make sure the active spec clearly supersedes it.

## Required Verification

Before finishing this task, verify:

1. Old CNN checkpoint folders still exist.
2. `checkpoints/segformer_b0_run3` still exists.
3. `features/segformer_b0_run3` still exists unless deliberately rebuilt as part of a later training task.
4. `outputs/vectorization/v008` still exists and still contains failed outputs.
5. Active specs no longer describe the current CNN as a 5-class model.
6. Outdated spec files are still present.
7. The repository no longer contains unnecessary inactive CNN cache/history/run artifacts outside the preservation rules above.

## Completion Criteria

This task is complete when:

1. Obsolete CNN artifacts have been removed according to the cleanup scope.
2. Whole checkpoint folders for previous CNN models are preserved.
3. The latest `segformer_b0_run3` files are preserved.
4. Existing `outputs/vectorization/v008` failed outputs are preserved.
5. Active specs accurately describe the current 7-class CNN pipeline.
6. Any remaining mismatch between current vectorization code and the 7-class CNN output is explicitly documented.
7. Outdated specs remain in place for historical reference.
