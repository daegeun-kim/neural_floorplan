# Task 22 - Raster-to-Graph Input Preprocessing Sweep

## Objective

Before regenerating or running the fine-tuning pipeline, test whether better preprocessing can make this project's `model_clean.png` inputs compatible with the pretrained Raster-to-Graph checkpoint.

The current assumption is:

```txt
The pretrained Raster-to-Graph model may already know how to convert regular floorplan rasters into graphs.
The main failure may be input-distribution mismatch, not lack of training.
```

Therefore this task prioritizes preprocessing and inference-only testing over training.

Do not start fine-tuning in this task.

Do not regenerate the fine-tuning pipeline in this task.

## Current Direction

Use the pretrained checkpoint:

```txt
checkpoints_Raster2Graph/checkpoint0299.pth
```

Use the official-style Raster-to-Graph code under:

```txt
external/raster_to_graph/
```

Primary input source:

```txt
docs/high_quality_architectural/<sample>/model_clean.png
```

Graph output target for future training/inference comparison:

```txt
docs/high_quality_architectural/<sample>/masks/wall_graph.json
```

## Important Scope Decision

For this task, keep the graph target and model output focused on wall graph structure only.

Do not include door nodes.

Do not include window nodes.

Do not include room labels.

Doors and windows should be pushed back to the later vectorization/CAD stage, where the existing seven-class semantic masks can still be used:

```txt
window
door_arc
door_leaf
door_origin
```

This task should not make Raster-to-Graph responsible for detailed opening reconstruction.

## Clean Up Outdated Phase 4 Artifacts In Docs

Before creating new preprocessing outputs, inspect the `docs/high_quality_architectural/.../<sample>/masks/` folders and any related Phase 4 per-sample files.

Replace or remove outdated Phase 4 artifacts that no longer match the current plan, especially:

```txt
old 512px preprocessed Raster-to-Graph images
old graph files that include door/window nodes
old graph debug files generated from a door/window-node version
old preprocessing artifacts with unclear naming
```

The new per-sample artifacts should be clean, current, and unambiguous.

Do not delete source files:

```txt
model.svg
model_clean.png
F1_scaled.png
semantic masks needed by earlier stages
```

Do not remove checkpoint folders.

## Required Folder And File Hygiene

Keep the folder structure clean.

Use sample-local files only when they are true per-sample data.

Use run/output folders only when they are experiment outputs.

Avoid creating parallel folders with nearly identical meanings.

Avoid leaving both old and new versions of the same preprocessing artifact in the same sample folder.

Recommended sample-local artifact names:

```txt
masks/wall_graph.json
masks/wall_graph_debug.svg
masks/wall_graph_debug.png
raster2graph/model_clean_r2g_512.png
raster2graph/model_clean_r2g_crop512.png
raster2graph/model_clean_r2g_crop512_thick.png
raster2graph/preprocess_metadata.json
```

If a `raster2graph/` subfolder is created inside each sample folder, keep all Raster-to-Graph preprocessing variants for that sample there.

Do not scatter preprocessing files directly across multiple unrelated folders.

## Preprocessing Variants To Test

Create a preprocessing sweep for a small test set first.

Required variants:

### Variant A - Official Resize

Match the original repo demo preprocessing:

```txt
load model_clean.png as RGB
scale longer edge to 512 px
preserve aspect ratio
center on white 512x512 canvas
normalize using original repo mean/std during inference
```

Save visual copy:

```txt
raster2graph/model_clean_r2g_512.png
```

### Variant B - Content Crop Then Resize

Crop to the visible floorplan drawing before resizing.

Process:

```txt
detect non-white / non-background bounding box
expand bbox by configurable margin
crop model_clean.png
scale longer crop edge to 512 px
center on white 512x512 canvas
```

Use at least two margins:

```txt
5 percent
10 percent
```

Save visual copies:

```txt
raster2graph/model_clean_r2g_crop512_margin05.png
raster2graph/model_clean_r2g_crop512_margin10.png
```

### Variant C - Crop + Wall/Line Thickening

Use the best crop variant and test mild line thickening after resize.

The goal is not to distort the plan; the goal is to compensate for line thinning after shrinking large CubiCasa renders to 512 px.

Suggested operations:

```txt
threshold dark linework
apply small dilation / max filter
merge back onto white canvas
```

Test only mild settings first:

```txt
1 px thickening
2 px thickening
```

Save visual copies:

```txt
raster2graph/model_clean_r2g_crop512_thick1.png
raster2graph/model_clean_r2g_crop512_thick2.png
```

### Variant D - Multi-Unit Detection / Split Candidate

Some CubiCasa samples contain more than one floorplan unit in a single image.

Detect whether the visible linework has multiple separated large components.

For likely multi-unit samples:

```txt
flag in preprocess_metadata.json
optionally save per-component crop candidates
do not silently squeeze multiple units into one 512 canvas
```

Suggested filenames:

```txt
raster2graph/model_clean_r2g_unit0_crop512.png
raster2graph/model_clean_r2g_unit1_crop512.png
```

## Metadata

For each processed sample, write:

```txt
raster2graph/preprocess_metadata.json
```

Include:

```json
{
  "sample_id": "example",
  "source": "model_clean.png",
  "source_size": [0, 0],
  "variants": [
    {
      "name": "official_512",
      "path": "raster2graph/model_clean_r2g_512.png",
      "crop_bbox": null,
      "scale": 0.0,
      "canvas_size": [512, 512],
      "notes": ""
    }
  ],
  "multi_unit_flag": false,
  "selected_variant_for_inference": null
}
```

## Inference Test Set

After preprocessing is implemented, prepare a small Raster-to-Graph inference test set.

Use a balanced set:

```txt
10 normal single-unit samples
3 tall/narrow samples
3 wide samples
2 multi-unit or suspected multi-unit samples
2 visually dense/large samples
```

If possible, include the samples already tested in Task 21:

```txt
12539
12967
1316
13736
```

The test set should be defined in a simple manifest:

```txt
data/raster2graph/preprocess_test_samples.json
```

Each entry should include:

```json
{
  "sample_id": "12539",
  "source_model_clean": "docs/high_quality_architectural/.../12539/model_clean.png",
  "wall_graph": "docs/high_quality_architectural/.../12539/masks/wall_graph.json",
  "preprocess_folder": "docs/high_quality_architectural/.../12539/raster2graph"
}
```

## Inference-Only Comparison

Run the pretrained Raster-to-Graph model on the preprocessing variants.

For each sample and variant, record:

```txt
variant name
number of predicted points
number of predicted edges
whether output is empty
whether graph looks plausibly wall-like
whether graph aligns with visible walls
notes on obvious failure
```

Save experiment outputs under:

```txt
outputs/vectorization/phase4_raster2graph_preprocessing_test/run001/<sample_id>/<variant>/
```

Each variant output folder should contain:

```txt
input.png
graph_pred.svg
graph_overlay.png
metrics.json
notes.txt
```

Do not use:

```txt
outputs/raster2graph/
```

That folder was testing-only and should remain retired.

## Evaluation Questions

At the end of the task, answer:

1. Which preprocessing variant produces the most non-empty predictions?
2. Which preprocessing variant produces the most plausible wall-like graph?
3. Does content cropping improve over official long-edge resize?
4. Does mild line thickening help or hurt?
5. Are multi-unit samples a major failure source?
6. Is the pretrained model close enough that small fine-tuning is worth attempting?
7. If fine-tuning is still needed, what exact preprocessing variant should become the training input?

## Completion Criteria

This task is complete when:

1. Outdated Phase 4 docs-folder preprocessing artifacts are removed or replaced.
2. Any old graph files with door/window nodes are replaced by wall-graph-only files or clearly retired.
3. The per-sample Raster-to-Graph preprocessing files are stored cleanly in sample-local `raster2graph/` folders.
4. A small test manifest exists at `data/raster2graph/preprocess_test_samples.json`.
5. Pretrained Raster-to-Graph inference has been run on the preprocessing variants.
6. Outputs are stored under `outputs/vectorization/phase4_raster2graph_preprocessing_test/run001/<sample_id>/<variant>/`.
7. A concise comparison summary identifies the best preprocessing variant.
8. No fine-tuning pipeline is regenerated yet.

## Notes For Claude

Keep this task focused.

Do not start full model fine-tuning.

Do not spend time optimizing the fine-tuning training loop.

Do not add door/window graph nodes back into the Raster-to-Graph target.

The purpose is to decide whether preprocessing can make the pretrained model useful before committing to expensive training.

