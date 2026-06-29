# Task 21 - Test Pretrained SizheHu Raster-to-Graph Checkpoint

## Objective

Test the downloaded pretrained Raster-to-Graph checkpoint from SizheHu's repository as an inference-only experiment.

This task is only for testing whether the pretrained model can produce a useful graph on the current project's sample raster previews.

Do not fine-tune.

Do not retrain.

Do not replace the existing SegFormer raster2graph baseline yet.

Do not delete or overwrite existing raster2graph outputs.

## Checkpoint

Use this downloaded checkpoint:

```txt
C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\checkpoints_Raster2Graph\checkpoint0299.pth
```

Treat this as an external pretrained model checkpoint.

The checkpoint should not be assumed to match the current `src/raster2graph/model.py` architecture.

## Test Input Folder

Use the current raster2graph preview folder as the first smoke-test source:

```txt
C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\outputs\raster2graph\runs\run001_wall_graph_baseline\previews
```

Current preview sample folders contain files such as:

```txt
ground_truth.png
prediction.png
error_overlay.png
```

First audit these preview files and decide which raster file can reasonably be used for a pretrained inference smoke test.

If these preview files are not appropriate as model input, trace the preview sample back to its original dataset input where possible:

```txt
masks/debug_overlay.png
```

The goal is to run a small visual test, not to create a final training dataset.

## Required External Code Check

Before trying to load the checkpoint, identify whether the repository contains the original Raster-to-Graph model code required by `checkpoint0299.pth`.

Check for:

```txt
model architecture class
checkpoint loading function
config or args used by the checkpoint
input preprocessing code
inference decoding code
graph export or visualization code
```

If the required model code is missing, do not force-load the checkpoint into the current SegFormer graph-head model.

Instead, document the missing files and what must be added from SizheHu's repository.

## Inference-Only Test

Once the matching model code is available:

1. Load `checkpoint0299.pth`.
2. Run inference on a small set of preview/sample raster images.
3. Save all outputs under a new folder:

```txt
outputs/raster2graph/pretrained_sizhehu_test/
```

Recommended structure:

```txt
outputs/raster2graph/pretrained_sizhehu_test/
  inputs/
  raw_outputs/
  prediction_svg/
  comparison/
  notes.md
```

## Required Output Format

The important output is SVG.

For every tested sample, generate:

```txt
prediction.svg
```

Also generate a comparison artifact against the existing reference graph when available:

```txt
reference.svg
comparison.svg
```

The comparison should make it clear whether the pretrained model is producing:

```txt
wall graph nodes
wall graph edges
orthogonal structure
usable topology
```

PNG previews are optional, but SVG prediction output is required.

## No Training Rule

This task must not call any training loop.

Do not modify:

```txt
checkpoints_Raster2Graph/checkpoint0299.pth
checkpoints/raster2graph_b0/
outputs/raster2graph/runs/run001_wall_graph_baseline/
```

The test must be non-destructive.

## Evaluation Questions

After running inference, answer:

1. Can `checkpoint0299.pth` be loaded with available model code?
2. What input format does the pretrained model expect?
3. Can it run on the project's current raster preview/sample images without retraining?
4. Does it output a visible graph?
5. Is the graph closer to the desired `wall_graph_debug.svg` target than the current SegFormer graph-head baseline?
6. Does it produce orthogonal wall-like structure?
7. Does the output format need conversion into this project's `wall_graph.json` schema?
8. Is direct inference promising enough to continue, or is dataset conversion/fine-tuning required?

## Completion Criteria

This task is complete when:

1. The checkpoint loading requirements are known.
2. Missing external Raster-to-Graph files, if any, are documented.
3. At least one inference attempt is made when the required code is available.
4. SVG prediction output is generated when inference succeeds.
5. Results are written under `outputs/raster2graph/pretrained_sizhehu_test/`.
6. A short `notes.md` explains whether direct pretrained inference is promising.
7. No fine-tuning or retraining is performed.

