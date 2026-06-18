# Task 03 - Rerun Vectorization With segformer_b0_run1 Best Checkpoint

## Context

The current vectorization output folder:

```txt
C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\outputs\vectorization\v008
```

contains SVG vector outputs that were generated from CNN prediction rasters associated with:

```txt
C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\checkpoints\segformer_b0_v005\best.pt
```

The project also has a separate checkpoint iteration:

```txt
C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\checkpoints\segformer_b0_run1\best.pt
```

`segformer_b0_run1` and future runs such as `segformer_b0_run2` should be treated as separate model iterations whose vectorization outputs can be compared. For this task, use `best.pt` only. Do not use `latest.pt`.

The current `outputs/vectorization/v008` folder contains only final vector SVG files. For comparison and debugging, each vectorization sample output should include all three pipeline artifacts:

1. Initial raster input.
2. Segmented raster prediction produced by the CNN model.
3. Final vector output.

## Objective

Create an execution notebook that reruns the v008 vectorization workflow using:

```txt
checkpoints/segformer_b0_run1/best.pt
```

The notebook should allow the user to run the process manually.

The process should overwrite:

```txt
outputs/vectorization/v008
```

with new outputs generated from `segformer_b0_run1/best.pt`.

## Requirements

1. Create a runnable Jupyter notebook for this workflow.
2. The notebook must load `checkpoints/segformer_b0_run1/best.pt`.
3. The notebook must not load or use `checkpoints/segformer_b0_run1/latest.pt`.
4. Use the same preview sample set currently used for the v008 vectorization comparison.
5. Generate CNN segmented raster prediction outputs from the selected checkpoint before vectorization.
6. Run the existing v008 mask-to-vector pipeline on those segmented prediction rasters.
7. Overwrite `outputs/vectorization/v008` with the new run outputs.
8. Do not change repository source code unless it is strictly required to make the notebook executable.
9. Do not retrain the CNN model.
10. Do not change model architecture, class mapping, or vectorization heuristics as part of this task.

## Output Folder Structure

Each sample must be stored in its own folder inside:

```txt
outputs/vectorization/v008
```

Use this structure:

```txt
outputs/vectorization/v008/
  sample_000/
    input.png
    prediction.png
    vector.svg
  sample_001/
    input.png
    prediction.png
    vector.svg
  sample_002/
    input.png
    prediction.png
    vector.svg
  sample_003/
    input.png
    prediction.png
    vector.svg
```

The exact number of sample folders should match the same preview sample count used by the existing v008 comparison workflow.

## Notebook Behavior

The notebook should be explicit and reproducible. It should show or define:

1. The checkpoint path being used.
2. The preview sample source being used.
3. The output directory being overwritten.
4. The generated segmented raster paths.
5. The generated vector SVG paths.

The notebook should make it easy to visually compare:

```txt
input.png -> prediction.png -> vector.svg
```

for each sample.

## Acceptance Criteria

This task is complete when:

- A new execution notebook exists for running this workflow manually.
- The notebook uses `checkpoints/segformer_b0_run1/best.pt`.
- The notebook does not use `latest.pt`.
- Running the notebook overwrites `outputs/vectorization/v008`.
- Each sample output folder contains:
  - `input.png`
  - `prediction.png`
  - `vector.svg`
- The same preview sample set is used for the comparison.
- At least 3 sample folders are generated.
- The generated SVG files come from the v008 vectorization pipeline.
- The notebook does not retrain the model.
- The output structure supports direct comparison between initial raster, CNN segmentation, and final vector output.
