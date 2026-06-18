# Task 04 - Refactor Vectorization Notebook Into an Import-Based Runner

## Context

Task 03 requested an execution notebook for rerunning the v008 vectorization workflow with:

```txt
checkpoints/segformer_b0_run1/best.pt
```

The generated notebook should not duplicate or regenerate the repository's source Python code inside notebook cells. The project implementation should remain in `.py` files under `src/`, and the notebook should act only as a manual execution wrapper.

The notebook should import and call existing project modules instead of pasting implementations from:

```txt
src/
```

This keeps the notebook short, reproducible, and aligned with the source code.

## Objective

Modify the generated execution notebook from Task 03 so it runs the existing repository Python code by importing `.py` modules into the notebook.

The notebook should provide one top-level variable that switches between model runs:

```python
MODEL_RUN = "run1"
```

Changing this value to:

```python
MODEL_RUN = "run2"
```

should switch the notebook to:

```txt
checkpoints/segformer_b0_run2/best.pt
```

without requiring changes elsewhere in the notebook.

## Requirements

1. Do not duplicate full source-code implementations inside notebook cells.
2. Do not regenerate repository `.py` files from inside the notebook.
3. Use the existing project source files under `src/` by importing them into the notebook.
4. Add one top-level run selector variable:

```python
MODEL_RUN = "run1"
```

5. Use `MODEL_RUN` to derive the checkpoint path:

```python
CHECKPOINT_PATH = Path(f"checkpoints/segformer_b0_{MODEL_RUN}/best.pt")
```

6. Validate that only supported run names are used:

```python
assert MODEL_RUN in {"run1", "run2"}
assert CHECKPOINT_PATH.exists(), CHECKPOINT_PATH
```

7. Use `best.pt` only.
8. Do not use `latest.pt`.
9. Preserve the Task 03 output structure:

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
```

10. The notebook should still overwrite `outputs/vectorization/v008`.
11. The notebook should still use the same preview sample set as Task 03.
12. The notebook should not retrain the model.

## Notebook Structure

The notebook should be organized as a clear execution sequence:

1. Imports and autoreload setup.
2. Run selector and path configuration.
3. Config loading.
4. Model and checkpoint loading.
5. Preview sample loading.
6. CNN prediction raster generation.
7. Per-sample output folder creation.
8. v008 vectorization call.
9. Output verification and optional visual display.

Near the top of the notebook, include:

```python
%load_ext autoreload
%autoreload 2
```

Use imports from existing project modules wherever possible, such as:

```python
from src.train_segmentation import load_config, build_preview_loader
from src.checkpointing import load_checkpoint
from src.models import build_backbone, build_decoder, FloorplanSegModel
from src.vectorization.run_mask_to_vector import run as run_vectorization
```

If the exact import list differs, prefer the existing project API and keep reusable logic in `.py` files.

## Guidance for Missing Helpers

If a small helper is needed to make the notebook clean and reusable, prefer adding that helper to an appropriate source module under `src/` instead of embedding a large implementation in the notebook.

Only add source helpers when they are genuinely needed. Do not refactor unrelated code.

## Acceptance Criteria

This task is complete when:

- The notebook no longer contains copied implementations of repository source modules.
- The notebook imports and calls existing `.py` files under `src/`.
- The notebook has exactly one top-level model run switch, `MODEL_RUN`.
- Setting `MODEL_RUN = "run1"` uses `checkpoints/segformer_b0_run1/best.pt`.
- Setting `MODEL_RUN = "run2"` uses `checkpoints/segformer_b0_run2/best.pt`.
- The notebook validates that the selected `best.pt` checkpoint exists.
- The notebook does not use `latest.pt`.
- The notebook does not retrain the model.
- The notebook still generates per-sample `input.png`, `prediction.png`, and `vector.svg` outputs under `outputs/vectorization/v008`.
- The notebook remains an execution wrapper rather than a second implementation of the pipeline.
