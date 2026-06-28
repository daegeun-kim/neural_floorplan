# Vectorization Outputs

Outputs here are organized by **phase** - the conceptual family of vectorization approach used to produce them. See `specs/vectorization_phase_history.md` for the full history; this file is a short pointer.

## What "phase" means

| Folder | Phase | Segmentation | Vectorization idea |
|---|---|---|---|
| `phase1_5class_line_vectorization/` | 1 | 5-class | pixels/lines converted directly to line-segment vector geometry |
| `phase2_7class_semantic_vectorization/` | 2 | 7-class | richer semantic pixel evidence (door_arc/door_leaf/door_origin/window) converted to vector geometry |
| `phase3_7class_point_vectorization/` | 3 | 7-class | architectural point recognition + graph construction (historical, superseded by Phase 4) |
| `phase4_raster2graph_generous_inference/` | 4 | `model_clean.png` RGB input | pretrained Raster-to-Graph inference with true padding, generous thresholds, scoring, multistart recovery, and merge/filter cleanup |

## What "iteration" and "run" mean (inside each phase folder)

- `iterationN` = a distinct vectorization method/version (a change to the code that converts segmented pixels into vector geometry).
- `runN` = a distinct CNN segmentation model generation (a different trained checkpoint that produced the prediction masks being vectorized).
- A folder name like `iteration5_run3` means "vectorization method 5, run against the model-3 checkpoint's predictions."

## Why "failed" appears in some folder names

`failed` marks an output set that did not meet required architectural/vectorization quality at the time it was produced - kept for historical comparison, not deleted. Folders without `failed` (e.g. `iteration5_run3`) are retained successful outputs for their historical phase. The current Phase 4 output root is `phase4_raster2graph_generous_inference/<sample>/`.

## Provenance note (task20 reorganization)

These folders previously lived flat under `outputs/vectorization/v008/<iteration_run_name>/`. Task20 moved them into the phase folders above (pure `git mv`, no content changes) and retired the `v008` folder name. Old-to-new mapping:

```txt
outputs/vectorization/v008/iteration1_run1_failed -> outputs/vectorization/phase1_5class_line_vectorization/iteration1_run1_failed
outputs/vectorization/v008/iteration2_run1_failed -> outputs/vectorization/phase1_5class_line_vectorization/iteration2_run1_failed
outputs/vectorization/v008/iteration2_run2_failed -> outputs/vectorization/phase1_5class_line_vectorization/iteration2_run2_failed
outputs/vectorization/v008/iteration3_run2_failed -> outputs/vectorization/phase2_7class_semantic_vectorization/iteration3_run2_failed
outputs/vectorization/v008/iteration4_run3_failed -> outputs/vectorization/phase2_7class_semantic_vectorization/iteration4_run3_failed
outputs/vectorization/v008/iteration5_run3        -> outputs/vectorization/phase3_7class_point_vectorization/iteration5_run3
```

Mentions of the old `outputs/vectorization/v008/iteration5_run3` path in `specs/spec_v008_phase3_mask_to_vector.md`'s Task14/15/17/18/19 Debugging Notes are historical narrative (accurate at the time they were written) and were intentionally left as-is rather than rewritten - use the mapping above if you need to resolve one of those paths today.

**Known drift, not fixed by this reorganization** (out of scope - task20 is organization only, and these are vectorization/inference source code, not config or docs):

- `scripts/run_vectorization_v008.py` and `notebooks/run_vectorization_v008_run1.ipynb` still default new output to `outputs/vectorization/v008/...`. Pass an explicit output path under the correct phase folder (e.g. `--output-name` placed under `phase3_7class_point_vectorization/`) until/unless those scripts are updated.
- `configs/vectorization_v008.yaml`'s `output.output_dir` was updated to the phase3 folder (see below) since that's a config value, not source code.
