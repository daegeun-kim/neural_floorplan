# Task 29 - Raster-to-Graph Fast Three-Sample Runtime-Limited Test

## Objective

Rerun the current Phase 4 Raster-to-Graph process on only three docs samples using a faster runtime-limited version of Task 28.

This task keeps the same main idea:

```txt
standardized 10 percent margin
very permissive thresholds
Task 27 hard filters and soft scoring
mask-and-rerun multistart
```

But reduces the expensive search settings:

```txt
monte_times = 4
max_new_starts = 2
```

Do not fine-tune.

Do not train.

Do not modify the checkpoint.

## Sample Limit

Only test three sample images from `docs`.

Use the first three available sample folders from:

```txt
docs/high_quality_architectural/
```

Each selected sample must contain:

```txt
model_clean.png
```

If the implementation already has a convenient fixed sample list, use three representative samples from the docs folder and record their IDs in `summary.md`.

Do not process the full dataset in this task.

## Output Replacement

Replace existing outputs in:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/
```

After this task, `run001/` should contain only the three tested sample folders plus run-level summaries.

Keep one direct output folder per sample:

```txt
run001/<sample_id>/input.png
run001/<sample_id>/graph_pred.json
run001/<sample_id>/graph_pred.svg
run001/<sample_id>/graph_overlay.png
run001/<sample_id>/graph_overlay_components.png
run001/<sample_id>/metrics.json
run001/<sample_id>/components.json
run001/<sample_id>/notes.txt
```

Do not create nested setting folders.

## Preprocessing

Use Task 28 preprocessing:

```txt
crop to actual dark content bbox
add standardized 10 percent margin
resize long edge to 512 px
center on 512x512 white canvas
```

Source variant:

```txt
crop512_margin10_standardized
```

No wall/content should directly touch the final 512px canvas edge. If it does, record it in `metrics.json`.

## Runtime-Limited Generation Settings

Use:

```txt
first_step_threshold = 0.02
later_step_threshold = 0.02
first_step_force_best = true
edge_search_threshold = 50
monte_times = 4
max_candidates_per_step = 40
max_new_starts = 2
```

This keeps Task 28's aggressive graph-generation thresholds, but reduces runtime.

## Keep Validation

Keep Task 27/28 graph validation:

```txt
discard edges outside +/-10 degrees of horizontal/vertical
filter tiny components
filter one-edge components unless strongly supported
filter short unsupported dangling edges
wall evidence alignment score
rectangle / closed-region reward
dangling-node penalty
unsupported-edge penalty
candidate validity reranking
```

Do not add junction support scoring.

## Summary Requirements

Write:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/summary.json
outputs/vectorization/phase4_raster2graph_generous_inference/summary.md
```

`summary.md` should include:

```txt
the three sample IDs tested
runtime per sample
non-empty graph rate
final node/edge counts
component counts
hard-filter removals
soft-score values
whether reduced monte_times still produces useful graphs
whether max_new_starts=2 is enough for these samples
```

## Spec Update

Update:

```txt
specs/spec_v010_phase4_raster2graph_modifications.md
```

Add that Task 29 is a runtime-limited smoke test:

```txt
monte_times = 4
max_new_starts = 2
three docs samples only
```

## Acceptance Criteria

1. Only three docs sample images are processed.
2. Existing `phase4_raster2graph_generous_inference/run001/` outputs are replaced.
3. `run001/` contains only the three tested sample folders plus summaries.
4. `monte_times = 4` is used.
5. `max_new_starts = 2` is used.
6. Task 28 standardized 10 percent margin preprocessing is used.
7. Task 27/28 hard filters and soft scoring remain active.
8. No junction support score is added.
9. `summary.md` reports sample IDs and runtime.
10. `spec_v010_phase4_raster2graph_modifications.md` is updated.
11. No training, fine-tuning, or checkpoint modification happens.

