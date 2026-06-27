# Task 28 - Raster-to-Graph Standardized 10 Percent Margin And Extreme Threshold Rerun

## Objective

Rerun the current Phase 4 Raster-to-Graph inference with:

```txt
standardized 10 percent input margin
more extremely permissive generation thresholds
the existing hard filters and soft scoring from Task 27
```

Observation:

```txt
some Phase 4 input/output images have walls directly touching the image edge
R2G tends to generate graphs near the center more reliably than near image edges
walls touching the 512px canvas edge are much less likely to be predicted
```

Therefore, every input image must be standardized so wall content never directly touches the canvas edge.

Do not fine-tune.

Do not train.

Do not modify the checkpoint.

## Output Replacement

Replace existing outputs in:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/
```

Keep the same simple structure:

```txt
run001/<sample_id>/input.png
run001/<sample_id>/graph_pred.json
run001/<sample_id>/graph_pred.svg
run001/<sample_id>/graph_overlay.png
run001/<sample_id>/metrics.json
run001/<sample_id>/components.json
run001/<sample_id>/notes.txt
```

Do not create nested setting folders.

## Required Preprocessing

For every submitted image:

```txt
1. detect actual dark floorplan content
2. remove existing uneven outer margin by cropping to content bbox
3. add standardized 10 percent margin around the content
4. resize the margined crop so the long edge fits 512 px
5. paste centered on a 512x512 white canvas
```

This replaces the previous `crop512_margin05` choice.

New source variant name:

```txt
crop512_margin10_standardized
```

Important:

```txt
no wall should directly touch the final 512x512 image edge
```

If content still touches an edge after preprocessing, record it in `metrics.json`.

## Extreme Generation Settings

Use much more permissive settings than Task 27:

```txt
first_step_threshold = 0.02
later_step_threshold = 0.02
first_step_force_best = true
edge_search_threshold = 50
monte_times = 12
max_candidates_per_step = 40
```

Rationale:

```txt
Task 27 hard filters and soft scoring work well
but not enough graph is generated to filter
therefore generate much more candidate graph content first
then let validity filtering/reranking clean it
```

## Keep Task 27 Validation

Keep all Task 27 hard filters and soft scores:

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

Still do not implement junction support score.

## Summary Requirements

Update:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/summary.md
```

Include:

```txt
whether 10 percent standardized margin increased graph production
how many final inputs had content touching the canvas edge
non-empty graph rate
average nodes/edges before and after filters
hard-filter removals
soft-score averages
whether extreme thresholds added useful graph content or too much noise
```

## Spec Update

Update:

```txt
specs/spec_v010_raster2graph_modifications.md
```

Add:

```txt
standardized 10 percent margin preprocessing
first_step_threshold = 0.02
later_step_threshold = 0.02
edge_search_threshold = 50
monte_times = 12
max_candidates_per_step = 40
```

## Acceptance Criteria

1. Existing `phase4_raster2graph_generous_inference/run001/` outputs are replaced.
2. Every sample uses standardized 10 percent margin preprocessing.
3. No walls/content directly touch the final 512px canvas edge, or failures are logged.
4. Extreme thresholds are used: `0.02`, `0.02`, `50px`, `monte_times=12`, `max_candidates_per_step=40`.
5. Task 27 hard filters and soft scoring remain active.
6. No junction support score is added.
7. Each sample has only one direct output folder.
8. `summary.md` reports margin and production-rate effects.
9. `spec_v010_raster2graph_modifications.md` is updated.
10. No training, fine-tuning, or checkpoint modification happens.

