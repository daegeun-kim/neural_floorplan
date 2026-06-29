# Task 26 - Raster-to-Graph Generous Single-Output Phase 4 Run

## Objective

Generate a new Phase 4 Raster-to-Graph inference run using much more generous thresholds.

The current finding is:

```txt
when graphs are produced, they are often extremely accurate
but production rate is too low
```

This means the current inference settings are still too conservative. The goal of this task is to prioritize graph production and geometric logic over pixel-perfect strictness.

Do not fine-tune.

Do not train.

Do not modify the checkpoint.

This is an inference/post-processing task only.

## Main Direction

Use one generous inference configuration only.

Do not run a settings sweep.

Do not create per-sample nested setting folders such as:

```txt
baseline_tuned_single_start
multistart_tuned_restart010_dist24_cap4
multistart_tuned_restart015_dist24_cap4
tuned_020_025_tol8_mc5
```

Each sample should have exactly one final prediction output folder.

## Required Cleanup Before New Outputs

Remove old Phase 4 Raster-to-Graph output folders under:

```txt
outputs/vectorization/
```

Remove these if present:

```txt
outputs/vectorization/phase4_raster2graph_finetuning/
outputs/vectorization/phase4_raster2graph_preprocessing_test/
outputs/vectorization/phase4_raster2graph_permissive_inference/
outputs/vectorization/phase4_raster2graph_tuned_inference/
outputs/vectorization/phase4_raster2graph_recovery_inference/
outputs/vectorization/phase4_raster2graph_multistart_inference/
```

Do not remove:

```txt
checkpoints_Raster2Graph/
external/raster_to_graph/
docs/original_vector/
data/raster2graph/
specs/
tasks/
```

Because `phase4_raster2graph_preprocessing_test` may be deleted, the new script must not depend on old output-folder images as its only input source.

Instead, regenerate the chosen input variant from each sample's source image.

## Input Source

Use:

```txt
docs/high_quality_architectural/<sample_id>/model_clean.png
```

Preprocess each sample using the current best variant:

```txt
crop512_margin05
```

That means:

```txt
1. detect dark-content bounding box
2. expand bbox by 5 percent
3. crop to expanded bbox
4. scale cropped image so long edge is 512 px
5. center on a 512x512 white canvas
6. normalize with original Raster-to-Graph mean/std
```

Do not use:

```txt
crop512_margin10
crop512_thick1
crop512_thick2
official_512
unit*_crop512
```

## Output Location

Create one new clean Phase 4 output folder:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/
```

Each sample should be stored directly under:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/<sample_id>/
```

Each sample folder should contain:

```txt
input.png
graph_pred.json
graph_pred.svg
graph_overlay.png
graph_overlay_components.png
metrics.json
components.json
notes.txt
```

Do not create nested setting folders inside each sample.

Correct:

```txt
run001/12539/graph_pred.svg
run001/12539/metrics.json
```

Incorrect:

```txt
run001/12539/baseline_tuned_single_start/graph_pred.svg
run001/12539/multistart_tuned_restart010_dist24_cap4/graph_pred.svg
```

Run-level summary files:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/summary.json
outputs/vectorization/phase4_raster2graph_generous_inference/summary.md
```

## Generous Inference Settings

Use the following settings as the primary run:

```txt
source_variant = crop512_margin05
first_step_threshold = 0.10
later_step_threshold = 0.10
first_step_force_best = true
edge_search_threshold = 30
monte_times = 7
max_candidates_per_step = 20
```

Meaning:

```txt
first_step_threshold = 0.10
  accept lower-confidence first-step candidates so graphs are less likely to be empty

later_step_threshold = 0.10
  continue graph generation with lower-confidence later candidates

first_step_force_best = true
  if no first-step candidate passes threshold, still use the best non-end candidate

edge_search_threshold = 30 px
  allow graph nodes to connect across a much wider alignment band
  this prioritizes geometric graph logic over pixel-perfect node alignment

monte_times = 7
  run 7 attempts and keep the best result

max_candidates_per_step = 20
  allow more candidate nodes per graph step before filtering
```

Rationale:

```txt
the model's accepted graphs are already visually/geometrically accurate
the current failure is low production rate
single-digit pixel thresholds are too strict on a 512px image
30px tolerance is acceptable for this experiment because the goal is graph logic, not exact pixel tracing
```

## Multi-Start Requirement

Use mask-and-rerun multi-start, not manual restart seeding.

Do not repeat the failed Task 25 strategy:

```txt
manual seed point
continue same autoregressive component from that seed
```

Instead:

```txt
1. run normal generous inference to produce component 0
2. mask/suppress the covered region of component 0
3. reset graph state
4. rerun normal generous inference on the remaining/suppressed image
5. accept this as component 1 if it is significant
6. repeat until no significant graph is found or restart cap is reached
```

Disconnected graph components are allowed.

Cap:

```txt
max_new_starts = 4
```

This means:

```txt
1 initial component + up to 4 additional components = up to 5 components total
```

## Mask-And-Rerun Settings

Use these defaults:

```txt
covered_node_radius_px = 24
covered_edge_width_px = 30
covered_mask_dilation_px = 20
suppression_fill = white
min_component_points = 3
min_component_edges = 2
max_new_starts = 4
```

Meaning:

```txt
covered_node_radius_px = 24
  mark accepted graph nodes as covered so the next rerun does not start there again

covered_edge_width_px = 30
  mark accepted graph edges as covered with a wide band

covered_mask_dilation_px = 20
  expand the covered mask to suppress nearby duplicate starts

suppression_fill = white
  replace covered graph regions with white background before rerunning inference

min_component_points = 3
  discard tiny components below 3 nodes

min_component_edges = 2
  discard tiny components below 2 edges
```

If white suppression creates artifacts, document it and try a second implementation that overlays the generated graph state instead of removing pixels. But the first priority is the mask-and-rerun reset strategy.

## Merge-On-Intersection Requirement

After generating multiple components, allow components to merge when they intersect or nearly touch.

Implement a post-process graph merge step:

```txt
1. collect all accepted components
2. snap nodes that are close together
3. split edges at intersections
4. merge duplicate / overlapping collinear edges
5. keep disconnected components only when they truly do not intersect
```

Recommended merge tolerances:

```txt
node_snap_tolerance_px = 10
edge_intersection_tolerance_px = 8
collinear_overlap_tolerance_px = 8
```

Meaning:

```txt
node_snap_tolerance_px = 10
  nodes within 10px are treated as the same junction

edge_intersection_tolerance_px = 8
  crossing or near-crossing orthogonal edges within 8px create a shared junction

collinear_overlap_tolerance_px = 8
  near-overlapping horizontal/vertical segments are merged or split into shared endpoints
```

The final `graph_pred.json` and `graph_pred.svg` should represent the merged graph.

`components.json` should still preserve original component membership before merge.

## Best Result Selection

For each component attempt and overall sample, prefer:

```txt
1. non-empty output
2. more accepted graph edges
3. more accepted graph nodes
4. fewer obviously duplicate components
5. normal end token if otherwise comparable
```

Do not over-prioritize pixel-perfect stop codes.

The goal is:

```txt
geometrically logical graph aligned with visible walls
```

not:

```txt
perfectly conservative graph with low production rate
```

## Script

Create or update a single current script:

```txt
external/raster_to_graph/run_inference_generous_phase4.py
```

This should become the current Phase 4 inference entrypoint.

Do not keep creating many parallel run scripts unless necessary.

The script should:

```txt
regenerate crop512_margin05 input from model_clean.png
run generous single/multistart inference
merge intersecting components
write exactly one prediction per sample
write clean run-level summary
```

## Summary Requirements

`summary.md` should include:

```txt
total samples processed
non-empty graph rate
average nodes and edges
average component count before merge
average component count after merge
samples with multiple components
samples where components merged
samples still empty
samples discarded as tiny/noisy
recommended next threshold adjustment
```

Also include a short qualitative note:

```txt
Are graphs geometrically logical compared to the input image?
Did the 30px edge search improve production rate without unacceptable noise?
Did mask-and-rerun produce real missing regions or duplicate the same region?
```

## Spec Update Requirement

After implementation, update:

```txt
specs/spec_v010_phase4_raster2graph_modifications.md
```

Add this task's final settings and whether they are:

```txt
recommended
experimental
retired
```

The spec must mention:

```txt
first_step_threshold = 0.10
later_step_threshold = 0.10
edge_search_threshold = 30
monte_times = 7
max_candidates_per_step = 20
mask-and-rerun multi-start
max_new_starts = 4
merge-on-intersection
single output folder per sample
```

## Acceptance Criteria

1. Old Phase 4 Raster-to-Graph output folders under `outputs/vectorization/` are removed before the new run.
2. A new clean output folder exists at `outputs/vectorization/phase4_raster2graph_generous_inference/`.
3. Each sample has exactly one direct output folder with no nested setting/iteration folders.
4. Each sample output includes `input.png`, `graph_pred.json`, `graph_pred.svg`, `graph_overlay.png`, `metrics.json`, and `components.json`.
5. The generous thresholds are used: `0.10`, `0.10`, `30px`, `monte_times=7`, `max_candidates_per_step=20`.
6. Mask-and-rerun multi-start is used instead of manual restart seeding.
7. Additional starts are capped at `max_new_starts = 4`.
8. Components can merge when they intersect or nearly touch.
9. `summary.md` reports production rate and qualitative graph quality.
10. `specs/spec_v010_phase4_raster2graph_modifications.md` is updated.
11. No training, fine-tuning, or checkpoint modification happens in this task.

