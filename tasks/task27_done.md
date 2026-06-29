# Task 27 - Raster-to-Graph Very Generous Generation With Graph Validity Reranking

## Objective

Rerun the current Phase 4 Raster-to-Graph process with even more permissive generation thresholds, then apply graph-level validity filtering and scoring.

Current state:

```txt
Task 26 generous inference produced a better balance:
some graphs are generated
generated graphs are often very accurate
some graphs are still missing or incomplete
```

Next direction:

```txt
allow the model to generate more candidate graph content
then use architectural graph validation to reject, clean, and rerank outputs
```

Do not fine-tune.

Do not train.

Do not modify the checkpoint.

This task is inference + graph post-processing only.

## Output Replacement Requirement

Replace the existing outputs in:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/
```

Do not create a new sibling folder such as:

```txt
phase4_raster2graph_very_generous_inference/
phase4_raster2graph_validated_inference/
run002/
```

The current Phase 4 output location should remain:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/
```

Before rerunning, remove the existing contents of `run001/` and regenerate them.

Keep the same single-output-per-sample structure:

```txt
run001/<sample_id>/input.png
run001/<sample_id>/graph_pred.json
run001/<sample_id>/graph_pred.svg
run001/<sample_id>/graph_overlay.png
run001/<sample_id>/metrics.json
run001/<sample_id>/components.json
run001/<sample_id>/notes.txt
```

Do not create nested setting folders inside each sample.

## Input Source

Use the same source and preprocessing as Task 26:

```txt
docs/high_quality_architectural/<sample_id>/model_clean.png
```

Preprocess as:

```txt
crop512_margin05
```

Do not use:

```txt
crop512_margin10
crop512_thick1
crop512_thick2
official_512
unit*_crop512
```

## Very Generous Generation Settings

Use this single generation setting:

```txt
source_variant = crop512_margin05
first_step_threshold = 0.05
later_step_threshold = 0.05
first_step_force_best = true
edge_search_threshold = 40
monte_times = 10
max_candidates_per_step = 30
```

Meaning:

```txt
first_step_threshold = 0.05
  allow much weaker first-step candidates so graph generation starts more often

later_step_threshold = 0.05
  allow continuation through weaker later-step predictions

first_step_force_best = true
  if no first-step candidate passes, use the best non-end candidate anyway

edge_search_threshold = 40 px
  allow a wide node-connection band on the 512px canvas
  this is intentionally generous because the goal is geometric graph logic, not pixel-perfect tracing

monte_times = 10
  generate more candidate attempts per sample/component

max_candidates_per_step = 30
  keep more candidates before graph assembly and validation
```

Do not go below `0.05` in this task.

## Hard Filters

Hard filters remove graph elements. Use them after generation and before final export.

### 1. Angle Filter

Discard any edge whose angle is not within +/-10 degrees of horizontal or vertical.

Allowed:

```txt
angle within 10 degrees of 0 degrees
angle within 10 degrees of 90 degrees
angle within 10 degrees of 180 degrees
```

Rejected:

```txt
diagonal edges outside those angle windows
```

Record:

```txt
edges_removed_angle_filter
```

### 2. Tiny Component Filter

Discard components that are too small to be useful.

Recommended:

```txt
min_component_points = 3
min_component_edges = 2
```

Record:

```txt
components_removed_tiny
points_removed_tiny_components
edges_removed_tiny_components
```

### 3. One-Edge Component Filter

Discard components with only one edge unless the edge is long and strongly supported by wall evidence.

Default:

```txt
discard_one_edge_components = true
```

Exception:

```txt
keep if edge_length_px >= 80
and wall_evidence_alignment_score >= 0.65
```

Record:

```txt
one_edge_components_removed
one_edge_components_kept_by_evidence
```

### 4. Unsupported Short Dangling Edge Filter

Remove short dangling edges if they are poorly supported by raster wall evidence.

Recommended:

```txt
dangling_short_edge_max_length_px = 35
dangling_edge_min_wall_evidence = 0.45
```

Meaning:

```txt
if an edge touches a degree-1 node
and edge length <= 35 px
and wall evidence score < 0.45
then remove it
```

Record:

```txt
dangling_edges_removed
```

## Soft Scores

Soft scores should be used for reranking candidate attempts/components.

Do not delete graph elements only because of a soft score unless a hard filter also applies.

### 1. Wall Evidence Alignment Score

Compare predicted edges to the input raster.

Use the preprocessed `input.png`.

For each edge:

```txt
sample pixels along the edge
also sample a tolerance band around the edge
count how much dark wall-like evidence lies near the edge
```

Recommended:

```txt
wall_evidence_band_px = 10
dark_pixel_threshold = 180
```

Output:

```txt
wall_evidence_alignment_score
unsupported_edge_ratio
```

Higher is better.

This is edge support, not junction support.

Do not implement pixel-perfect junction support in this task.

### 2. Rectangle / Closed-Region Reward

Reward graphs that form good rectangular or near-rectangular closed regions.

Compute:

```txt
orthogonal cycles
closed rectangular / L-shaped room-like cycles
reasonable cycle area
```

Recommended:

```txt
min_cycle_area_px2 = 400
max_cycle_aspect_ratio = 8.0
```

Output:

```txt
rectangle_cycle_count
rectangle_cycle_score
```

Higher is better.

### 3. Dangling Node Penalty

Penalize graphs with too many degree-1 nodes.

Do not remove all dangling nodes. Some wall ends can be valid.

Compute:

```txt
dangling_node_count
dangling_node_ratio
dangling_penalty
```

Recommended:

```txt
dangling_ratio_soft_limit = 0.35
```

Meaning:

```txt
if more than 35 percent of nodes are degree 1, lower the graph score
```

### 4. Tiny Component Penalty

Penalize graphs with many small components, even if some survive hard filtering.

Compute:

```txt
small_component_count
small_component_penalty
```

### 5. Unsupported Edge Penalty

Penalize graphs with many edges that have weak raster support.

Recommended:

```txt
unsupported_edge_threshold = 0.35
```

Meaning:

```txt
edges with wall evidence score below 0.35 count as unsupported for soft scoring
```

## Do Not Implement Junction Support Yet

Do not add a junction support score in this task.

Rationale:

```txt
the current goal is valid graph topology and edge relationship
not pixel-perfect node placement
junction-pixel evidence can be too strict at this stage
```

## Candidate Reranking

Currently the best inference attempt may be selected mostly by edge/node count.

Replace that with graph validity reranking.

For each Monte Carlo attempt, compute a final candidate score:

```txt
candidate_score =
  + 3.0 * wall_evidence_alignment_score
  + 2.0 * rectangle_cycle_score
  - 1.5 * dangling_penalty
  - 2.0 * unsupported_edge_ratio
  - 1.0 * small_component_penalty
  + 0.2 * normalized_edge_count
```

The exact weights may be adjusted if needed, but document any change in `metrics.json` and `summary.md`.

The selected graph for each sample should be:

```txt
the highest-scoring candidate after hard filters
```

not simply:

```txt
the candidate with the most nodes or edges
```

## Mask-And-Rerun Multistart

Keep the Task 26 direction:

```txt
mask-and-rerun multistart
not manual restart seeding
```

Use:

```txt
max_new_starts = 4
```

Keep or update these generous defaults:

```txt
covered_node_radius_px = 24
covered_edge_width_px = 30
covered_mask_dilation_px = 20
suppression_fill = white
```

The same hard filters and soft scoring should apply to each component and to the final merged graph.

## Merge-On-Intersection

Keep merge-on-intersection from Task 26.

Recommended:

```txt
node_snap_tolerance_px = 10
edge_intersection_tolerance_px = 8
collinear_overlap_tolerance_px = 8
```

Final exports should use the merged graph:

```txt
graph_pred.json
graph_pred.svg
graph_overlay.png
```

`components.json` should preserve pre-merge component information.

## Metrics

Each sample `metrics.json` should include:

```json
{
  "source_variant": "crop512_margin05",
  "first_step_threshold": 0.05,
  "later_step_threshold": 0.05,
  "edge_search_threshold": 40,
  "monte_times": 10,
  "max_candidates_per_step": 30,
  "max_new_starts": 4,
  "hard_filters": {
    "edges_removed_angle_filter": 0,
    "components_removed_tiny": 0,
    "one_edge_components_removed": 0,
    "dangling_edges_removed": 0
  },
  "soft_scores": {
    "wall_evidence_alignment_score": 0.0,
    "rectangle_cycle_score": 0.0,
    "dangling_penalty": 0.0,
    "unsupported_edge_ratio": 0.0,
    "small_component_penalty": 0.0,
    "candidate_score": 0.0
  },
  "final_num_points": 0,
  "final_num_edges": 0,
  "final_num_components": 0,
  "empty": false
}
```

Also record:

```txt
attempt_scores
selected_attempt_index
components_before_merge
components_after_merge
```

## Output Files

Keep the existing clean Task 26 structure:

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

Do not create nested setting folders.

## Summary Requirements

Update:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/summary.json
outputs/vectorization/phase4_raster2graph_generous_inference/summary.md
```

`summary.md` should include:

```txt
production rate before/after this rerun if previous summary exists
non-empty graph rate
average final nodes/edges
average wall evidence score
average rectangle cycle score
average dangling penalty
average unsupported edge ratio
number of edges removed by angle filter
number of tiny/one-edge components removed
samples improved by validity reranking
samples where generous generation produced too much noise
recommended next threshold adjustment
```

## Spec Update Requirement

Update:

```txt
specs/spec_v010_phase4_raster2graph_modifications.md
```

Add Task 27 details:

```txt
first_step_threshold = 0.05
later_step_threshold = 0.05
edge_search_threshold = 40
monte_times = 10
max_candidates_per_step = 30
hard filters
soft scoring / reranking
no junction support score yet
outputs replace phase4_raster2graph_generous_inference/run001
```

## Acceptance Criteria

1. Existing contents of `outputs/vectorization/phase4_raster2graph_generous_inference/` are replaced.
2. No new sibling Phase 4 output folder is created.
3. Each sample still has only one direct output folder and no nested setting folders.
4. The very generous thresholds are used: `0.05`, `0.05`, `40px`, `monte_times=10`, `max_candidates_per_step=30`.
5. Edges outside +/-10 degrees of horizontal/vertical are discarded.
6. Tiny and one-edge components are filtered according to the rules above.
7. Wall evidence alignment score is implemented.
8. Rectangle/closed-region reward is implemented.
9. Dangling-node and unsupported-edge penalties are implemented.
10. Junction support score is not implemented.
11. Candidate selection uses graph validity reranking, not raw node/edge count alone.
12. `summary.md` reports hard-filter removals and soft scores.
13. `specs/spec_v010_phase4_raster2graph_modifications.md` is updated.
14. No training, fine-tuning, or checkpoint modification happens in this task.

