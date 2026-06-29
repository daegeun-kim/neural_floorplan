# Task 24 - Raster-to-Graph Autoregressive Recovery And Continuation

## Objective

Test whether partially generated Raster-to-Graph outputs can be improved by recovering from failed autoregressive steps instead of stopping the entire graph generation process.

Current behavior:

```txt
predict next frontier
filter low-confidence / invalid nodes
if no valid nodes -> stop whole graph
if end token appears -> stop whole graph
otherwise add nodes and edges, draw them on the input, continue
```

This can produce good graphs for one part of a floorplan, then stop before generating the rest.

This task should modify inference only:

```txt
if one graph step fails, try controlled recovery before stopping
```

Do not fine-tune in this task.

Do not train or modify the checkpoint.

Do not change preprocessing.

## Dependency

Run Task 23 first.

Use the best recommended tuned setting from:

```txt
outputs/vectorization/phase4_raster2graph_tuned_inference/run001/summary.md
```

If Task 23 has not been completed yet, use this fallback setting:

```txt
first_step_threshold = 0.20
later_step_threshold = 0.25
first_step_force_best = true
edge_search_threshold = 8
monte_times = 5
max_candidates_per_step = 12
```

## Input Source

Use the same single best preprocessing variant as Task 23:

```txt
outputs/vectorization/phase4_raster2graph_preprocessing_test/run001/<sample_id>/crop512_margin05/input.png
```

Do not process:

```txt
official_512
crop512_margin10
crop512_thick1
crop512_thick2
unit*_crop512
```

## Output Location

Write outputs under:

```txt
outputs/vectorization/phase4_raster2graph_recovery_inference/run001/<sample_id>/<setting>/
```

Each setting folder should contain:

```txt
input.png
graph_pred.svg
graph_overlay.png
graph_overlay_rescue_debug.png
metrics.json
notes.txt
```

Also write:

```txt
outputs/vectorization/phase4_raster2graph_recovery_inference/run001/summary.json
outputs/vectorization/phase4_raster2graph_recovery_inference/run001/summary.md
```

## Implementation Base

Create a new script:

```txt
external/raster_to_graph/run_inference_recovery.py
```

Use the cleaned/tuned inference code from Task 23 as the starting point.

Do not overwrite:

```txt
external/raster_to_graph/run_inference_sweep.py
external/raster_to_graph/run_inference_permissive_sweep.py
external/raster_to_graph/run_inference_margin05_tuned.py
```

if they exist.

## Recovery Strategy 1 - Reserve Low-Confidence Candidates

During each autoregressive step, keep two candidate lists:

```txt
main_candidates:
  score >= normal threshold

reserve_candidates:
  reserve threshold <= score < normal threshold
```

Recommended thresholds:

```txt
normal first-step threshold = best Task 23 first_step_threshold
normal later-step threshold = best Task 23 later_step_threshold
reserve_candidate_threshold = 0.05
```

If the main candidate list fails or becomes empty:

```txt
try the best reserve candidates before stopping
```

Record every reserve use.

Do not silently treat reserve candidates as normal predictions.

## Recovery Strategy 2 - Early Stop Rescue

The model may predict the end token too early.

Add:

```txt
ignore_early_end_until_min_nodes = 8
```

Optional comparison:

```txt
ignore_early_end_until_min_nodes = 10
```

If an end token appears before the graph has the minimum node count:

```txt
remove the end-token candidate
try the next best non-end candidate
if no non-end candidate exists, try reserve candidates
only stop if recovery fails
```

Record:

```txt
early_end_ignored_count
first_ignored_end_step
```

## Recovery Strategy 3 - Edge Connection Retry

A candidate node may be rejected because it cannot connect within the standard line-search tolerance.

Add controlled retry:

```txt
normal edge_search_threshold = best Task 23 edge_search_threshold
retry edge_search_threshold = 12
```

If a candidate set fails the inter-level or intra-level connection check:

```txt
retry once using edge_search_threshold = 12
```

Do not make `12` the default unless it clearly improves quality.

Record:

```txt
edge_retry_used_count
steps_recovered_by_edge_retry
```

## Recovery Strategy 4 - Limited Step Retry

For each autoregressive step, allow a small number of recovery attempts:

```txt
max_recovery_attempts_per_step = 2
```

Optional comparison:

```txt
max_recovery_attempts_per_step = 3
```

Recovery order:

```txt
1. remove early end token if graph is too small
2. try main candidates with normal edge tolerance
3. try main candidates with retry edge tolerance
4. try reserve candidates with normal edge tolerance
5. try reserve candidates with retry edge tolerance
6. stop only if all attempts fail
```

Keep this controlled. Do not endlessly search.

## Recovery Strategy 5 - Optional Multi-Component Restart

This is lower priority than step recovery.

If the graph stops with a partial output, optionally attempt one new component start:

```txt
max_component_restarts = 1
```

Restart idea:

```txt
keep the generated graph drawn on the tensor
ask the model for a new non-end candidate not overlapping existing predicted nodes
start a second component if the candidate is far enough from existing graph nodes
```

Recommended safety rules:

```txt
minimum distance from existing predicted node = 20 px
restart threshold = 0.10
restart only if generated graph has at least 8 nodes
restart only if the previous stop was not a clean end after sufficient graph size
```

Record second-component nodes separately.

This option is useful for wide or multi-unit plans, but it can also hallucinate disconnected fragments. Keep it off by default for the first run unless Task 23 shows many high-quality partial graphs.

## Recommended Setting Matrix

Run a small controlled matrix.

```txt
recovery_reserve_end_tol12
  reserve_candidate_threshold = 0.05
  ignore_early_end_until_min_nodes = 8
  retry_edge_search_threshold = 12
  max_recovery_attempts_per_step = 2
  max_component_restarts = 0

recovery_reserve_end_only
  reserve_candidate_threshold = 0.05
  ignore_early_end_until_min_nodes = 8
  retry_edge_search_threshold = null
  max_recovery_attempts_per_step = 2
  max_component_restarts = 0

recovery_reserve_end_tol12_restart1
  reserve_candidate_threshold = 0.05
  ignore_early_end_until_min_nodes = 8
  retry_edge_search_threshold = 12
  max_recovery_attempts_per_step = 2
  max_component_restarts = 1

baseline_tuned_no_recovery
  use best Task 23 setting
  recovery disabled
```

If runtime is too high, run only:

```txt
recovery_reserve_end_tol12
baseline_tuned_no_recovery
```

## Visual Debug Requirements

The normal overlay should show the final graph.

The rescue debug overlay should distinguish:

```txt
normal predicted nodes: red
normal predicted edges: blue
reserve/recovered nodes: orange
reserve/recovered edges: purple
component-restart nodes: green
component-restart edges: dark green
ignored early end token marker: violet at candidate point, if visualizable
```

If exact coloring is difficult, at minimum write separate JSON metadata for each predicted node/edge:

```txt
source = normal | reserve | edge_retry | component_restart
step_index
score
edge_class
```

## Metrics

For each sample/setting, write:

```json
{
  "sample_id": "1316",
  "source_variant": "crop512_margin05",
  "setting": "recovery_reserve_end_tol12",
  "base_first_step_threshold": 0.20,
  "base_later_step_threshold": 0.25,
  "reserve_candidate_threshold": 0.05,
  "ignore_early_end_until_min_nodes": 8,
  "base_edge_search_threshold": 8,
  "retry_edge_search_threshold": 12,
  "max_recovery_attempts_per_step": 2,
  "max_component_restarts": 0,
  "gt_node_count": 19,
  "gt_edge_count": 17,
  "node_count_bin": "10-50",
  "num_points": 15,
  "num_edges": 17,
  "normal_points": 12,
  "recovered_points": 3,
  "component_restart_points": 0,
  "stop_code": 2,
  "empty": false,
  "early_end_ignored_count": 1,
  "reserve_used_count": 2,
  "edge_retry_used_count": 1,
  "recovery_success_count": 2,
  "elapsed_s": 1.4
}
```

Also write per-attempt metadata if `monte_times > 1`.

## Summary Requirements

`summary.md` should compare recovery against the tuned baseline:

```txt
empty rate before/after recovery
average predicted nodes before/after recovery
average predicted edges before/after recovery
normal-stop rate before/after recovery
samples improved by recovery
samples made worse/noisier by recovery
samples where reserve candidates were essential
samples where early end token was ignored
samples where edge retry was essential
results by node_count_bin
recommendation: keep recovery on/off for next phase
```

Include a short qualitative note after visually checking overlays:

```txt
Does recovery continue plausible missing walls?
Or does recovery mostly add noisy hallucinated fragments?
```

## Interpretation

### Good Recovery

```txt
partial graphs continue into plausible missing floorplan regions
recovered nodes align with real wall junctions
edge count increases without obvious random fragments
10-50 and 51-80 node samples improve most
component restart helps only where the plan is visibly multi-component
```

### Bad Recovery

```txt
recovered nodes drift away from walls
edge count increases but graph quality gets worse
component restart creates disconnected hallucinations
early-end ignoring causes long noisy tails
large 80+ node plans still fail
```

If recovery is bad, prefer the tuned Task 23 setting and move toward fine-tuning or component-wise preprocessing instead.

## Acceptance Criteria

1. A new recovery inference script exists and does not overwrite prior sweep scripts.
2. The script uses only `crop512_margin05/input.png` from the preprocessing-test output folder.
3. The tuned Task 23 best setting is used as the baseline, or the fallback tuned setting is documented.
4. Reserve candidate recovery is implemented and logged.
5. Early end-token ignoring is implemented and logged.
6. Edge retry is implemented and logged, at least for one setting.
7. Component restart is either implemented as an optional setting or explicitly deferred in `summary.md`.
8. Recovered nodes/edges are visually or structurally distinguishable from normal predictions.
9. Outputs are written under `outputs/vectorization/phase4_raster2graph_recovery_inference/run001/`.
10. `summary.md` recommends whether recovery should become part of the next Raster-to-Graph inference pipeline.
11. No fine-tuning, training, or checkpoint modification happens in this task.

