# Task 23 - Raster-to-Graph Tuned Inference Sweep On Best Variant

## Objective

The previous permissive Raster-to-Graph inference sweep showed clear improvement:

```txt
lower thresholds produced more graph outputs
many generated graphs were mostly accurate
crop512_margin05 performed best overall
```

This task should regenerate the inference process with stricter thresholds and cleaner pass criteria than the first permissive sweep, while still staying more permissive than the original Raster-to-Graph demo.

The goal is to find a practical inference configuration that produces useful graph predictions without requiring fine-tuning yet.

Do not fine-tune in this task.

Do not train or modify the checkpoint.

Do not rerun all preprocessing variants.

Only run the model on:

```txt
crop512_margin05
```

## Input Source

Use the sample images already generated in:

```txt
outputs/vectorization/phase4_raster2graph_preprocessing_test/run001/
```

For each sample, use only:

```txt
<sample_id>/crop512_margin05/input.png
```

Do not use:

```txt
official_512
crop512_margin10
crop512_thick1
crop512_thick2
unit*_crop512
```

Those variants are no longer needed for this task.

## Checkpoint

Use the existing pretrained Raster-to-Graph checkpoint:

```txt
checkpoints_Raster2Graph/checkpoint0299.pth
```

Implementation base:

```txt
external/raster_to_graph/run_inference_permissive_sweep.py
```

If that file is messy from the first diagnostic task, create a cleaner follow-up script:

```txt
external/raster_to_graph/run_inference_margin05_tuned.py
```

Do not overwrite Task 22 or previous Task 23 outputs.

## Output Location

Write outputs under:

```txt
outputs/vectorization/phase4_raster2graph_tuned_inference/run001/<sample_id>/<setting>/
```

Each setting folder should contain:

```txt
input.png
graph_pred.svg
graph_overlay.png
metrics.json
notes.txt
```

Also write:

```txt
outputs/vectorization/phase4_raster2graph_tuned_inference/run001/summary.json
outputs/vectorization/phase4_raster2graph_tuned_inference/run001/summary.md
```

## Main Change From Previous Task

The previous task intentionally lowered boundaries extremely.

This task should push thresholds and graph acceptance standards upward to reduce noisy extra predictions while preserving the improved non-empty rate.

Use the previous results as evidence that relaxed decoding helps, but now search for a balanced setting.

## Threshold Sweep

Use thresholds centered around:

```txt
0.15
0.20
0.25
0.30
```

Recommended settings:

```txt
tuned_015_020_tol8_mc5
  first_step_threshold = 0.15
  later_step_threshold = 0.20
  first_step_force_best = true
  edge_search_threshold = 8
  monte_times = 5

tuned_020_025_tol8_mc5
  first_step_threshold = 0.20
  later_step_threshold = 0.25
  first_step_force_best = true
  edge_search_threshold = 8
  monte_times = 5

tuned_020_030_tol8_mc5
  first_step_threshold = 0.20
  later_step_threshold = 0.30
  first_step_force_best = true
  edge_search_threshold = 8
  monte_times = 5

tuned_025_030_tol8_mc5
  first_step_threshold = 0.25
  later_step_threshold = 0.30
  first_step_force_best = true
  edge_search_threshold = 8
  monte_times = 5

tuned_020_025_tol5_mc5
  first_step_threshold = 0.20
  later_step_threshold = 0.25
  first_step_force_best = true
  edge_search_threshold = 5
  monte_times = 5

baseline_task22_margin05
  first_step_threshold = 0.30
  later_step_threshold = 0.30
  first_step_force_best = false
  edge_search_threshold = 5
  monte_times = 1
```

If runtime is too high, run only:

```txt
tuned_015_020_tol8_mc5
tuned_020_025_tol8_mc5
tuned_025_030_tol8_mc5
baseline_task22_margin05
```

## First-Step Fallback

Keep first-step fallback enabled for the tuned settings:

```txt
first_step_force_best = true
```

But record whether it was actually used.

If a sample only works when fallback is used, mark that in the summary.

Fallback rules:

```txt
Only apply at iter_time == 0.
Prefer a non-end candidate.
Prefer candidates where edge class is not 0 and not 16.
Record selected candidate score and edge class.
Do not use fallback for the baseline setting.
```

## Candidate Control

The previous permissive task allowed weak predictions to test whether decoding could start.

This task should reduce noisy predictions by capping candidates per step.

Recommended:

```txt
max_candidates_per_step = 12
```

Optional comparison:

```txt
max_candidates_per_step = 8
max_candidates_per_step = 16
```

If adding candidate-count variants creates too many runs, keep only `12`.

## Edge Search Tolerance

Default tuned tolerance:

```txt
edge_search_threshold = 8
```

Also test one tighter setting:

```txt
edge_search_threshold = 5
```

Do not test `12` unless graphs are still ending too early. The previous task already showed that very loose boundaries can increase output; this task should now clean that up.

## Monte Carlo Attempts

Use:

```txt
monte_times = 5
```

Keep best result by:

```txt
1. prefer normal stop_code == 2
2. then maximize edge count
3. then maximize node count
4. then prefer fewer fallback uses
```

Do not use `monte_times = 10` unless the output is still too unstable.

## Node Count Handling

For each sample, read the original wall graph if available:

```txt
docs/high_quality_architectural/<sample_id>/masks/wall_graph.json
```

Record:

```txt
gt_node_count
gt_edge_count
node_count_bin
```

Use bins:

```txt
10-50       R2G-like
51-80       slightly above original range
81-120      large
120+        very large / likely out of distribution
```

Node count is not the top priority for this task, but the summary must show whether tuned inference is mainly working on the R2G-like group or also pushing into the 51-80 group.

If a sample is over 80 nodes and still fails, do not spend extra time tuning around it in this task.

## Metrics

For each sample/setting, write:

```json
{
  "sample_id": "1316",
  "source_variant": "crop512_margin05",
  "setting": "tuned_020_025_tol8_mc5",
  "first_step_threshold": 0.20,
  "later_step_threshold": 0.25,
  "first_step_force_best": true,
  "first_step_fallback_used": false,
  "first_step_fallback_score": null,
  "edge_search_threshold": 8,
  "monte_times": 5,
  "max_candidates_per_step": 12,
  "gt_node_count": 19,
  "gt_edge_count": 17,
  "node_count_bin": "10-50",
  "num_points": 12,
  "num_edges": 14,
  "stop_code": 2,
  "empty": false,
  "elapsed_s": 0.8
}
```

## Summary Requirements

`summary.md` should include:

```txt
best overall tuned setting
empty rate by setting
normal-stop rate by setting
average predicted nodes/edges by setting
samples improved over baseline_task22_margin05
samples worse than baseline_task22_margin05
samples that only work with first_step_force_best
results by node_count_bin
recommended final inference setting for next phase
```

## Interpretation

### Good Tuned Setting

```txt
keeps most of the non-empty gains from permissive inference
reduces obvious extra/random nodes
produces mostly accurate wall graphs on crop512_margin05
does not rely on fallback for most samples
works especially well for 10-50 and 51-80 node samples
```

### Too Strict

```txt
many samples return to empty output
first-step fallback is needed frequently
predicted graph count drops below useful levels
```

### Too Loose

```txt
graphs are non-empty but noisy
extra wall nodes appear away from real junctions
edge count is inflated
normal-stop rate is low
```

## Acceptance Criteria

1. `tasks/task23.md` is the active task file for this tuned follow-up.
2. The previous broad permissive task is no longer the active instructions.
3. The model is applied only to `crop512_margin05/input.png` samples under `outputs/vectorization/phase4_raster2graph_preprocessing_test/run001/`.
4. No margin10, thick1, thick2, official512, or unit-crop variants are processed in this task.
5. The tuned threshold matrix is run, or a reduced version is documented if runtime is too high.
6. Outputs are written under `outputs/vectorization/phase4_raster2graph_tuned_inference/run001/`.
7. `summary.md` recommends one final inference configuration for the next phase.
8. No fine-tuning, training, or checkpoint modification happens in this task.

