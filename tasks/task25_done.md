# Task 25 - Raster-to-Graph Multi-Start Disconnected Component Inference

## Objective

Implement an inference-only Raster-to-Graph mode that can generate multiple disconnected graph components from one floorplan image.

Current problem:

```txt
The autoregressive graph starts in one region.
When that graph traversal stops, generation ends.
If the remaining floorplan region was never reached by the traversal, no graph is generated there.
```

New desired behavior:

```txt
Generate one graph component.
When it stops, search for a new significant starting point in another region.
Start a new graph component.
Repeat until no significant start is found or the restart cap is reached.
```

Disconnected graph components are acceptable in this task.

This task is still inference-only:

```txt
Do not fine-tune.
Do not train.
Do not modify the checkpoint.
Do not change preprocessing.
```

## Dependency

Use the best practical Task 23 tuned setting as the base inference configuration.

If needed, use this fallback base setting:

```txt
first_step_threshold = 0.20
later_step_threshold = 0.25
first_step_force_best = true
edge_search_threshold = 8
monte_times = 5
max_candidates_per_step = 12
```

Task 24 showed that recovery inside the same traversal has little effect. This task should not focus on same-frontier recovery. It should focus on starting new disconnected components.

## Input Source

Use only the best preprocessing variant:

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
outputs/vectorization/phase4_raster2graph_multistart_inference/run001/<sample_id>/<setting>/
```

Each setting folder should contain:

```txt
input.png
graph_pred.svg
graph_overlay.png
graph_overlay_components.png
metrics.json
components.json
notes.txt
```

Also write:

```txt
outputs/vectorization/phase4_raster2graph_multistart_inference/run001/summary.json
outputs/vectorization/phase4_raster2graph_multistart_inference/run001/summary.md
```

## Implementation Base

Create a new script:

```txt
external/raster_to_graph/run_inference_multistart.py
```

Use the Task 23 tuned script as the main base.

You may reuse parts of Task 24 recovery code, but do not let same-frontier recovery become the main mechanism.

## Core Algorithm

For each input image:

```txt
1. Run normal tuned R2G inference to generate component 0.
2. Save accepted nodes/edges for component 0.
3. When the component stops, keep its generated graph drawn or masked.
4. Search for a new significant start candidate that is far from existing components.
5. If a valid start is found, begin component 1 from that candidate.
6. Repeat until:
   - no significant start candidate is found
   - or component restart count reaches 4
```

Cap:

```txt
max_new_starts = 4
```

This means a sample may have:

```txt
1 initial component + up to 4 additional components = up to 5 components total
```

## New Start Search

After a component stops, run the model on the current tensor state and inspect candidate predictions.

A new start candidate should satisfy:

```txt
edge class is not 0
edge class is not 16 end token
score >= restart_start_threshold
distance from all existing graph nodes >= restart_min_distance_px
candidate is not inside an already covered component region
candidate is not near the violet end-token marker
```

Recommended:

```txt
restart_start_threshold = 0.10
restart_min_distance_px = 24
restart_max_candidates_checked = 50
```

If no candidate passes:

```txt
stop multi-start inference
```

Do not force a new start if the only candidates are weak/noisy.

## Covered Region Mask

Maintain a simple covered-region mask from generated components.

At minimum:

```txt
draw all accepted component nodes as disks with radius 16 px
draw all accepted component edges as thick lines with radius/width 16 px
```

A restart candidate inside this covered mask should be rejected.

Recommended optional expansion:

```txt
covered_mask_dilation_px = 12
```

This prevents the model from repeatedly restarting in the same graph area.

## Starting A New Component

The original R2G model expects an empty graph start/top-left traversal convention.

For multi-start, do not assume the model will naturally choose a new region.

Use a controlled restart seed:

```txt
select the best restart candidate
append it as the first prediction of the new component
draw that candidate node onto the tensor
continue normal autoregressive decoding from there
```

If the implementation cannot cleanly seed a component this way, document the limitation and implement the closest feasible version:

```txt
rerun normal inference after masking/covering previous components
take the first valid far-away component as new component
```

But prefer explicit restart seeding if feasible.

## Component Stopping

Each component should stop independently.

A component stops when:

```txt
end token is accepted
no valid next candidates exist
max_component_steps is reached
graph becomes too noisy
```

Recommended:

```txt
max_component_steps = 200
min_component_points = 3
```

If a new component produces fewer than `min_component_points`, discard it unless visual inspection suggests it is a valid tiny wall fragment.

Record discarded components in metrics.

## Candidate And Threshold Settings

Use the Task 23 tuned setting for normal decoding.

Recommended multi-start settings:

```txt
multistart_tuned_restart010_dist24_cap4
  first_step_threshold = 0.20
  later_step_threshold = 0.25
  edge_search_threshold = 8
  monte_times = 5
  max_candidates_per_step = 12
  restart_start_threshold = 0.10
  restart_min_distance_px = 24
  max_new_starts = 4
  covered_mask_dilation_px = 12

multistart_tuned_restart015_dist24_cap4
  first_step_threshold = 0.20
  later_step_threshold = 0.25
  edge_search_threshold = 8
  monte_times = 5
  max_candidates_per_step = 12
  restart_start_threshold = 0.15
  restart_min_distance_px = 24
  max_new_starts = 4
  covered_mask_dilation_px = 12

baseline_tuned_single_start
  use best Task 23 tuned setting
  max_new_starts = 0
```

If runtime is high, run only:

```txt
multistart_tuned_restart010_dist24_cap4
baseline_tuned_single_start
```

## Component Metadata

Write `components.json` with:

```json
{
  "sample_id": "12539",
  "components": [
    {
      "component_id": 0,
      "source": "initial",
      "num_points": 20,
      "num_edges": 24,
      "stop_reason": "no_valid_candidates",
      "start_point": [120, 80],
      "start_score": 0.42,
      "accepted": true
    },
    {
      "component_id": 1,
      "source": "restart",
      "num_points": 12,
      "num_edges": 13,
      "stop_reason": "end_token",
      "start_point": [350, 95],
      "start_score": 0.18,
      "accepted": true
    }
  ],
  "discarded_components": []
}
```

## Metrics

For each sample/setting, write:

```json
{
  "sample_id": "12539",
  "source_variant": "crop512_margin05",
  "setting": "multistart_tuned_restart010_dist24_cap4",
  "max_new_starts": 4,
  "restart_start_threshold": 0.10,
  "restart_min_distance_px": 24,
  "covered_mask_dilation_px": 12,
  "num_components": 2,
  "num_restarts_attempted": 2,
  "num_restarts_accepted": 1,
  "num_restarts_discarded": 1,
  "total_points": 32,
  "total_edges": 37,
  "single_start_points": 20,
  "single_start_edges": 24,
  "additional_points_from_restarts": 12,
  "additional_edges_from_restarts": 13,
  "empty": false,
  "elapsed_s": 2.4
}
```

Also include node-count metadata if the ground-truth wall graph exists:

```txt
gt_node_count
gt_edge_count
node_count_bin
```

Use bins:

```txt
10-50
51-80
81-120
120+
```

## Visualization

`graph_overlay_components.png` should color components separately.

Suggested colors:

```txt
component 0: blue edges, red nodes
component 1: green edges, green nodes
component 2: purple edges, purple nodes
component 3: orange edges, orange nodes
component 4: cyan edges, cyan nodes
discarded candidate starts: gray circles
```

The visualization should make it obvious whether restarts find real missing regions or just duplicate/noisy fragments.

## Summary Requirements

`summary.md` should compare multi-start against single-start tuned baseline:

```txt
empty rate before/after
average total points/edges before/after
average number of components
average accepted restarts
samples improved by restarts
samples where restarts duplicate existing graph area
samples where restarts add noisy fragments
samples where no significant start was found
results by node_count_bin
recommendation: keep multi-start on/off
recommended restart threshold
recommended restart distance
```

Include a short qualitative visual-inspection note:

```txt
Do additional components align with missing floorplan regions?
Do they improve coverage without damaging graph quality?
```

## Interpretation

### Good Multi-Start Behavior

```txt
new components appear in previously missing plan regions
components do not duplicate the first component
total graph coverage improves
component count stays small
restart starts have visible architectural support
```

### Bad Multi-Start Behavior

```txt
restarts repeatedly occur near the original component
new components are tiny or noisy
restart starts land on text, furniture, or symbols
large samples get more nodes but not better structure
```

If multi-start is bad, keep the Task 23 tuned single-start setting and move toward component-wise image cropping or fine-tuning.

## Acceptance Criteria

1. A new multi-start inference script exists.
2. The script processes only `crop512_margin05/input.png` samples.
3. Single-start tuned baseline is included for comparison.
4. New disconnected graph components are allowed.
5. Additional starts are capped at `max_new_starts = 4`.
6. Restart candidates must be far from existing graph nodes and outside the covered mask.
7. Tiny/noisy restart components are discarded or clearly flagged.
8. `components.json` records each component and stop reason.
9. `graph_overlay_components.png` colors components separately.
10. Outputs are written under `outputs/vectorization/phase4_raster2graph_multistart_inference/run001/`.
11. `summary.md` recommends whether multi-start should become part of the next inference pipeline.
12. No fine-tuning, training, or checkpoint modification happens in this task.

