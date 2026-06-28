# Spec v010: Raster-to-Graph Local Modifications

## 0. Purpose

This is a living record of how this project modifies the original Raster-to-Graph process.

Whenever the Raster-to-Graph preprocessing, inference, decoding, output format, or training plan changes, update this file.

The goal is to keep a clear separation between:

```txt
original Raster-to-Graph repo / paper behavior
local neural_floorplan adaptations
experimental inference settings
future changes that should be preserved
```

## Current Settled Status

The current Phase 4 method is settled as pretrained Raster-to-Graph inference. Fine-tuning is not needed for the current project version because the local preprocessing and inference modifications now produce satisfactory wall graphs.

Current method summary:

```txt
model_clean.png
-> crop content bbox
-> add true 20% white padding around the crop
-> scale long edge to 512 px and center on white canvas
-> pretrained checkpoint0299.pth inference
-> generous thresholds and MC reranking
-> hard/soft graph validity scoring
-> mask-and-rerun multistart recovery
-> merge-on-intersection
-> light post-merge filtering
-> graph_pred.json / graph_pred.svg / overlays / metrics
```

Current output folder:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/<sample_id>/
```

Training/fine-tuning notes in older specs are retained as historical fallback information only.

## 1. Original Raster-to-Graph Baseline

The original Raster-to-Graph process is an autoregressive graph-prediction model.

Core behavior:

```txt
floorplan raster
-> 512x512 centered RGB input
-> ResNet + Deformable DETR style model
-> predict next graph frontier
-> draw accepted predicted graph state back onto the input tensor
-> repeat until no valid prediction or end token
```

Important original assumptions:

```txt
input raster is normalized with original repo mean/std
graph nodes are wall junctions
graph edges are wall segments
walls are horizontal / vertical
training data is filtered to connected graph-like plans
many original samples are roughly 10-50 graph nodes
```

Original demo-style inference behavior:

```txt
long edge scaled to 512 px
centered on 512x512 white canvas
score threshold around 0.5 in demo.py
monte_times = 1
single connected autoregressive traversal
stop whole generation when no valid next prediction exists
stop whole generation when end token is accepted
```

## 2. External Code And Checkpoint Use

This project uses the official-style Raster-to-Graph code copied under:

```txt
external/raster_to_graph/
```

The pretrained checkpoint is stored at:

```txt
checkpoints_Raster2Graph/checkpoint0299.pth
```

The checkpoint is treated as an external pretrained model.

Local code should preserve attribution and license notices for copied upstream code.

The project README cites the Raster-to-Graph paper and upstream repository.

## 3. Input Source Modification

Original Raster-to-Graph was trained on its own raster dataset.

This project first tried applying the pretrained checkpoint directly to local floorplan samples. Direct use often produced either:

```txt
highly accurate graph
or no graph at all
```

The local input source was then standardized to this project's clean CubiCasa render:

```txt
docs/high_quality_architectural/<sample>/model_clean.png
```

Rationale:

```txt
model_clean.png is cleaner and more controllable than F1_original.png / F1_scaled.png
it isolates graph prediction from scan/listing/image noise
it is generated from the same SVG source used for labels
```

Current inference experiments no longer use all raw sample files directly. They use preprocessed R2G-ready PNGs generated from `model_clean.png`.

## 4. Target Graph Simplification

The local Phase 4 goal was narrowed to wall graph prediction first.

Current local graph target:

```txt
wall junction nodes
orthogonal wall segment edges
wall graph only
```

Deferred to later vectorization/CAD stages:

```txt
door nodes
window nodes
room labels
semantic CAD classification
```

Rationale:

```txt
the pretrained R2G model already has a strict graph-generation problem
doors/windows increase target complexity
the existing 7-class segmentation model can still support door/window attachment later
```

## 5. Output Folder Organization

Old direct test output:

```txt
outputs/raster2graph/
```

This folder was testing-only and should not be used for current Phase 4 outputs.

Current Phase 4 outputs are grouped under:

```txt
outputs/vectorization/
```

Historical Raster-to-Graph experiment folders:

```txt
outputs/vectorization/phase4_raster2graph_preprocessing_test/
outputs/vectorization/phase4_raster2graph_permissive_inference/
outputs/vectorization/phase4_raster2graph_tuned_inference/
outputs/vectorization/phase4_raster2graph_recovery_inference/
outputs/vectorization/phase4_raster2graph_multistart_inference/
```

Current settled output folder:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/<sample_id>/
```

Preferred organization:

```txt
one folder per sample
inside that folder, keep input PNG, output JSON/SVG/overlay/metrics together
```

Do not organize current outputs only by file type.

## 6. Preprocessing Sweep Modification

Original R2G demo preprocessing:

```txt
scale long edge to 512
center on white 512x512 canvas
normalize with original mean/std
```

Local Task 22 expanded preprocessing as an inference-only sweep.

Generated variants:

```txt
official_512
crop512_margin05
crop512_margin10
crop512_thick1
crop512_thick2
unit*_crop512
```

Observed result:

```txt
crop512_margin05 performed best overall
margin10 and thickening were less useful
thickening sometimes made outputs worse or did not materially help
unit crops are useful only as a separate strategy for multi-unit / large plans
```

Current preferred input variant:

```txt
crop512_margin05/input.png
```

Current source for tuned and future inference:

```txt
outputs/vectorization/phase4_raster2graph_preprocessing_test/run001/<sample_id>/crop512_margin05/input.png
```

## 7. Threshold And Decoding Modification

Original demo-style inference uses a stricter threshold and stops quickly.

Local inference changed this because many samples failed at the first autoregressive step.

Important failure mode:

```txt
if no valid first node survives threshold/type filtering
-> is_stop(this_preds) returns stop
-> graph output is empty
```

Task 23 tuned inference uses lower but not ultra-low thresholds.

Current recommended fallback setting if no summary is available:

```txt
first_step_threshold = 0.20
later_step_threshold = 0.25
first_step_force_best = true
edge_search_threshold = 8
monte_times = 5
max_candidates_per_step = 12
```

Baseline comparison:

```txt
first_step_threshold = 0.30
later_step_threshold = 0.30
first_step_force_best = false
edge_search_threshold = 5
monte_times = 1
```

Observed result:

```txt
tuned thresholds reduced empty predictions
crop512_margin05 + tuned thresholds generated more useful graphs
first_step_force_best helped at least one previously empty sample
```

## 8. Candidate And Monte Carlo Modification

Original demo uses:

```txt
monte_times = 1
```

Local tuned inference uses:

```txt
monte_times = 5
```

Best result selection:

```txt
prefer normal end stop_code == 2
then maximize edge count
then maximize node count
then prefer fewer fallback uses
```

Candidate count is capped to reduce noisy additions:

```txt
max_candidates_per_step = 12
```

This is a local inference stability mechanism, not part of the original paper baseline.

## 9. Node Count Analysis Modification

The original R2G paper filtered its dataset heavily.

Local experiments therefore track ground-truth graph size:

```txt
10-50       R2G-like
51-80       slightly above original range
81-120      large
120+        very large / likely out of distribution
```

This is analysis metadata, not an inference rule.

Current interpretation:

```txt
10-50 and 51-80 are the main target ranges for pretrained-checkpoint inference
80+ samples may need multi-start, component-wise crops, or future fine-tuning
```

## 10. Recovery Attempt Modification

Task 24 tested recovery within the same autoregressive traversal.

Recovery ideas:

```txt
reserve low-confidence candidates
ignore early end token until minimum node count
retry edge connection with wider tolerance
optional single restart
```

Output folder:

```txt
outputs/vectorization/phase4_raster2graph_recovery_inference/
```

Observed result:

```txt
recovery produced little or no meaningful difference from tuned inference
once the model stopped, same-frontier recovery usually did not continue the missing region
```

Current conclusion:

```txt
same-traversal recovery is not the main path forward
keep it documented, but do not rely on it as the final inference strategy
```

## 11. Multi-Start Disconnected Component Modification

Task 25 introduces the next major local change:

```txt
when one autoregressive graph component stops,
search for a new significant start in another region,
generate another disconnected graph component,
repeat until no significant start is found or restart cap is reached
```

Disconnected components are acceptable.

Current cap:

```txt
max_new_starts = 4
```

This means:

```txt
1 initial component + up to 4 additional starts = up to 5 components total
```

Planned script:

```txt
external/raster_to_graph/run_inference_multistart.py
```

Planned output folder:

```txt
outputs/vectorization/phase4_raster2graph_multistart_inference/run001/
```

Restart candidate constraints:

```txt
edge class is not 0
edge class is not 16 end token
score >= restart_start_threshold
distance from all existing graph nodes >= restart_min_distance_px
outside covered component mask
```

Recommended settings:

```txt
restart_start_threshold = 0.10 or 0.15
restart_min_distance_px = 24
covered_mask_dilation_px = 12
max_new_starts = 4
```

Required metadata:

```txt
components.json
component-colored overlay
stop reason per component
accepted/discarded restart counts
```

## 12. Generous Inference And Mask-And-Rerun (Task 26)

Task 26 introduces a new inference strategy replacing manual restart seeding from Task 25.

### Status: recommended (supersedes Task 25 multistart)

### Inference Settings

```txt
first_step_threshold    = 0.10   (experimental → recommended)
later_step_threshold    = 0.10   (experimental → recommended)
first_step_force_best   = true
edge_search_threshold   = 30 px  (up from 8 px in Task 23/25)
monte_times             = 7      (up from 5)
max_candidates_per_step = 20     (up from 12)
```

Rationale:
```txt
model-accepted graphs are geometrically accurate when produced
failure mode is low production rate (empty output)
30px edge_search_threshold trades pixel-exact alignment for higher graph coverage
10/10 thresholds ensure first-step rarely fails
```

### Mask-And-Rerun Multi-Start (replaces Task 25 manual restart seeding)

```txt
1. run generous inference → component 0
2. build covered mask from component 0's nodes and edges
3. apply white suppression to covered pixels in source PIL image
4. reset tensor state entirely (normalize fresh suppressed image)
5. rerun generous inference on suppressed image → component 1
6. accept if significant (min_component_points=3, min_component_edges=2)
7. repeat until no significant component or max_new_starts cap reached
```

Settings:
```txt
covered_node_radius_px   = 24
covered_edge_width_px    = 30
covered_mask_dilation_px = 20
suppression_fill         = white
min_component_points     = 3
min_component_edges      = 2
max_new_starts           = 4
```

Why mask-and-rerun over Task 25 manual restart seeding:
```txt
Task 25 seeded a new node on the existing tensor and continued same traversal.
_assemble_step requires last_edges to point back to prior nodes.
Isolated new seed nodes violated this constraint → all restart components discarded.
Mask-and-rerun resets completely — fresh inference, no constraint violations.
```

### Merge-On-Intersection

After all components are accepted, merge them into one graph:

```txt
1. snap nodes within node_snap_tolerance_px = 10
2. find H-V crossing edges; insert junction nodes; split at crossing
3. merge overlapping collinear parallel edges
4. re-snap after splits
```

Settings:
```txt
node_snap_tolerance_px         = 10
edge_intersection_tolerance_px =  8
collinear_overlap_tolerance_px =  8
```

Final `graph_pred.json` / `graph_pred.svg` represent the merged graph.
`components.json` preserves original component membership before merge.

### Output Folder

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/<sample_id>/
```

One flat folder per sample. No nested setting/iteration sub-folders.

Required files per sample:
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

### Script

```txt
external/raster_to_graph/run_inference_generous_phase4.py
```

This is the current Phase 4 inference entrypoint.

### Old Phase 4 Output Folders (retired)

These folders are cleaned up automatically by `--cleanup` (default):
```txt
outputs/vectorization/phase4_raster2graph_preprocessing_test/     retired
outputs/vectorization/phase4_raster2graph_permissive_inference/   retired
outputs/vectorization/phase4_raster2graph_tuned_inference/        retired
outputs/vectorization/phase4_raster2graph_recovery_inference/     retired
outputs/vectorization/phase4_raster2graph_multistart_inference/   retired
```

## 14. What Has Not Been Changed

These parts are not intentionally changed so far:

```txt
pretrained checkpoint weights
core model architecture
ResNet / Deformable DETR backbone-decoder structure
512x512 input canvas size
original repo mean/std normalization
axis-aligned wall assumption
external Raster-to-Graph attribution requirement
```

If any of these change later, update this file immediately.

## 13. Very Generous Inference, Hard Filters, And Validity Reranking (Task 27)

Task 27 updates `run_inference_generous_phase4.py` in-place with more permissive thresholds, hard post-generation filters, and validity-based MC reranking. Outputs replace `run001/` contents.

### Status: recommended (supersedes Task 26 settings)

### Generation Settings

```txt
first_step_threshold    = 0.05   (down from 0.10)
later_step_threshold    = 0.05   (down from 0.10)
first_step_force_best   = true
edge_search_threshold   = 40 px  (up from 30 px)
monte_times             = 10     (up from 7)
max_candidates_per_step = 30     (up from 20)
```

### Hard Filters (applied after generation, before export)

```txt
1. Angle filter
   discard edges not within ±10 degrees of horizontal or vertical
   records: edges_removed_angle_filter

2. Tiny component filter
   discard components with < 3 points or < 2 edges
   records: components_removed_tiny, points_removed_tiny_components, edges_removed_tiny_components

3. One-edge component filter
   discard single-edge components unless length >= 80 px AND wall_evidence >= 0.65
   records: one_edge_components_removed, one_edge_components_kept_by_evidence

4. Short dangling edge filter
   remove edges at degree-1 nodes if length <= 35 px AND wall_evidence < 0.45
   records: dangling_edges_removed
```

### Soft Scores (used for MC candidate reranking)

```txt
wall_evidence_alignment_score  — mean dark-pixel fraction along edge band (band_px=10, dark_thresh=180)
unsupported_edge_ratio         — fraction of edges with wall evidence < 0.35
rectangle_cycle_count          — count of orthogonal closed cycles (min_area=400px2, max_aspect=8.0)
rectangle_cycle_score          — normalized: min(count / 5, 1.0)
dangling_node_ratio            — degree-1 nodes / total nodes
dangling_penalty               — excess above 0.35 soft limit, clamped to [0,1]
small_component_count          — number of connected components after hard filters
small_component_penalty        — min(max(n-1, 0) / 4, 1.0)
```

### Candidate Reranking Formula

```txt
candidate_score =
  + 3.0 * wall_evidence_alignment_score
  + 2.0 * rectangle_cycle_score
  - 1.5 * dangling_penalty
  - 2.0 * unsupported_edge_ratio
  - 1.0 * small_component_penalty
  + 0.2 * normalized_edge_count
```

The MC attempt with the highest candidate_score is selected (not the one with most nodes/edges).

### Junction Support

Not implemented in Task 27. Deferred: junction pixel evidence is too strict at this stage.

### Output Folder

Same as Task 26 (in-place replacement):
```txt
outputs/vectorization/phase4_raster2graph_generous_inference/<sample_id>/
```

### metrics.json Extensions

```json
{
  "hard_filters": {
    "edges_removed_angle_filter": 0,
    "components_removed_tiny": 0,
    "points_removed_tiny_components": 0,
    "edges_removed_tiny_components": 0,
    "one_edge_components_removed": 0,
    "one_edge_components_kept_by_evidence": 0,
    "dangling_edges_removed": 0
  },
  "soft_scores": {
    "wall_evidence_alignment_score": 0.0,
    "rectangle_cycle_count": 0,
    "rectangle_cycle_score": 0.0,
    "dangling_node_count": 0,
    "dangling_penalty": 0.0,
    "unsupported_edge_ratio": 0.0,
    "small_component_count": 0,
    "small_component_penalty": 0.0,
    "candidate_score": 0.0
  },
  "components_before_merge": 0,
  "components_after_merge": 0,
  "selected_attempt": 0
}
```

Per-attempt reranking details available in `components.json` → `attempt_summaries`.

## 14. Extreme Threshold Inference And Standardized Margin (Task 28)

Task 28 updates `run_inference_generous_phase4.py` in-place with even more permissive thresholds and standardized 10% margin preprocessing.

### Status: recommended (supersedes Task 27 settings)

### Preprocessing Change

```txt
Previous: crop512_margin05  (5% of content bbox on each side)
Current:  crop512_margin10_standardized  (10% of content bbox on each side)
```

Rationale:
```txt
R2G generates graphs more reliably when walls are not near the canvas edge.
5% margin sometimes left walls within a few pixels of the 512px boundary.
10% standardized margin ensures consistent clearance.
Content-edge-touching is detected and recorded in metrics.json (content_touches_edge).
```

### Generation Settings

```txt
first_step_threshold    = 0.02   (down from 0.05)
later_step_threshold    = 0.02   (down from 0.05)
first_step_force_best   = true
edge_search_threshold   = 50 px  (up from 40 px)
monte_times             = 12     (up from 10)
max_candidates_per_step = 40     (up from 30)
```

Rationale:
```txt
Task 27 hard filters and soft scoring work well.
But not enough graph content was generated for them to filter effectively.
Extreme thresholds generate much more candidate content first.
Hard filters and validity reranking clean the result.
```

### Hard Filters And Soft Scoring

All Task 27 hard filters and soft scores are kept unchanged. See Section 13.

Junction support score is still not implemented.

### metrics.json Extension

```json
{
  "source_variant": "crop512_margin10_standardized",
  "content_touches_edge": false
}
```

### Output Folder

Same location (in-place replacement):
```txt
outputs/vectorization/phase4_raster2graph_generous_inference/<sample_id>/
```

## 16. Fast Runtime-Limited Smoke Test (Task 29)

Task 29 is a three-sample smoke test using reduced runtime settings to verify the Task 28 pipeline produces useful output before running the full 20-sample dataset.

### Status: smoke test (three samples only)

### Settings Reduced From Task 28

```txt
monte_times      = 4   (down from 12)
max_new_starts   = 2   (down from 4)
```

Everything else kept from Task 28:
```txt
first_step_threshold    = 0.02
later_step_threshold    = 0.02
edge_search_threshold   = 50 px
max_candidates_per_step = 40
crop512_margin10_standardized preprocessing
all hard filters and soft scoring
```

### Sample Selection

First three entries from `data/raster2graph/preprocess_test_samples.json`, all pointing to:
```txt
docs/high_quality_architectural/<sample_id>/model_clean.png
```

Run with: `python external/raster_to_graph/run_inference_generous_phase4.py --max-samples 3 --no-cleanup`

### Output Folder

Same location (in-place replacement, 3 sample folders only):
```txt
outputs/vectorization/phase4_raster2graph_generous_inference/<sample_id>/
```

## 15. Current Practical Pipeline (updated for Task 30)

Current inference-oriented Phase 4 pipeline:

```txt
model_clean.png
-> detect content bbox
-> crop exactly to content bbox
-> create new white image with 20% padding on each side (true padding)
-> scale long edge to 512px
-> center on 512x512 white canvas
-> detect and log content-edge-touching (should be false after true padding)
-> normalize with original R2G mean/std
-> MC inference (4 attempts × component)
  -> per attempt: full hard filters (angle / tiny / one-edge / dangling)
  -> per attempt: soft scoring (wall evidence / cycles / dangling / unsupported)
  -> select best attempt by candidate_score
-> mask-and-rerun multi-start (up to 2 additional components)
-> merge-on-intersection (snap[tol=6] + split[tol=8] + collinear merge[tol=8])
-> light post-merge filter (angle + dedup only — no tiny/one-edge/dangling deletion)
-> graph_overlay_components.png / graph_overlay_merged.png / graph_overlay.png
-> graph_pred.json / graph_pred.svg / metrics.json / components.json
```

Current best variant:

```txt
crop512_margin20_truepad
```

Current best inference family:

```txt
extreme (0.02/0.02, 50px, monte_times=4, max_new_starts=2)
+ full hard filters per MC attempt + validity reranking
+ mask-and-rerun + merge + light post-merge filter
script: external/raster_to_graph/run_inference_generous_phase4.py
```

## 18. True Padding Fix And Less Destructive Final Filtering (Task 30)

Task 30 fixes two issues observed in Task 29 outputs and updates `run_inference_generous_phase4.py` in-place.

### Status: recommended (supersedes Task 29 settings)

### Issue 1: Preprocessing — Clamped Bbox Expansion Retired

**Old behavior (retired)**:
```python
# Fails when content touches original image edge — margin on that side becomes 0
cropped = img.crop((max(0, x0 - pad_x), max(0, y0 - pad_y),
                    min(W, x1 + pad_x), min(H, y1 + pad_y)))
```

**New behavior (crop512_margin20_truepad)**:
```txt
1. detect dark content bbox in original image
2. crop exactly to content bbox
3. create new white image with 20% padding on each side
4. scale padded image so long edge = 512px
5. center on 512x512 white canvas
```

This guarantees margin even when content touches the original image boundary.

### Preprocessing Settings

```txt
source_variant:     crop512_margin20_truepad   (was: crop512_margin10_standardized)
standardized_margin: 0.20                       (was: 0.10, and was clamped)
```

### metrics.json — New Preprocessing Fields

```json
{
  "source_variant": "crop512_margin20_truepad",
  "standardized_margin": 0.20,
  "content_bbox_original": [x0, y0, x1, y1],
  "content_bbox_after_preprocess": [fx0, fy0, fx1, fy1],
  "final_canvas_margins_px": {
    "left": 0, "top": 0, "right": 0, "bottom": 0
  },
  "content_touches_edge": false
}
```

If `content_touches_edge` is true after true padding, it is a bug and reported in `summary.md`.

### Issue 2: Post-Merge Filtering — Less Destructive

**Task 29 behavior**: full hard filter applied after merge (angle + tiny + one-edge + dangling).
**Task 30 behavior**: light post-merge filter only.

Light post-merge filter keeps:
```txt
angle filter:  discard edges outside ±10 degrees of H/V
deduplication: remove exact duplicate edges
self-loop:     remove zero-length edges
```

Light post-merge filter does NOT apply:
```txt
tiny component deletion
one-edge component deletion
short dangling edge deletion
```

Rationale: merge splits edges at intersections creating short fragments that are still valid
wall segments. Applying full hard filters after merge removes these valid fragments.

Full hard filters (all four) continue to be applied per MC attempt before candidate reranking.

### Three Visual Overlays

```txt
graph_overlay_components.png   — per-component colored, before merge
graph_overlay_merged.png       — after merge-on-intersection, before light post-merge filter
graph_overlay.png              — final after light post-merge filter
```

### Node Snap Tolerance Reduction

```txt
node_snap_tolerance_px = 6   (reduced from 10, Task 30 — less destructive geometry shift)
```

### Runtime Settings Restored

```txt
monte_times             = 4    (restored to Task 29 spec; was accidentally 3 in script)
max_candidates_per_step = 40   (restored to Task 29 spec; was accidentally 15 in script)
```

### Stage-by-Stage Counts in metrics.json

```json
{
  "stage_counts": {
    "components_nodes": 0,
    "components_edges": 0,
    "merged_nodes": 0,
    "merged_edges": 0,
    "final_nodes": 0,
    "final_edges": 0,
    "nodes_removed_by_post_merge_filter": 0,
    "edges_removed_by_post_merge_filter": 0
  },
  "hard_filters_per_attempt": { ... },
  "light_post_merge_filter": {
    "edges_removed_angle_filter": 0,
    "duplicate_edges_removed": 0,
    "self_loop_edges_removed": 0
  }
}
```

### Output Folder

Same location (in-place replacement):
```txt
outputs/vectorization/phase4_raster2graph_generous_inference/<sample_id>/
```

New file per sample:
```txt
graph_overlay_merged.png   (added in Task 30)
```

## 17. Update Rules

Update this file when any of the following changes:

```txt
input source changes
preprocessing variant changes
thresholds or inference settings change
candidate filtering changes
edge-search tolerance changes
restart / recovery / multi-start logic changes
output folder structure changes
graph schema changes
fine-tuning begins
checkpoint or architecture changes
```

When updating, add:

```txt
what changed
why it changed
which task/spec/output folder proves the change
whether it is recommended, experimental, or retired
```
