# Task 30 - Raster-to-Graph True 20 Percent Padding And Less Destructive Final Filtering

## Objective

Debug and improve the current Phase 4 Raster-to-Graph preprocessing and final graph filtering.

Two issues were observed in Task 29 outputs:

```txt
1. Some preprocessed input.png files still have walls/content touching the 512px canvas edge.
2. graph_overlay_components.png often looks closer to the desired graph than graph_overlay.png.
```

This suggests:

```txt
preprocessing margin is not truly being added when source content already touches the original image boundary
final merge/filter cleanup is too destructive and removes good generated graph edges/nodes
```

Do not fine-tune.

Do not train.

Do not modify the checkpoint.

## Output Replacement

Replace existing outputs in:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/
```

Keep the same single-output-per-sample structure:

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

Use the same three-sample smoke-test scope as Task 29 unless explicitly requested otherwise.

## Part A - Fix Preprocessing Padding

Current problem:

```txt
the code expands the content bbox by margin
then clamps the expanded bbox to original image bounds
if content touches the original image edge, the added margin on that side becomes 0
```

Do not do this:

```txt
crop(max(0, x0 - pad), min(W, x1 + pad))
```

because it cannot create margin outside the original image.

Required new preprocessing:

```txt
1. detect actual dark floorplan content bbox
2. crop exactly to the content bbox
3. create a new larger white image
4. paste the content crop into the larger white image with padding on all sides
5. resize this padded image so the long edge fits 512 px
6. center on a 512x512 white canvas
```

Increase padding from 10 percent to 20 percent:

```txt
standardized_margin = 0.20
```

New source variant name:

```txt
crop512_margin20_truepad
```

Required guarantee:

```txt
no dark floorplan content should touch the final 512px image edge
```

Record in `metrics.json`:

```json
{
  "source_variant": "crop512_margin20_truepad",
  "standardized_margin": 0.20,
  "content_bbox_original": [0, 0, 0, 0],
  "content_bbox_after_preprocess": [0, 0, 0, 0],
  "final_canvas_margins_px": {
    "left": 0,
    "top": 0,
    "right": 0,
    "bottom": 0
  },
  "content_touches_edge": false
}
```

If `content_touches_edge` is still true after true padding, treat it as a bug and report it in `summary.md`.

## Part B - Preserve Component Output Better

Observation:

```txt
graph_overlay_components.png often looks better than graph_overlay.png
```

Meaning:

```txt
the generated component graph is often useful
but merge + second hard-filter cleanup removes too much
```

Therefore, make final filtering less destructive.

## Overlay Meaning

Keep three visual outputs if possible:

```txt
graph_overlay_components.png
  raw accepted component outputs before final merge

graph_overlay_merged.png
  graph after merge-on-intersection, before second final hard-filter pass

graph_overlay.png
  final exported graph after conservative cleanup
```

If only two overlays are kept, then:

```txt
graph_overlay_components.png = pre-final-filter diagnostic
graph_overlay.png = final output
```

But `metrics.json` must report how many nodes/edges were removed between each stage.

## Less Destructive Final Filtering

Keep these hard filters:

```txt
angle filter: discard edges outside +/-10 degrees of horizontal/vertical
```

Soften or disable these after the merge stage:

```txt
tiny component deletion
one-edge component deletion
short dangling edge deletion
```

Recommended behavior:

```txt
apply tiny / one-edge / short-dangling filters before candidate reranking
do not re-apply them aggressively after merge
after merge, only remove severe angle violations and exact duplicate/invalid edges
```

Rationale:

```txt
merge can split edges and create temporary short fragments
second hard-filter pass may delete valid wall fragments created by intersection insertion
one-edge or dangling edges may be valid partial wall outputs at this stage
```

## Node Snap / Merge Caution

The merge stage may shift geometry:

```txt
node snap averages nearby nodes
intersection insertion splits edges
collinear merge dissolves overlapping edges
```

Reduce destructive geometry changes:

```txt
node_snap_tolerance_px = 6
edge_intersection_tolerance_px = 8
collinear_overlap_tolerance_px = 8
```

If node snapping shifts nodes visibly away from walls, prefer snapping to the higher-confidence component node rather than averaging.

## Keep Current Runtime-Limited Settings

Use the Task 29 runtime-limited settings unless there is a clear reason to change them:

```txt
first_step_threshold = 0.02
later_step_threshold = 0.02
first_step_force_best = true
edge_search_threshold = 50
monte_times = 4
max_candidates_per_step = 40
max_new_starts = 2
```

If the script currently uses lower runtime values such as:

```txt
monte_times = 3
max_candidates_per_step = 15
```

either restore the Task 29 requested values or clearly document the runtime override in `summary.md`.

## Keep Validation But Reduce Deletion

Keep soft scoring:

```txt
wall evidence alignment score
rectangle / closed-region reward
dangling-node penalty
unsupported-edge penalty
candidate validity reranking
```

Do not add junction support scoring.

Use soft scores to rank candidates, not to delete good graph pieces after final merge.

## Summary Requirements

Update:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/summary.md
```

Include:

```txt
which three samples were tested
whether 20 percent true padding prevented edge-touching content
final canvas margins for each sample
node/edge counts in graph_overlay_components
node/edge counts after merge
node/edge counts in final graph_overlay
how many nodes/edges were removed by final filtering
whether graph_overlay now preserves the useful component graph better
whether final filtering is still too destructive
```

## Spec Update

Update:

```txt
specs/spec_v010_phase4_raster2graph_modifications.md
```

Add:

```txt
Task 30 true padding fix
20 percent standardized margin
crop512_margin20_truepad
clamped bbox expansion was retired
final post-merge filtering made less destructive
graph_overlay_merged.png if implemented
```

## Acceptance Criteria

1. Preprocessing uses true white padding around the content crop, not clamped bbox expansion.
2. Padding is increased to 20 percent.
3. Final `input.png` files do not have dark content touching the 512px canvas edge.
4. `metrics.json` records final canvas margins.
5. Final filtering after merge is less destructive than Task 29.
6. `graph_overlay.png` preserves more useful edges/nodes from `graph_overlay_components.png`.
7. Stage-by-stage node/edge counts are recorded.
8. Junction support score is not added.
9. `summary.md` explains whether preprocessing or filtering was the main improvement.
10. `spec_v010_phase4_raster2graph_modifications.md` is updated.
11. No training, fine-tuning, or checkpoint modification happens.

