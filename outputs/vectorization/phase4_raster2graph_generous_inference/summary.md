# Task 30 — True 20% Padding and Less Destructive Final Filtering

## Samples Tested

Three samples from `data/raster2graph/preprocess_test_samples.json` (same as Task 29):
- Sample `12539` (category=normal, bin=51-80)
- Sample `1316` (category=normal, bin=10-50)
- Sample `13736` (category=normal, bin=10-50)
- Sample `1` (category=normal, bin=51-80)
- Sample `10018` (category=normal, bin=10-50)
- Sample `10025` (category=normal, bin=10-50)
- Sample `10026` (category=normal, bin=10-50)
- Sample `10029` (category=normal, bin=51-80)

## Preprocessing: crop512_margin20_truepad

- **Source variant**: `crop512_margin20_truepad`
- **Standardized margin**: 20% of content bbox on each side
- **Method**: crop content bbox exactly → paste into new white image with 20% padding → scale long edge to 512px → center on 512×512 white canvas
- **Previous method (retired)**: clamped bbox expansion — `crop(max(0, x0-pad), min(W, x1+pad))` failed when content touched the original image boundary
- Samples where content still touches 512px canvas edge after true padding: **0/8**
- **All samples**: no wall content touching canvas edge after 20% true-pad preprocessing. Fix confirmed.

### Final Canvas Margins Per Sample

| Sample | Left px | Top px | Right px | Bottom px | Edge-Touch |
|--------|---------|--------|----------|-----------|------------|
| 12539 | 73 | 102 | 73 | 112 | ok |
| 1316 | 148 | 73 | 150 | 73 | ok |
| 13736 | 73 | 116 | 73 | 117 | ok |
| 1 | 73 | 142 | 73 | 137 | ok |
| 10018 | 78 | 73 | 79 | 73 | ok |
| 10025 | 80 | 73 | 80 | 73 | ok |
| 10026 | 73 | 99 | 73 | 100 | ok |
| 10029 | 73 | 128 | 73 | 128 | ok |

## Generation Settings

| Parameter | Value | Change from Task 29 |
|-----------|-------|---------------------|
| source_variant | crop512_margin20_truepad | changed (was margin10) |
| first_step_threshold | 0.02 | unchanged |
| later_step_threshold | 0.02 | unchanged |
| first_step_force_best | True | unchanged |
| edge_search_threshold | 50 px | unchanged |
| monte_times | 4 | restored to Task 29 spec (was 3 in script) |
| max_candidates_per_step | 40 | restored to Task 29 spec (was 15 in script) |
| max_new_starts | 2 | unchanged |
| node_snap_tolerance_px | 6 | reduced from 10 (less geometry shift) |

## Stage-by-Stage Node/Edge Counts

| Sample | Comp Nodes | Comp Edges | Merged Nodes | Merged Edges | Final Nodes | Final Edges | Removed by Filter |
|--------|-----------|-----------|-------------|-------------|------------|------------|-------------------|
| 12539 | 47 | 48 | 39 | 23 | 33 | 23 | 6n/0e |
| 1316 | 9 | 11 | 9 | 7 | 9 | 7 | 0n/0e |
| 13736 | 22 | 28 | 19 | 11 | 15 | 11 | 4n/0e |
| 1 | 36 | 42 | 34 | 18 | 30 | 18 | 4n/0e |
| 10018 | 21 | 24 | 18 | 9 | 13 | 9 | 5n/0e |
| 10025 | 27 | 35 | 23 | 14 | 19 | 14 | 4n/0e |
| 10026 | 19 | 20 | 20 | 13 | 18 | 13 | 2n/0e |
| 10029 | 32 | 38 | 30 | 17 | 27 | 17 | 3n/0e |

## Post-Merge Filtering: Less Destructive

**Task 29 behavior**: full hard filter applied after merge (angle + tiny + one-edge + dangling).
**Task 30 behavior**: light post-merge filter only (angle violations + exact duplicate edges).

Rationale: merge splits edges at intersections creating short fragments that are still valid
wall segments. Tiny/one-edge/dangling deletion after merge removes these valid fragments.

- Light post-merge angle violations removed (total): 0
- Light post-merge duplicate edges removed (total): 0

### Whether graph_overlay preserves component graph better

Compare `graph_overlay_components.png` vs `graph_overlay.png` per sample.
Stage counts above show how many nodes/edges survived from component → merge → final.
- Sample 12539: 48 component edges → 23 final edges (48% retained)
- Sample 1316: 11 component edges → 7 final edges (64% retained)
- Sample 13736: 28 component edges → 11 final edges (39% retained)
- Sample 1: 42 component edges → 18 final edges (43% retained)
- Sample 10018: 24 component edges → 9 final edges (38% retained)
- Sample 10025: 35 component edges → 14 final edges (40% retained)
- Sample 10026: 20 component edges → 13 final edges (65% retained)
- Sample 10029: 38 component edges → 17 final edges (45% retained)

## Hard Filter Removals Per MC Attempt (before candidate reranking)

- Edges removed by angle filter: 73
- Tiny components removed: 4
- One-edge components removed: 0
- Short dangling edges removed: 22

## Final Results

- Total samples processed: 8
- Non-empty graph rate: 8/8 = 100.0%
- Average final nodes: 20.5
- Average final edges: 14.0
- Average wall evidence score: 0.316
- Average rectangle cycle score: 0.050
- Average dangling penalty: 0.389
- Average unsupported edge ratio: 0.669

## Per-Sample Results

| Sample | Category | Bin | Nodes | Edges | Wall Ev | Cycles | Edge-Touch | Empty |
|--------|----------|-----|-------|-------|---------|--------|------------|-------|
| 12539 | normal | 51-80 | 33 | 23 | 0.23 | 0 | ok | no |
| 1316 | normal | 10-50 | 9 | 7 | 0.32 | 1 | ok | no |
| 13736 | normal | 10-50 | 15 | 11 | 0.32 | 0 | ok | no |
| 1 | normal | 51-80 | 30 | 18 | 0.31 | 0 | ok | no |
| 10018 | normal | 10-50 | 13 | 9 | 0.40 | 0 | ok | no |
| 10025 | normal | 10-50 | 19 | 14 | 0.42 | 0 | ok | no |
| 10026 | normal | 10-50 | 18 | 13 | 0.26 | 1 | ok | no |
| 10029 | normal | 51-80 | 27 | 17 | 0.27 | 0 | ok | no |

## Analysis: What Was The Main Improvement?

**Preprocessing fix** (true 20% padding):
  Task 29: 3/3 samples had content touching canvas edge.
  Task 30: 0/3 samples have content touching canvas edge.

**Filtering fix** (less destructive post-merge):
  Compare stage counts: if final nodes/edges are closer to component counts than in Task 29,
  the light filter is preserving more useful geometry.

**Whether filtering is still too destructive**:
  Check `graph_overlay_components.png` vs `graph_overlay.png` visually.
  If graph_overlay.png still looks much sparser, the merge stage itself may be
  collapsing geometry via snapping (node_snap_tolerance_px reduced to 6 in Task 30).

## Three Visual Overlays

Each sample now has three overlays:
- `graph_overlay_components.png` — per-component colored, before merge
- `graph_overlay_merged.png` — after merge, before light post-merge filter
- `graph_overlay.png` — final after light post-merge filter

