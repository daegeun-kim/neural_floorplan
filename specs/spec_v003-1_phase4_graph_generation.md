# Spec v003-1: SVG To Orthogonal Wall Graph Generation

## 0. Purpose

This spec defines the SVG-derived wall graph generation stage originally planned for Phase 4 vectorization.

The goal is to generate a very simple orthogonal wall graph from each original CubiCasa `model.svg`.

This graph is not the final vectorization output. In the current settled Phase 4 method, it is optional QA/reference data and a possible future fine-tuning fallback label. The active Phase 4 wall graph now comes from pretrained Raster-to-Graph inference on `model_clean.png`.

The graph should primarily represent the structural wall layout:

```txt
nodes = wall endpoints / wall junctions
edges = orthogonal wall segments between nodes
```

The graph may also include opening-center annotation nodes:

```txt
door_center nodes = centers of door openings
window_center nodes = centers of window openings
```

Opening-center nodes must not participate in wall edge connectivity. They are spatial annotations hosted on wall edges.

Do not attempt to encode room categories, furniture, fixtures, symbols, or wall thickness in the first version.

## 1. Input Dataset

Use the original vector SVG files under:

```txt
C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\docs\high_quality_architectural
```

Each sample folder contains:

```txt
model.svg
masks/
```

Example:

```txt
docs/high_quality_architectural/1/model.svg
```

## 2. Output Files

For each sample, write graph labels into the sample's `masks` folder.

Required first-version outputs:

```txt
masks/wall_graph.json
masks/wall_graph_debug.svg
masks/wall_graph_debug.png
```

`wall_graph.json` is the optional reference/future-training label.

`wall_graph_debug.svg` and `wall_graph_debug.png` are visual QA artifacts.

## 3. Graph Format

The graph must be extremely simple.

Recommended JSON:

```json
{
  "sample_id": "1",
  "source_svg": "model.svg",
  "coordinate_space": "raster",
  "image_width": 768,
  "image_height": 768,
  "nodes": [
    {"id": 0, "type": "wall_node", "x": 100.0, "y": 80.0},
    {"id": 1, "type": "wall_node", "x": 240.0, "y": 80.0},
    {"id": 2, "type": "door_center", "x": 170.0, "y": 80.0, "host_edge": 0},
    {"id": 3, "type": "window_center", "x": 210.0, "y": 80.0, "host_edge": 0}
  ],
  "edges": [
    {"id": 0, "start": 0, "end": 1}
  ]
}
```

Wall node type may be kept simple in v1.

If more detail is included, derive wall-node subtype mechanically from degree and incident directions:

```txt
end
corner
t_junction
cross
```

The model target may distinguish:

```txt
wall_node
door_center
window_center
```

`door_center` and `window_center` nodes are not edge endpoints. They must include enough metadata to recover their hosted wall relationship, preferably:

```txt
host_edge
opening_width_px or opening_width_mm when reliable
source_bbox when available
orientation
```

The wall graph topology must not depend on opening-center nodes.

## 4. Coordinate System

Graph coordinates must match the raster/mask coordinate system used by the dataset.

Required convention:

```txt
x increases right
y increases down
origin is top-left
coordinates align with generated mask PNGs
```

If `model.svg` uses a different viewBox or transform, convert graph coordinates into the same pixel coordinate system as the generated masks.

## 5. Graph Generation Strategy

The original SVG walls are often not clean CAD centerlines. They may be complex layered vector drawings.

Therefore, do not assume walls can be extracted by simply reading SVG `<line>` elements.

Use the SVG as a high-quality vector source to generate wall evidence:

```txt
model.svg
-> isolate/render wall-relevant SVG evidence
-> wall-only raster or high-resolution wall mask
-> clean mask
-> extract centerline/skeleton
-> simplify and orthogonalize
-> build graph
```

The first implementation may use the existing mask generation logic as a source of wall-only evidence if it is more reliable than direct SVG parsing.

## 6. Wall Evidence Extraction

Prefer this order:

1. Use known CubiCasa SVG group/layer names for walls where available.
2. Use the existing semantic mask generation process to create a wall mask.
3. Use SVG style/color/stroke-width heuristics only when group names are insufficient.

Ignore:

```txt
furniture
fixtures
icons
text
dimension markers
doors
windows
floor fills
background
```

Door/window evidence may create gaps in walls. For graph-label generation, bridge short wall gaps across hosted openings where the wall line is structurally continuous.

Also extract hosted opening centers when reliable:

```txt
door openings -> door_center annotation nodes
window openings -> window_center annotation nodes
```

Each opening-center node should be placed at the center of the opening span and projected onto its host wall edge. The host wall edge must remain a single continuous wall edge unless an actual wall junction requires a split.

## 7. Centerline Extraction

From the wall evidence mask:

1. Clean small artifacts.
2. Close small gaps.
3. Skeletonize wall regions.
4. Extract skeleton endpoints and junctions.
5. Split skeleton into chains.
6. Convert chains into candidate line segments.

Use high-resolution rendering if needed, then downscale coordinates back to the dataset coordinate system.

## 8. Orthogonalization

The output graph must be orthogonal.

Allowed edge directions:

```txt
horizontal
vertical
```

Rules:

```txt
near-horizontal chains snap to horizontal
near-vertical chains snap to vertical
diagonal/noisy chains are rejected or split only if a clear orthogonal interpretation exists
```

Do not output diagonal graph edges in v1.

## 9. Node And Edge Construction

After orthogonalization:

1. Merge near-duplicate nodes.
2. Intersect horizontal and vertical lines to form junctions.
3. Split long lines at junctions.
4. Remove very short noisy stubs.
5. Keep connected wall graph components that represent the building structure.

Each edge must connect two existing node IDs.

Opening-center nodes are allowed in `nodes`, but they must not be referenced by `edges.start` or `edges.end`.

Opening-center nodes must be typed separately:

```txt
door_center
window_center
```

Each opening-center node should reference its host wall edge by ID. The host wall edge should pass through the opening center within configured tolerance.

Each edge must be either:

```txt
x1 == x2
```

or:

```txt
y1 == y2
```

within configured tolerance.

## 10. Quality Filtering

Not every source SVG needs to produce a usable training label.

Reject or mark samples as unusable when:

```txt
wall graph is disconnected in implausible ways
too many tiny components remain
too many diagonal chains are present
node count is extremely small or extremely large
edge count is inconsistent with node count
graph is visually far from the wall mask
```

Write the reason into:

```txt
masks/wall_graph_metrics.json
```

Do not silently include bad graph labels in training.

## 11. Debug Visualization

`wall_graph_debug.svg` and `wall_graph_debug.png` must show:

```txt
source wall evidence or wall mask
graph nodes
graph edges
door_center annotation nodes
window_center annotation nodes
rejected/suppressed tiny stubs if practical
```

Recommended colors:

```txt
wall mask: light gray
graph edges: blue
graph nodes: red
door_center nodes: orange
window_center nodes: green
```

The debug visualization should make it easy to visually decide whether the graph label is suitable for training.

## 12. Relationship To Existing Specs

This spec is between:

```txt
spec_v003_semantic_mask_generation.md
spec_v005_phase4_raster2graph.md
```

It uses the original SVG source to create graph labels.

`spec_v005_phase4_raster2graph.md` uses those graph labels to train a model.

This stage does not replace semantic mask generation. It adds graph labels alongside masks.

## 13. Completion Criteria

This spec is satisfied when:

1. Each selected sample can produce `masks/wall_graph.json`.
2. The graph is in raster coordinate space.
3. All graph edges are orthogonal.
4. Debug visualizations are generated.
5. Bad or uncertain samples are flagged instead of being silently used.
6. Door centers and window centers are represented as distinct annotation node types when reliable.
7. Opening-center annotation nodes do not split wall edges and are not used as edge endpoints.
8. The output graph is simple enough to train a raster-to-graph model.

## 14. Implementation Notes

Implemented in `src/generate_wall_graphs.py` (CLI: `python -m src.generate_wall_graphs <root_dir>
[--overwrite] [--verbose] [--limit N]`), tested in `tests/test_generate_wall_graphs.py`.

### Reuse of the v008 mask-to-vector pipeline

This stage works on clean SVG-derived ground truth, not noisy CNN predictions, but most of
the skeletonize -> walk -> orthogonalize machinery already existed in
`src/vectorization/{components,point_detection}.py` (built for spec_v008) and is reused
directly:

- `components.extract_components` for cleaning + connected components + skeletonization.
- `point_detection.build_wall_skeleton_graph` / `_classify_wall_nodes` / `_finalize_free_ends` /
  `_link_skeleton_edges_to_points` for the skeleton walk, corner-splitting, and
  junction/free-end classification.

`point_connection.build_wall_edges` was **not** reused â€” it deliberately splits a wall chain at
hosted window/door points, which this spec explicitly forbids (SS9: opening-center nodes must
never be edge endpoints). A dedicated, simpler edge builder connects each skeleton chain's two
endpoints directly instead.

### Bridging openings before skeletonizing

`masks/wall_mask.png` (already rendered by `generate_semantic_masks`) has a real pixel gap at
every window/door, because the original SVG's `Window`/`Door > Threshold` polygons use a
white/light fill. Rather than guessing a gap-closing distance, this stage rasterizes those exact
opening polygons back onto the wall mask before skeletonizing (`bridge_wall_mask`), so the
skeleton passes straight through every opening as one continuous chain.

### Door swing-arc contamination (judgment call)

The original SVG's `Door > Panel > path` (the swing-arc/leaf visual) has no fill/stroke of its
own â€” it inherits `stroke="#000000"` from the `Wall`/`Door` ancestor groups â€” so
`generate_semantic_masks`'s "wall" category render (which keeps the whole `Wall` subtree as-is)
picks up that curved stroke as wall evidence. Left in, the skeleton walk orthogonalizes the
curve into spurious stair-step loops next to every door. Fixed by `strip_door_swing_evidence`,
which subtracts a dilated `door_arc_mask.png`/`door_leaf_mask.png` (already-rendered synthetic
copies of that same geometry, spec_v005 run3) from the wall mask before bridging/skeletonizing.

### Near-duplicate node merging and exact orthogonality (judgment call)

Skeletonize commonly leaves a handful of separate junction/free-end pixels within a couple of
pixels of each other at an ordinary corner (most often where two wall bands of equal thickness
meet, e.g. a partition meeting an exterior wall) â€” left alone these become spurious
near-duplicate nodes joined only by short, non-cardinal "spur" chains that get rejected as
diagonal, fragmenting the graph. `merge_near_duplicate_points` collapses these via a simple
distance-based union-find before edges are built.

`src.vectorization.point_alignment._assert_wall_edge_axes` (reused by spec_v008) only snaps each
*pair* of skeleton-edge-connected points to their average axis value, which does not propagate
consistently across a junction with 3+ incident edges. Since this spec requires *exact*
`x1==x2`/`y1==y2` (SS9), `snap_shared_axes` replaces it with a transitive union-find: every point
connected through horizontal edges is forced to one shared y, and through vertical edges to one
shared x.

### Quality-gate calibration (judgment call)

`wall_graph_metrics.json`'s `too_many_diagonal_chains` check budgets the harmless per-corner
"spur" noise above (`diagonal_chains_per_node_budget` x node_count + `diagonal_chains_absolute_floor`)
rather than a flat ratio of rejected-to-accepted chains â€” that noise scales with junction count,
not with how clean the wall evidence actually is, so a flat ratio over-flagged small/simple floor
plans in testing.

### Environment

Same Cairo DLL path note as spec_v003 applies (this stage calls
`generate_semantic_masks.generate_masks` when a sample's `wall_mask.png`/`window_mask.png` are
missing):

```powershell
$env:PATH = "C:\Users\kdgki\anaconda3\envs\floorplan-cad\Library\bin;" + $env:PATH
```
