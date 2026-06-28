# Spec v008: Phase 4 Graph-To-Vector Wall, Door, And Window Reconstruction

## 0. Purpose

This spec defines the next Phase 4 vectorization stage after Raster-to-Graph wall graph inference.

The active Phase 4 direction is:

```txt
input raster
-> shared preprocessing
-> Raster-to-Graph wall centerline graph
-> 7-class segmentation evidence on the same preprocessed image
-> graph orthogonalization
-> scale inference
-> door/window hosting on the wall graph
-> wall interval trimming
-> final wall/window/door vector output
```

The key design decision is that Raster-to-Graph owns wall topology. The 7-class segmentation model supplies semantic evidence for scale, doors, windows, and debug confidence.

This stage must not return to the Phase 3 behavior where wall topology is inferred primarily from black wall pixels in the segmentation output. Black wall pixels may be used as evidence or confidence, but the wall graph source of truth is the Raster-to-Graph output.

## 1. Required Inputs

The implementation needs both model outputs from one identical preprocessed canvas.

Required inputs:

```txt
1. original raster image or model_clean.png
2. Raster-to-Graph graph_pred.json
3. Raster-to-Graph graph_pred.svg
4. 7-class segmentation prediction on the same preprocessed input
```

The 7-class segmentation classes are:

```txt
0 background
1 floor
2 wall
3 window
4 door_arc
5 door_leaf
6 door_origin
```

## 2. Shared Preprocessing Contract

Raster-to-Graph and 7-class segmentation must run on the identical preprocessed image.

The preprocessing stage is:

```txt
original image
-> detect content bbox
-> crop exactly to content bbox
-> add true white padding around the crop
-> scale long edge to 512 px
-> center on 512x512 white canvas
```

The current Phase 4 Raster-to-Graph preprocessing is:

```txt
crop512_margin20_truepad
```

The implementation must save the preprocessing transform metadata:

```json
{
  "source_image": "...",
  "source_width": 0,
  "source_height": 0,
  "content_bbox_original": [0, 0, 0, 0],
  "padding_fraction": 0.20,
  "padded_width": 0,
  "padded_height": 0,
  "scale_to_512": 1.0,
  "canvas_offset_x": 0,
  "canvas_offset_y": 0,
  "coordinate_space": "preprocessed_512",
  "source_variant": "crop512_margin20_truepad"
}
```

All graph nodes, segmentation masks, opening points, and debug overlays are first computed in `preprocessed_512` coordinates. Conversion back to original image coordinates or millimeters must use this manifest.

## 3. Required Outputs

For each input image, the notebook/pipeline must produce:

```txt
input.png
image_segmentation.png
image_debug_overlay.png
graph_pred.svg
graph_pred.json
graph_overlay.png
graph_overlay_aligned.png
final_vector.svg
final_vector.json
```

File meanings:

```txt
input.png                 preprocessed 512x512 image
image_segmentation.png    7-class segmentation preview on input.png
image_debug_overlay.png   final door/window/scale/hosting debug overlay
graph_pred.svg            raw Raster-to-Graph wall graph
graph_pred.json           raw Raster-to-Graph wall graph JSON
graph_overlay.png         raw graph overlay on input.png
graph_overlay_aligned.png orthogonally aligned graph overlay
final_vector.svg          final CAD-like wall/window/door SVG
final_vector.json         final typed geometry and metadata
```

Do not use the same filename twice for different artifacts. The aligned graph overlay must be named `graph_overlay_aligned.png`.

## 4. Wall Graph Normalization

Input graph schema:

```json
{
  "nodes": [[x, y], ...],
  "edges": [[x1, y1, x2, y2], ...]
}
```

The graph normalization stage must:

```txt
1. load Raster-to-Graph nodes and edges
2. remove zero-length edges
3. remove exact duplicate edges
4. cluster near-equal x axes
5. cluster near-equal y axes
6. snap all edges to horizontal or vertical
7. split horizontal/vertical intersections into graph nodes
8. merge collinear overlapping edges
9. preserve connected component IDs for debug
```

All final wall graph edges must be orthogonal.

Allowed final edge directions:

```txt
0 degrees
90 degrees
180 degrees
270 degrees
```

Edges outside `+/-10 degrees` of horizontal or vertical must be rejected before final vector output.

## 5. Orthogonal Alignment

The graph should be aligned by solving shared axes, not by independently rotating each edge.

Recommended behavior:

```txt
near-horizontal edge -> y coordinate is assigned from its endpoint/axis cluster
near-vertical edge   -> x coordinate is assigned from its endpoint/axis cluster
junction nodes       -> preserve connectivity during axis snapping
short edges          -> may collapse only if they become zero-length after snapping
```

Axis clustering should run before opening insertion.

Opening points inserted later must snap onto the already-aligned wall graph.

## 6. Scale Inference

Scale inference is identical in priority to the Phase 3 attempt, but its output is consumed by the graph-to-vector stage.

Scale priority:

```txt
1. explicit metadata, if available
2. red door_arc connected-component bbox long edge
3. door_origin width as secondary cross-check only
4. wall thickness as weak secondary cross-check only
5. unknown scale
```

Allowed door modules:

```txt
700 mm
900 mm
```

For each accepted red door_arc component:

```txt
long_edge_px = max(bbox_width_px, bbox_height_px)
candidate_px_to_mm = 700 / long_edge_px
candidate_px_to_mm = 900 / long_edge_px
```

Cluster all candidates and choose the best-supported group. If both 700mm and 900mm interpretations remain plausible, record the ambiguity in metrics and choose the cluster with stronger multi-door support.

Wall thickness must not override a resolved red door-arc scale.

If scale is unknown, the implementation may still output pixel-space SVG and JSON, but all metric decisions must be marked `scale_blocked`.

## 7. Door Generation

Door existence and count are driven by red `door_arc` components from the 7-class segmentation.

Each accepted red door_arc component produces exactly one door candidate unless it fails a basic component cleanup rule such as minimum area or implausible bbox aspect ratio.

The two wall-door points should be inferred from graph proximity, not from black segmentation pixels alone.

Door point inference:

```txt
1. compute the red door_arc component bbox
2. take the 4 bbox vertices as raw candidates
3. identify the likely wall-facing bbox edge using proximity to the aligned R2G wall graph
4. use purple/black/orange evidence only as confidence and tie-break evidence
5. choose the 2 adjacent vertices on the wall-facing edge as raw hinge/end points
6. determine hinge vs end using nearby door_leaf evidence where available
7. snap both points to the same compatible wall graph edge
8. insert both snapped points as new nodes on that edge
9. trim the wall interval between them
10. generate door origin, leaf, and arc primitives procedurally
```

Important rule:

```txt
The two door points must snap to the same wall edge or the same wall chain interval.
They must not independently snap to two unrelated edges.
```

This is stricter than Phase 3. The old Phase 3 behavior allowed each point to seek its nearest wall separately. That can create doors spanning disconnected graph fragments and must not be used here.

If the two raw door points cannot be hosted on the same compatible wall edge or wall chain, the door candidate must be rejected or marked unresolved in debug output. Do not silently create a free-floating door.

Door graph-hosting score should prefer:

```txt
1. nearest aligned graph edge with compatible orientation
2. edge whose projection interval contains both door points or can contain both after module snapping
3. edge close to red bbox wall-facing side
4. edge supported by nearby purple door_origin or black wall pixels
5. edge with sufficient length for 700mm/900mm door module
```

After hosting, snap the door width to the nearest allowed module when scale is resolved:

```txt
700 mm
900 mm
```

The snapped endpoint positions must remain on the same host edge.

## 8. Window Generation

Window existence is driven by blue `window` components from the 7-class segmentation.

Window point inference:

```txt
1. extract blue window connected components
2. estimate each component's major axis
3. infer two window endpoints from the component extent
4. host the two endpoints onto the same compatible wall graph edge
5. insert the two endpoints as new wall graph nodes
6. trim the wall interval between them
7. generate a window primitive in the resulting gap
```

The old assumption that window points are simply the intersection between blue and black pixels is not reliable enough. Blue/black contact may be fragmented, absent, or shifted in the segmentation output.

Blue/black intersection evidence may be used as a confidence feature, but the final window endpoints must be hosted by the graph.

Important rule:

```txt
The two window endpoints must snap to the same wall edge or same wall chain interval.
They must not independently snap to two unrelated edges.
```

If both endpoints cannot be hosted on the same compatible wall edge or wall chain, reject the window candidate into debug output.

Minimum final window width:

```txt
300 mm
```

If scale is unknown, the minimum-width decision is scale-blocked and must be recorded rather than replaced with an arbitrary pixel threshold.

## 9. Opening Interval Editing

Doors and windows edit the wall graph before wall polygon generation.

The correct representation is:

```txt
wall centerline graph
-> opening intervals inserted on wall edges/chains
-> wall intervals trimmed
-> remaining wall centerlines buffered
```

Do not buffer walls before subtracting openings.

For each hosted opening:

```txt
host_edge_id
host_chain_id
opening_type
start_point
end_point
projection_t_start
projection_t_end
width_px
width_mm
confidence
source_component_id
```

Opening intervals on the same wall chain must be sorted and checked for overlap. Overlapping or conflicting door/window intervals must be resolved before wall generation.

Conflict behavior:

```txt
door vs window overlap:
  preserve both valid openings when possible
  keep the door interval fixed
  move/shrink the window interval away from the door until no trim overlap remains

door vs door overlap:
  preserve both valid doors when possible
  keep the higher-confidence red door_arc candidate fixed
  move/shrink the lower-confidence door interval until no trim overlap remains

window vs window overlap:
  preserve both valid windows when possible
  keep the higher-confidence blue window candidate fixed
  move/shrink the lower-confidence window interval until no trim overlap remains
```

Rejecting an opening is a last resort only when there is no physically possible non-overlapping placement on the host wall chain. The default behavior must be de-overlap by interval adjustment, not deletion.

## 10. Wall Polygon Generation

Wall polygon generation happens only after:

```txt
1. graph orthogonalization
2. graph intersection splitting
3. door/window endpoint insertion
4. door/window interval trimming
5. connected wall-chain reconstruction
```

The remaining wall edges must be connected into graph chains before buffering. This is required so corners and junctions render cleanly.

Required behavior:

```txt
wall graph fragments sharing endpoints are joined into continuous LineString/MultiLineString geometry
T-junctions and corners are preserved as connected graph topology
the connected line system is buffered once as a wall system
```

Wall thickness:

```txt
total wall thickness = 200 mm
buffer distance = 100 mm on each side of centerline
```

When scale is resolved:

```txt
half_width_px = 100 / px_to_mm
```

When scale is unknown:

```txt
use configured preview wall half-width only for SVG preview
mark wall thickness as scale_blocked in final_vector.json
```

Recommended implementation:

```txt
networkx for topology
shapely LineString/MultiLineString for chain geometry
shapely buffer(cap_style="flat", join_style="mitre") for wall polygons
shapely union_all for final wall system
```

The final wall output must be a closed filled polygon system, not SVG stroke-width lines.

## 11. Final Vector JSON

`final_vector.json` must include:

```json
{
  "coordinate_space": "preprocessed_512",
  "preprocessing": {},
  "scale": {
    "status": "resolved|estimated|unknown",
    "px_to_mm": null,
    "evidence": []
  },
  "wall_graph": {
    "nodes": [],
    "edges": [],
    "aligned_nodes": [],
    "aligned_edges": []
  },
  "openings": {
    "doors": [],
    "windows": [],
    "rejected": []
  },
  "geometry": {
    "walls": [],
    "doors": [],
    "windows": []
  },
  "metrics": {}
}
```

Each accepted door/window must store:

```txt
source_component_id
host_edge_id or host_chain_id
raw_points
snapped_points
width_px
width_mm
confidence
rejection_reason if unresolved
```

## 12. Final SVG

`final_vector.svg` must contain only final architectural output and no debug-only geometry.

Visible final groups:

```txt
walls
windows
door_origins
door_leaves
door_arcs
```

Do not include:

```txt
raw segmentation masks
raw R2G candidate components
rejected evidence boxes
untyped points
debug labels
```

Debug geometry belongs in `image_debug_overlay.png`.

## 13. Debug Overlay

`image_debug_overlay.png` must show enough information to audit failures:

```txt
preprocessed image
aligned graph edges
accepted door/window components
raw opening endpoint candidates
snapped opening endpoints
host edge/chain
rejected opening components with reason
scale evidence components
```

The overlay should make same-edge hosting visible. A door/window whose two endpoints would snap to different edges must be visibly rejected, not hidden.

## 14. Main Risks And Required Safeguards

### 14.1 Coordinate Mismatch

Risk:

```txt
R2G and segmentation run on slightly different canvases.
```

Safeguard:

```txt
one preprocessing function, one manifest, both models use input.png
```

### 14.2 Door Bbox Vertices Not On True Wall Line

Risk:

```txt
red bbox vertices are approximate and may not lie exactly on the wall graph.
```

Safeguard:

```txt
infer the wall-facing bbox edge by proximity to the aligned R2G graph,
then project the two door points onto one compatible host graph edge/chain.
```

This should work better than Phase 3 black-pixel inference because the R2G graph is cleaner and more topological than the segmentation wall mask. Black pixels remain useful as confidence, but they should not decide the final wall line.

### 14.3 Disconnected R2G Wall Graph Regions

Risk:

```txt
the two endpoints of one door/window snap to two separate disconnected edges.
```

Safeguard:

```txt
same-opening endpoints must host on the same edge or same wall chain interval.
If that is impossible, reject or flag the opening instead of creating a cross-fragment opening.
```

This is a must-rule for Phase 4 vectorization.

### 14.4 Buffering Before Trimming

Risk:

```txt
cutting door/window holes from buffered wall polygons is unstable.
```

Safeguard:

```txt
insert opening nodes and trim centerline intervals before any wall buffering.
```

### 14.5 Unclean Corners And Junctions

Risk:

```txt
buffering independent wall segments creates capped rectangles and messy overlaps.
```

Safeguard:

```txt
reconstruct connected graph chains first, then buffer the connected line system.
```

## 15. Notebook Contract

The Phase 4 vectorization notebook is required, but it must be generated only after the Phase 4 vectorization source modules/classes/functions exist.

Do not create a placeholder notebook before the source implementation is available.

After implementation, Claude must generate:

```txt
notebooks/phase4_vectorization.ipynb
```

It replaces `notebooks/phase4_raster2graph.ipynb` as the main Phase 4 demonstration notebook.

The notebook must import and call the implemented Phase 4 vectorization classes/functions from `src/` rather than duplicating the algorithm inline.

The notebook must:

```txt
1. accept a single image path
2. run shared preprocessing
3. run Raster-to-Graph inference
4. run 7-class segmentation on the same preprocessed image
5. orthogonally align the graph
6. infer scale
7. attach doors/windows to the graph
8. trim wall intervals
9. generate final wall/window/door vectors
10. write all required output files
```

The notebook must be generated as a final integration/demo artifact after the code path is implemented and importable.

For every future Phase 4 vectorization task that changes source behavior, Claude must also check and update `notebooks/phase4_vectorization.ipynb` so the notebook continues to call the current source path used for manual sample testing.

Notebook requirements after any source update:

```txt
1. no stale inline copy of pipeline logic
2. imports point to the current src/vectorization/phase4 modules
3. autoreload or clear restart instructions are present
4. Run All executes the same code path as the source pipeline
5. final_vector.svg/json reflect current source behavior, including adjusted openings and door orientation
```

If the notebook cannot be updated in the same task, the task is incomplete.

## 16. Completion Criteria

This spec is complete when the Phase 4 notebook can take an input image and produce:

```txt
input.png
image_segmentation.png
image_debug_overlay.png
graph_pred.svg
graph_pred.json
graph_overlay.png
graph_overlay_aligned.png
final_vector.svg
final_vector.json
```

The final vector output must satisfy:

```txt
walls are generated from connected graph chains before buffering
walls are 200mm total thickness when scale is resolved
doors/windows are hosted on one wall edge or one wall chain interval
opening intervals are trimmed before wall polygon buffering
final SVG contains only final architectural geometry
debug evidence is kept out of final SVG
```

## 17. Implementation Notes (task31 — 2026-06-27)

Module layout implemented as `src/vectorization/phase4/`:

```txt
__init__.py             — public API: run_phase4_pipeline, Phase4Result
preprocessing.py        — wraps preprocess_crop512_margin20_truepad + manifest
graph_alignment.py      — orthogonal normalization pipeline (steps 1-7)
segmentation_inference.py — loads FloorplanSegModel, runs inference on 512x512
scale_inference.py      — thin wrapper over vectorization.scale
opening_detection.py    — DoorCandidate / WindowCandidate from segmentation
opening_hosting.py      — same-edge constraint; HostedOpening / RejectedOpening
wall_interval_editing.py — insert opening nodes, trim wall intervals
wall_buffering.py       — connected chain buffering → WallGeometry
export_json.py          — build_final_vector_json / write_final_vector_json
export_svg.py           — build_final_svg / write_final_svg
debug_overlay.py        — build_debug_overlay / write_debug_overlay
pipeline.py             — Phase4Result dataclass + run_phase4_pipeline()
```

Key implementation decisions:

```txt
graph_alignment.py: collinear merge runs BEFORE intersection splitting so
  original wall-line segments get joined but intersection nodes are preserved.
  _merge_intervals uses touch_tol=0.5px for pre-split merging; splitting does
  not re-merge (touching-only segments are kept separate after the split).

opening_hosting.py: both endpoints must project onto THE SAME wall edge
  (or be rejected). _try_host_on_edge checks that both perpendicular
  distances are within max_perp_dist_px (default 20px) and the snapped
  width is above min_width_px.

wall_interval_editing.py: conflict resolution keeps highest-confidence
  opening on each edge. Trimming inserts nodes and emits wall sub-segments
  outside the opening gap; the opening span is never emitted as wall.

wall_buffering.py: uses shapely linemerge + buffer(cap_style='flat',
  join_style='mitre'). Connected chains buffered as one system.

segmentation_inference.py: loads FloorplanSegModel (backbone + decoder)
  from a training checkpoint saved by src/checkpointing.py; imagenet
  normalisation matches training.
```

Tests: `tests/test_phase4_vectorization.py` — 34 tests, all pure geometry
(no GPU), covering all pipeline stages.

Notebook: `notebooks/phase4_vectorization.ipynb` — imports from
`src/vectorization/phase4/`, accepts a single image path, writes all
required output files, prints summary and shows 8-panel visualisation.

## 18. Door Primitive Fix (task32 — 2026-06-27)

Added `src/vectorization/phase4/door_geometry.py` — `compute_door_geometry()`
derives `hinge_point`, `origin_far_point`, `leaf_end`, `swing_side` from a
`HostedOpening`. Fallback hinge = `snapped_points[0]`; fallback swing =
`"fallback_left"`. Both recorded in `hinge_source` / `swing_source` fields.

`export_svg.py` was rewritten to use Phase 3 primitives directly:
- `DoorOriginPrimitive` → purple line along the hosted wall opening (p0→p1)
- `DoorLeafPrimitive` → orange perpendicular line from hinge_point
- `DoorArcPrimitive` → red 90° arc from origin_far_point to leaf_end, center=hinge

The old incorrect behavior (treating the origin edge as the leaf, starting the
arc at the hinge, drawing only a circle for origin) is fully replaced.

`export_json.py` now adds `door_geometry` sub-dict to each `geometry.doors`
entry with: `hinge_point`, `origin_far_point`, `leaf_end`, `swing_side`,
`width_px`, `width_mm`, `orientation_angle_deg`, `hinge_source`, `swing_source`,
`primitive_contract: "door_origin_leaf_arc"`.

12 new tests added to `TestDoorGeometry` in `test_phase4_vectorization.py`
(total: 46 phase4 tests, 314 overall).

## 19. Opening Interval De-Overlap (task33 — 2026-06-28)

`wall_interval_editing.py` rewritten to adjust overlapping intervals instead of rejecting openings:

Conflict resolution priority (§9 update):

```txt
door > window  (type priority; door always fixed over window)
higher confidence > lower confidence (within same type)
```

New `AdjustedOpening` dataclass carries:

```txt
original_t_start / original_t_end   — interval before adjustment
adjusted_t_start / adjusted_t_end   — interval after de-overlap
was_adjusted / adjustment_reason    — audit trail
adjustment_px / adjustment_mm       — shift distance
overlap_resolution_priority         — "door_fixed"|"higher_confidence_fixed"|"not_needed"
large_adjustment_flagged            — set when shift exceeds max_opening_adjustment_mm (200mm)
```

`TrimmedGraph` gains `last_resort_rejected: list[dict]` for openings with no feasible
non-overlapping placement. These carry `rejection_reason = "no_feasible_non_overlapping_interval"`.

Configurable constants in `wall_interval_editing.py`:

```txt
_MIN_OPENING_SEPARATOR_MM = 50.0
_MAX_OPENING_ADJUSTMENT_MM = 200.0
_MIN_SEPARATOR_FALLBACK_PX = 3.0    (scale unknown)
_MAX_ADJUSTMENT_FALLBACK_PX = 30.0  (scale unknown)
```

`trim_wall_intervals()` now accepts `px_to_mm` (from scale_info) so separator thresholds
are metric-aware. Wall trimming uses adjusted t-intervals for all opening gaps.

`export_json.py` adds per-opening interval adjustment fields to `openings.doors` /
`openings.windows` entries. `openings.rejected` now also includes last-resort rejected
from conflict resolution.

`debug_overlay.py` draws original intervals (orange-red, offset above edge) and adjusted
intervals (teal-green, offset below edge) for any opening that was moved. Last-resort
rejected openings get a cross at their original midpoint with label `no-fit:<type>`.

`pipeline.py` passes `px_to_mm=scale_info.px_to_mm` to `trim_wall_intervals` and
`trimmed_graph=trimmed` to `build_debug_overlay`.

12 new tests in `TestOpeningIntervalDeOverlap` in `test_phase4_vectorization.py`
(total: 58 phase4 tests).

## 20. Pipeline Contract Enforcement (task34 — 2026-06-28)

Four contract violations fixed; source of truth is now the adjusted geometry state
at every stage: wall trim → SVG primitives → JSON → debug overlay.

### Part A — Adjusted Interval Propagation

`apply_adjusted_intervals_to_hosted_openings(trimmed_graph, hosted_openings)` added
to `wall_interval_editing.py`.  After `trim_wall_intervals()` adjusts opening
intervals, this function creates new `HostedOpening` objects whose `snapped_points`
match the wall trim endpoints exactly (from `opening_gaps[].snapped_points`).

`pipeline.py` now calls this after trimming to produce `final_doors` / `final_windows`.
All downstream consumers (`build_final_svg`, `build_final_vector_json`,
`build_debug_overlay`) receive these adjusted openings.

`export_json.py` adds `final_points`, `snapped_points_adjusted`, `snapped_points_original`
to each door/window record so the JSON carries the full adjustment audit trail.

### Part B — Evidence-Based Door Direction

`door_geometry.py` gains:
- `_score_arc_pixels()` — samples N points along a 90° arc, counts mask pixel hits
- `_score_line_pixels()` — samples N points along a line, counts mask pixel hits
- `infer_door_direction_from_evidence(p0, p1, door_arc_mask, door_leaf_mask)` — tests
  4 hypotheses (hinge∈{p0,p1} × swing∈{left,right}), picks highest arc-overlap score
- `compute_door_geometry()` now accepts `door_arc_mask` / `door_leaf_mask`; uses
  evidence scoring when a mask is provided, records `hinge_source` / `swing_source`

`pipeline.py` crops the door_arc mask to each door's component bbox (±8px margin)
so evidence from adjacent doors does not bleed. Reports `N/M from evidence, K fallback`.

`build_final_svg()` and `build_final_vector_json()` accept optional `door_geometries`
list (pre-computed `DoorGeometry` objects) so the SVG and JSON use evidence-driven
hinge/swing without recomputing geometry independently.

### Part C — Topology-Safe Wall Snap Before Buffering

`wall_buffering.py` gains `_topology_snap_edges(edges, tol=1.5px)`:
- Clusters all wall edge endpoints within `tol` pixels (union-find)
- Replaces each endpoint with its cluster centroid
- Drops zero-length and duplicate edges after snapping
- Returns `(snapped_edges, metrics)` where metrics = {`pre_buffer_node_count`,
  `post_snap_node_count`, `pre_buffer_edge_count`, `disconnected_endpoint_count`}

`buffer_wall_chains()` now runs `_topology_snap_edges()` before `shapely.linemerge()`
so nearly-coincident corners connect properly into chains.

`WallGeometry` dataclass gets four new fields: `pre_buffer_node_count`,
`post_snap_node_count`, `pre_buffer_edge_count`, `disconnected_endpoint_count`.

`build_final_vector_json()` merges these into the `metrics` dict automatically.

`debug_overlay.py` accepts optional `door_geometries`; draws:
- Yellow circle at hinge point with swing-side label per door
- White circles at final primitive endpoints (on top of cyan)

### Part D — Notebook Update

`notebooks/phase4_vectorization.ipynb` cell 3 replaced: the ~200-line inline
`vectorize_image()` duplicate of the pipeline is removed; replaced by a 60-line
wrapper that calls `run_phase4_pipeline()` directly.  All geometry (Parts A/B/C)
is in `src/` and autoreload keeps the notebook in sync without a kernel restart.
Summary dict now includes `disconnected_endpoints`, `pre_buffer_nodes`,
`post_snap_nodes`.

17 new tests (total: 75 phase4 tests):
- `TestAdjustedIntervalPropagation` (4 tests): Part A
- `TestDoorEvidenceScoring` (6 tests): Part B
- `TestTopologySnapEdges` (7 tests): Part C
