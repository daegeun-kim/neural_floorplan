# Spec v008: Orthogonal Point-Graph Mask-to-Vector Reconstruction

## 0. Purpose

This spec defines the fresh vectorization process for rebuilding `src/vectorization` from zero.

The active vectorizer must convert `segformer_b0_run3` 7-class CNN predictions into a clean architectural point graph, then export wall/window/door vectors from that graph.

The final vector output must be an architectural abstraction. It must not be a contour trace of noisy prediction pixels.

Pipeline:

```txt
7-class semantic prediction
-> decoded class masks
-> class components
-> searched architectural points
-> orthogonal point alignment
-> typed point connections
-> wall/window/door primitives
-> SVG output
```

Component primitive definitions are in `spec_v007_phase3_component_primitives.md`. This v008 spec controls reconstruction order, topology, validation, and implementation behavior under `src/vectorization`.

JSON, DXF, room graph inference, Grasshopper export, and interactive correction are out of scope.

## 1. Implementation Scope

This restart covers:

- reading 7-class class-ID masks
- decoding RGB prediction previews when needed
- extracting connected components for wall, window, door arc, door leaf, and door origin
- searching directly for the seven allowed point types
- aligning searched points into an orthogonal graph
- connecting points into wall, window, and door-origin segments
- generating door leaf and door arc geometry procedurally
- rendering final wall/window/door SVG output
- writing debug overlays and metrics for all rejected evidence

This restart excludes:

- floor generation
- room graph construction
- diagonal or 45-degree wall support
- retired 5-class `opening`, `room`, or `icon` assumptions
- contour-based fallback output

Floor may be added in a later spec once wall/window/door topology is reliable.

## 2. Active Input Classes

The expected input is a 7-class semantic prediction or decoded RGB semantic mask:

| ID | Class | Meaning |
|---:|---|---|
| 0 | background | outside or unused area |
| 1 | floor | ignored for this restart |
| 2 | wall | structural wall evidence |
| 3 | window | wall-hosted window evidence |
| 4 | door_arc | door swing arc evidence |
| 5 | door_leaf | open door leaf evidence |
| 6 | door_origin | wall-aligned door threshold/origin evidence |

Run3 preview palette:

| ID | Class | RGB |
|---:|---|---|
| 0 | background | `(200, 200, 200)` |
| 1 | floor | `(245, 240, 232)` |
| 2 | wall | `(30, 30, 30)` |
| 3 | window | `(60, 120, 220)` |
| 4 | door_arc | `(220, 90, 90)` |
| 5 | door_leaf | `(235, 140, 80)` |
| 6 | door_origin | `(160, 70, 180)` |

If RGB values do not match the configured palette, fail clearly unless an explicit RGB tolerance is configured.

The retired 5-class mapping must not be accepted:

```txt
0 background
1 wall
2 opening
3 room
4 icon
```

If a retired 5-class mask is passed to v008, raise an incompatible-input error.

## 3. Core Design Rules

The vectorizer is graph-first.

Required behavior:

- use wall/window/door evidence as high-confidence topology
- ignore floor for this restart
- search for the seven allowed point types directly
- align all final graph geometry orthogonally
- host every final window and door on wall topology
- replace wall intervals with hosted window and door-origin intervals
- generate door leaf and door arc from the accepted door-origin segment
- emit rejected evidence only in debug artifacts and metrics

Forbidden final behavior:

- free-floating windows
- free-floating doors
- floor/background-border wall tracing
- jagged wall contours
- raw door-leaf contour tracing
- raw door-arc contour tracing
- arbitrary-angle wall/window/door-origin output
- 45-degree wall output
- untyped final points
- unresolved final points
- debug or unidentified visible groups inside `vector.svg`

## 4. Coordinate, Direction, and Angle Conventions

All geometry uses image coordinates unless explicitly converted to metric units:

- `x` increases to the right
- `y` increases downward
- horizontal means constant `y`
- vertical means constant `x`
- directions are `left`, `right`, `up`, and `down`

Final wall, window, and door-origin graph segments must be orthogonal:

```txt
0 degrees
90 degrees
180 degrees
270 degrees
```

Do not support diagonal or 45-degree walls in this restart. Ambiguous or diagonal-looking evidence must either snap to the strongest supported orthogonal interpretation or be rejected into debug output.

Each point attachment must store:

- semantic type: `wall`, `window`, or `door_origin`
- direction leaving the point: `left`, `right`, `up`, or `down`
- source class evidence
- evidence length/area or confidence

Door leaf and door arc are generated after door-origin pairing. They are not point-search attachment types.

## 5. Scale Rules

The graph may be constructed in pixel space, but metric validation requires a scale.

Scale priority (task12 SS1):

1. explicit metadata, if available
2. red `door_arc` connected-component bounding-box long edge, evaluated against both `700 mm` and `900 mm` door modules
3. clustered door-origin widths - secondary cross-check/debug evidence only, never standalone scale-setting
4. clustered wall thicknesses - weak secondary cross-check/debug evidence only, never standalone scale-setting
5. unknown scale, only if no usable red `door_arc` cluster exists (door-origin/wall evidence alone never resolves scale)

For each connected red `door_arc` pixel cluster, take the long edge of its bounding box as a candidate standard door width, and evaluate `px_to_mm = 700 / long_edge_px` and `px_to_mm = 900 / long_edge_px`. When multiple clusters exist, vote/cluster over the resulting `px_to_mm` candidates, choose the candidate supported by the most clusters, and use the median of that winning group. Reject only obvious outliers; do not let noisy wall connected-component thickness override a resolved red door-arc scale.

Allowed door modules:

- `700 mm`
- `900 mm`

Allowed wall thickness modules:

- `100 mm`
- `200 mm`

Window minimum width:

- `300 mm`

Default point merge and axis alignment tolerance:

- `500 mm`

If scale is unknown, the implementation may still build a pixel-space graph, but metric-only decisions must be recorded as scale-blocked instead of using arbitrary pixel defaults.

Metric-only decisions include:

- validating a door width as `700 mm` or `900 mm`
- enforcing the `300 mm` minimum window width
- applying the `500 mm` point merge or axis alignment tolerance
- exporting dimensions as true millimeters

## 6. Source Module Instructions

The new implementation should be organized around explicit graph stages. Existing files may be replaced or simplified if needed.

Expected source layout:

```txt
src/vectorization/
  __init__.py
  decode_prediction.py
  masks.py
  components.py
  scale.py
  graph_types.py
  point_detection.py
  point_alignment.py
  point_connection.py
  door_geometry.py
  wall_geometry.py
  export_svg.py
  debug.py
  run_mask_to_vector.py
  primitives/
    base.py
    wall.py
    window.py
    door.py
```

Implementation responsibilities:

| Module | Responsibility |
|---|---|
| `decode_prediction.py` | load class-ID or RGB masks and reject incompatible input |
| `masks.py` | expose boolean masks for each active class |
| `components.py` | extract cleaned connected components and component metadata |
| `scale.py` | resolve or estimate `px_to_mm` and record scale status |
| `graph_types.py` | define point, attachment, edge, component, and validation data structures |
| `point_detection.py` | search directly for the seven allowed point types |
| `point_alignment.py` | align compatible points onto shared orthogonal axes |
| `point_connection.py` | build wall, window, and door-origin edges |
| `door_geometry.py` | infer hinge, door leaf, and 90-degree arc geometry |
| `wall_geometry.py` | turn connected wall/window graph edges into final SVG geometry |
| `export_svg.py` | write final SVG with only allowed visible groups |
| `debug.py` | write overlays, metrics, rejected evidence, and validation diagnostics |
| `run_mask_to_vector.py` | orchestrate the stage order and CLI/config entrypoint |

The implementation should expose intermediate artifacts for tests:

- decoded masks
- connected components
- searched points
- aligned points
- connection graph
- rejected evidence list
- final primitives
- validation report

## 7. Reconstruction Order

The vectorizer must run in this order:

```txt
0. load config
1. decode class-ID or RGB prediction
2. reject retired/incompatible masks
3. clean masks and extract connected components
4. resolve or estimate scale
5. search for the seven allowed point types
6. validate searched point counts and attachment directions
7. align compatible points onto orthogonal axes
8. connect aligned points into wall, window, and door-origin graph edges
9. validate graph topology
10. generate door leaf and door arc geometry
11. generate wall and window final geometry
12. export SVG
13. write debug overlay and metrics
```

Do not generate floor in v008 restart output.

## 8. Component Processing

Before point search, extract connected components for:

- `wall`
- `window`
- `door_arc`
- `door_leaf`
- `door_origin`

Recommended component metadata:

- class ID and class name
- connected-component ID
- area in pixels
- bounding box
- centroid
- skeleton or centerline
- endpoint candidates
- orientation estimate
- source mask slice

Remove components below configured area thresholds, but record removed components in metrics.

Door handling is driven by accepted red `door_arc` components. Each accepted red component should correspond to exactly one final door.

Window handling is driven by accepted blue `window` components. Each accepted window component should correspond to one final window unless nearby collinear blue fragments must be merged.

Purple `door_origin` and orange `door_leaf` evidence help locate a door associated with a red arc. They do not create a door without red evidence.

## 9. Point Detection

Point detection is not a generic keypoint detector followed by classification.

The implementation must search for each of the seven allowed point types directly, one by one. Therefore, there should be no unresolved final point category.

Every final detected point must be exactly one of these types:

| Point type | Required attachments | Meaning |
|---|---|---|
| `1_wall_point` | one wall segment | free-standing end of a wall |
| `2_wall_point` | two wall segments at 90 degrees | wall corner |
| `3_wall_point` | three wall segments | T-junction or branch from a continuous wall |
| `4_wall_point` | four wall segments | cross-junction with wall evidence in all cardinal directions |
| `wall_window_point` | one wall segment and one window segment in opposite directions | end of a wall-hosted window |
| `wall_door_hinge_point` | one wall segment and one door-origin segment in opposite directions | door hinge point on the wall |
| `wall_door_end_point` | one wall segment and one door-origin segment in opposite directions | far end of a door origin, not the hinge |

Each point must store a local attachment table.

Example:

```txt
point_type: wall_window_point
coordinate: (x, y)
attachments:
  - type: wall
    direction: down
    source: wall
  - type: window
    direction: up
    source: window
```

### 9.1 Wall Points

Search wall point types from wall skeleton or centerline evidence.

Rules:

- `1_wall_point` is a legitimate free-standing wall end.
- A `1_wall_point` must not be extended or snapped merely because another wall is nearby.
- If evidence shows a nearby branch or connection, detect a `3_wall_point` or `4_wall_point` instead of treating the endpoint as free-standing.
- A `1_wall_point` is valid only when the wall end is surrounded by background/floor pixels alone (task12 SS2.1). If window (`blue`), door_arc (`red`), door_leaf (`orange`), or door_origin (`purple`) evidence touches or sits immediately near the wall end, it must resolve to `wall_window_point`, `wall_door_hinge_point`, or `wall_door_end_point` instead - never `1_wall_point`. The neighborhood radius used for this check is an implementation judgment call.
- `2_wall_point` requires exactly two wall attachments at a 90-degree corner.
- `3_wall_point` requires one continuous wall direction pair plus one branch, or equivalent T-junction evidence.
- `4_wall_point` requires wall evidence in all four cardinal directions.

### 9.2 Window Points

Search `wall_window_point` endpoints from the transition between wall evidence and window evidence.

Rules:

- every final window has exactly two `wall_window_point` endpoints
- the two endpoints must be paired from the same window component or compatible merged window evidence
- each endpoint has one wall attachment and one window attachment in opposite directions
- the two window attachments must face each other along the window axis

### 9.3 Door Points (task16/task17: bbox-vertex method)

Search door points from red `door_arc` components first - the red bbox is trusted unconditionally (task17 "Red Bbox Assumption") once it passes one shape floor (task18): its bounding box's long/short side ratio must be at most `2:1` (square up to `2:1` accepted; a more elongated bbox such as `1:3` is rejected outright, before vertex selection runs at all - must-rule 53). Past that floor, vertex selection is never the reason a door is rejected.

`wall_door_hinge_point` and `wall_door_end_point` are 2 *adjacent* vertices of the red `door_arc` component's own bounding box - not points searched from mask intersections or arc-geometry inference:

- score each of the bbox's 4 edges by the combined count of purple (`door_origin`) and black (`wall`) pixels in a band straddling that edge (widened only perpendicular to the edge, never along it, so a short run of evidence near one edge cannot bleed into and tie with an adjacent edge's band)
- the edge with the highest score is the wall-facing edge; its 2 endpoint vertices are the hinge and end candidates - this is always evaluated and a pair is always selected, even when every edge scores 0 (task17: vertex selection itself never rejects a cluster)
- of those 2 vertices, the one with more nearby orange (`door_leaf`) evidence is the hinge (the leaf pivots open from the hinge, not from the end) - the door-leaf clue is what tells the hinge apart from the end once the wall-facing edge itself is known

Rules:

- each accepted red `door_arc` component requires one `wall_door_hinge_point`
- each accepted red `door_arc` component requires one `wall_door_end_point`
- red pixels define the existence and count of doors
- purple door-origin evidence without a red arc is debug-only and never creates a door (must-rule 52)

After the 2 vertices are chosen, the raw hinge-to-end span is snapped to the nearest `700`/`900 mm` module (## 5). Each point is then hosted onto the nearest real wall edge independently (a door commonly sits between two separate wall fragments, one on each side) - task17: this hosting search must prefer a wall edge whose own running direction matches the hinge-to-end vector's orientation (sharing the same axis the door's own gap shares) over a merely-closer but orientation-incompatible wall, since a point can only ever connect into a wall chain whose direction label matches its own attachment direction (## 12.1) - falling back to plain nearest-by-distance only when no orientation-matching wall exists within range.

### 9.3.1 Forceful Inference Rule (task13, revised task16/task17)

A connected red `door_arc` cluster is a door, always, once it survives component cleanup (task17: bbox vertex selection never rejects regardless of how little evidence exists). Purple/black evidence (edge scoring) and orange evidence (hinge/end disambiguation) refine the geometry but never decide whether the red cluster is a door:

- `wall_door_hinge_point` and `wall_door_end_point` must each lie within `200 mm` of the red cluster's bounding box (the point may sit just outside the box, not only inside it, once the width-module snap is applied). Scale must be resolved before this check runs.
- fragmented, missing, or noisy purple/orange evidence must lower the resulting `door_confidence` but must not delete the door.
- a red cluster may be rejected only when it is below the minimum component area, when no real wall exists anywhere to host the resulting hinge/end at all, or when even the snapped width's hinge/end can't land within the `200 mm` floor.

## 10. Point Detection Invariants

After point search:

- every final point is one of the seven allowed point types
- every attachment has a type and direction
- every attachment direction is cardinal
- `wall_window_point` count is even
- accepted red `door_arc` component count equals `wall_door_hinge_point` count
- `wall_door_hinge_point` count equals `wall_door_end_point` count
- no final door exists without an accepted red `door_arc` component
- no final window exists without exactly two compatible `wall_window_point` endpoints

If an invariant fails, reject only the affected component or graph region when possible, write diagnostics, and do not export the affected primitive as final geometry.

## 11. Point Alignment

Point alignment converts approximate pixel detections into clean orthogonal graph coordinates.

### 11.1 Axis Alignment (task17: door-anchored, distance-only, per-axis-independent)

Red `door_arc` bbox vertices (`wall_door_hinge_point`/`wall_door_end_point`) are trusted anchors and are never moved by this step - they were already established directly from the bbox (## 9.3). `wall_point` and `wall_window_point` ("followers") cluster onto a shared axis with *each other* whenever their coordinates are within the tolerance on *either* axis, checked independently per axis and regardless of how far apart they are on the other one (e.g. if two points' `x` difference is `500 mm`, their `x` is forced equal regardless of their `y` difference) - then any follower within tolerance of a door anchor on a given axis snaps (overrides, not averages) onto that anchor's exact value. Door anchors always win; a follower never pulls a door anchor toward it.

If two or more points have nearly equal `x` coordinates (within tolerance), they share one vertical axis - their `x` is forced exactly equal.

If two or more points have nearly equal `y` coordinates (within tolerance), they share one horizontal axis - their `y` is forced exactly equal.

The default tolerance is `500 mm` after scale resolution (task17: reverted from task15's `1000 mm` experiment - door-derived axes are the trusted anchors now, so the goal is no longer to over-merge distant geometry by raw distance alone). If scale is unavailable, use a configured pixel fallback only for graph construction and mark metric alignment decisions as scale-blocked.

After alignment:

- points on the same vertical axis have exactly equal `x`
- points on the same horizontal axis have exactly equal `y`
- final edges are horizontal or vertical only
- a door anchor's own coordinate is identical before and after this step

### 11.2 Alignment Evidence (task16: removed for generic clustering)

Generic distance-based clustering (## 11.1) requires no semantic evidence beyond the coordinate distance itself - task15's wall-pixel-corridor requirement was dropped per the literal task16 instruction. Two genuinely unrelated followers that happen to be within tolerance on one axis are still aligned to each other (or to a nearby door anchor); `point_connection.py`'s separate axis *connection* step (## 12.1) is what continues to require real wall pixel evidence before drawing an edge between two now-aligned points, so a spurious alignment does not by itself produce a spurious wall.

The wall-skeleton-edge pass (a continuous wall chain's own two endpoints) and the opening-pair pass (a window's two points, or a door's hinge/end) remain priority, exact, evidence-grounded passes that run *before* generic clustering and are re-asserted once more *after* the door-anchor snap - so a door's own hinge-to-end span (or a window's own two points) is never collapsed by either pass moving either side independently.

### 11.3 Window and Door Direction Alignment

For `wall_window_point`, `wall_door_hinge_point`, and `wall_door_end_point`, the opening direction decides which axis is shared.

Examples:

- if a window starts to the right of one point and a nearby compatible point has a window starting to the left, align those points on the horizontal axis by giving them the same `y`
- if a door-origin segment leaves one point downward and leaves the paired point upward, align those points on the vertical axis by giving them the same `x`

Window and door-origin pair alignment must prioritize the axis implied by the opening segment direction over generic coordinate clustering.

## 12. Point Connection

Point connection creates graph edges from aligned points.

Only connect points that are compatible by:

- point type
- attachment type
- attachment direction
- shared axis
- semantic evidence
- absence of an unrelated point between them

### 12.1 Wall Connections

Connect two points with a wall segment when:

- both points have wall attachments on the same aligned axis
- their wall attachment directions face each other
- no unrelated point lies between them on that axis
- wall evidence supports the interval, or opening evidence indicates a gap that will be replaced by a hosted window or door

When wall segments meet at a point, they are part of one wall graph. Final wall polygon generation must treat connected wall graph edges as joined geometry instead of unrelated capped segments.

Do not duplicate already-consumed outer-wall evidence as inner-wall geometry.

### 12.2 Window Connections

Connect two `wall_window_point` endpoints with a window segment when:

- the endpoints share the opening-implied aligned axis
- their window directions face each other
- they belong to the same accepted or merged window evidence
- no unrelated point lies between them
- the length is at least `300 mm` when scale is known or estimated

The window segment replaces the wall segment over the same interval.

Adjacent wall endpoints and window endpoints must coincide exactly after graph construction.

### 12.3 Door-Origin Connections

Connect one `wall_door_hinge_point` and one `wall_door_end_point` with a door-origin segment when:

- the endpoints share the opening-implied aligned axis
- their door-origin directions face each other
- they are paired with the same accepted red `door_arc` component
- no unrelated point lies between them
- the length snaps to `700 mm` or `900 mm` when scale is known or estimated

The door-origin segment replaces the wall segment over the same interval.

Door count and door location are driven by accepted red `door_arc` components, not by purple door-origin pixels alone.

## 13. Door Leaf and Door Arc Generation

After the door-origin segment is accepted, generate door leaf and arc procedurally.

### 13.1 Door Leaf

The door leaf:

- starts at the `wall_door_hinge_point`
- is perpendicular to the door-origin segment
- has the same length as the door-origin segment
- opens to the side indicated by orange `door_leaf` and red `door_arc` evidence
- is exported as a thin symbolic orange line

Do not trace the raw orange door-leaf pixel contour as final geometry.

### 13.2 Door Arc

The door arc:

- has center at the hinge point
- has radius equal to the door-origin length
- spans exactly 90 degrees
- connects the closed-door direction to the opened leaf direction
- follows the side indicated by red `door_arc` evidence
- is exported as a thin symbolic red arc

Do not trace the raw red arc component as an irregular contour.

The SVG arc flags must be computed from hinge, origin-end, and leaf-end geometry so the rendered arc remains centered on the hinge point for every orthogonal wall orientation.

## 14. Output Geometry Rules

Final visible SVG groups:

```txt
wall
window
door
```

No `floor` group is required for this restart.

Required drawing order:

```txt
wall
window
door
```

Required colors:

```txt
wall        black   #000000
window      blue    #3c78dc
door_origin purple  #a046b4
door_leaf   orange  #eb8c50
door_arc    red     #dc5a5a
```

Geometry representation:

| Component | Final geometry |
|---|---|
| wall | closed filled polygon generated from connected wall graph edges |
| window | closed filled polygon generated from hosted window graph edge |
| door_origin | thin symbolic SVG line |
| door_leaf | thin symbolic SVG line |
| door_arc | thin symbolic 90-degree SVG arc |

Wall thickness:

- normalize to `100 mm` or `200 mm`
- use evidence to choose the module when scale is known or estimated
- keep measured pixel thickness when scale is unknown

Window thickness:

- use `100 mm` total width when scale is known
- use half the host wall pixel thickness when scale is unknown

Door components are not offset into polygons.

Required root SVG metadata:

```txt
data-unit="mm" or "px"
data-scale-status="resolved" | "estimated" | "unknown"
data-px-to-mm="..."
data-scale-source="..."
```

No debug group, dashed unresolved marker, or retired-class group may appear in `vector.svg`.

## 15. Debug Output and Metrics

Required files per sample:

```txt
input.png
prediction.png
vector.svg
metrics.json
debug_overlay.png
```

Debug output must show or record:

- decoded class masks
- cleaned components
- removed tiny components
- searched points by type
- rejected candidate evidence (task19: `metrics.json` only - no longer drawn in `debug_overlay.png`, see below)
- aligned axes
- wall/window/door-origin graph edges
- door hinge choices
- inferred door origins
- scale estimate and confidence
- validation failures
- every red `door_arc` candidate bounding box and its inferred `wall_door_hinge_point`/`wall_door_end_point`, with the candidate's bbox+connector line color distinguishing low-confidence from high-confidence (task13) - the hinge/end points themselves are drawn in their own final `POINT_COLORS` (orange hinge / purple end), not the candidate's confidence color (task19)

`metrics.json` must additionally include, under the scale diagnostics (task12 SS1):

```txt
red_arc_bbox_long_edges_px
red_arc_px_to_mm_candidates
red_arc_selected_modules_mm
selected_px_to_mm
scale_source
scale_rejected_outliers
```

and one record per accepted red `door_arc` cluster (task13 "Metrics Requirements", extended task17), so it is obvious whether the cluster became a door and how its hinge/end points were inferred:

```txt
red_component_id
red_bbox
red_bbox_long_edge_px
created_door_candidate
scale_candidate_px_to_mm
hinge_candidate_support_classes
end_candidate_support_classes
hinge_distance_to_red_bbox_mm
end_distance_to_red_bbox_mm
door_confidence
door_inference_notes
all_four_bbox_vertices
selected_hinge_vertex
selected_end_vertex
hinge_vertex_score
end_vertex_score
selected_bbox_edge
host_wall_alignment_score
door_width_mm
```

Rejected evidence belongs only in `metrics.json` (task19); low-confidence/forced-inference door candidates belong only in `debug_overlay.png` and `metrics.json`. Neither ever appears in `vector.svg`.

## 16. Configuration

Expected config file:

```txt
configs/vectorization_v008.yaml
```

Required config shape:

```yaml
input:
  prediction_path:
  palette: run3
  rgb_tolerance: 0

scale:
  allow_estimated_scale: true
  door_width_modules_mm: [700, 900]
  wall_thickness_modules_mm: [100, 200]
  min_scale_confidence_for_metric: 0.70

geometry:
  allowed_angles_deg: [0, 90, 180, 270]
  allow_diagonal_walls: false
  point_merge_tolerance_mm: 500
  axis_alignment_tolerance_mm: 500
  min_segment_length_mm: 100

components:
  min_wall_area_px: 4
  min_window_area_px: 4
  min_door_arc_area_px: 4
  min_door_leaf_area_px: 2
  min_door_origin_area_px: 2

windows:
  min_width_mm: 300
  require_two_wall_window_points: true
  replace_wall_segment: true

doors:
  require_arc_group: true
  infer_origin_when_purple_missing: true
  reject_without_red_arc: true
  hinge_intersection_tolerance_px: 6
  hinge_snap_to_wall_max_dist_px: 40
  door_width_modules_mm: [700, 900]
  replace_wall_segment: true

output:
  include_floor: false
  write_debug_overlay: true
  write_metrics: true
```

## 17. Validation Requirements

Tests must cover:

1. 7-class class-ID mask decoding.
2. RGB run3 palette decoding.
3. Retired 5-class mask rejection.
4. Floor class is ignored by this restart.
5. Scale resolution from door modules `700 mm` and `900 mm`.
6. Scale cross-check from wall modules `100 mm` and `200 mm`.
7. Only orthogonal final graph edges are exported.
8. 45-degree wall evidence is rejected or orthogonally snapped; no 45-degree final wall is produced.
9. The seven allowed point types are searched directly.
10. No unresolved final point type exists.
11. `1_wall_point` remains a valid free wall end and is not auto-connected to nearby topology.
12. Evidence for a nearby branch creates `3_wall_point`, not a forced `1_wall_point` extension.
13. `wall_window_point` endpoints pair by opposing window directions.
14. Window pair direction determines axis alignment.
15. Window length is at least `300 mm` when scale is known.
16. Window replaces the corresponding wall interval.
17. Red `door_arc` components determine final door count.
18. Purple/orange door evidence without red arc is rejected from final output.
19. Red arc with missing/noisy purple origin can infer a door-origin segment from host wall and orange/red evidence.
20. Door hinge prefers orange/purple intersection when present.
21. Door hinge can be inferred from red arc geometry when purple/orange intersection is missing.
22. Door origin length snaps to exactly `700 mm` or `900 mm` when scale is known.
23. Door leaf starts at the hinge and is perpendicular to the door origin.
24. Door arc is centered on the hinge and spans exactly 90 degrees.
25. Wall graph edges sharing endpoints render as connected polygon geometry with clean joins.
26. Wall final geometry is black closed polygons.
27. Window final geometry is blue closed polygons.
28. Door origin is a thin purple symbolic line.
29. Door leaf is a thin orange symbolic line.
30. Door arc is a thin red symbolic arc.
31. Final SVG contains only `wall`, `window`, and `door` visible groups for this restart.
32. Final SVG contains no debug, dashed, retired `room`, retired `icon`, or generic `opening` group.
33. `metrics.json` records rejected components and validation failures.
34. `debug_overlay.png` includes rejected/unpaired evidence.
35. Red `door_arc` bounding-box long edge resolves scale to `700 mm` or `900 mm`, and beats conflicting wall-thickness evidence (task12).
36. Multiple red `door_arc` clusters use robust candidate voting/median scale, rejecting only obvious outliers (task12).
37. `1_wall_point` is suppressed in favor of a window/door point type when window/door evidence sits near the wall end (task12).
38. `wall_door_hinge_point` and `wall_door_end_point` each resolve within `200 mm` of their red cluster's bounding box (task12/task13).
39. A red `door_arc` cluster is never rejected merely because purple/orange evidence is fragmented, missing, or noisy - the door is created with a forced hinge/end inference and a lower `door_confidence` instead (task13).
40. Door count equals accepted red `door_arc` cluster count, and `metrics.json` reports one door-candidate record per accepted red cluster (task13).
41. Wall graph construction does not require accurate `1`/`2`/`3`/`4` wall-point subtype classification - junctions and free ends all finalize as one generic `wall_point` carrying their real attachment directions (task15).
42. After axis alignment, any two wall-participating points (generic `wall_point`, `wall_window_point`, `wall_door_hinge_point`, `wall_door_end_point`) that share an exact aligned axis and have corridor wall evidence between them connect, even across a noisy mask break the original skeleton chain never spanned (task15).
43. The default point merge/axis-alignment tolerance is `1000 mm` (task15).
44. Two points are never wall-connected unless they share the same aligned axis exactly - no "close enough" fallback (task15).
45. An opening's own two endpoints (a window's paired points, or a door's hinge/end) are never bridged by a parallel `wall` edge, even when both land on the same aligned axis - the opening already replaces that wall interval (task15).
46. A red `door_arc` cluster's recognition as a door is not gated by hosting-search radius - only must-rule 17's `200 mm` arc-bbox-proximity floor can reject it (task15).
47. Two `wall_point`/`wall_window_point` followers share an axis whenever they are within the configured tolerance (`500 mm`, task17) on that axis alone, independently per axis, regardless of their distance on the other axis - no corridor/wall-evidence gate (task16, tolerance reverted task17).
48. An opening's own two endpoints (a window's paired points, or a door's hinge/end) are never collapsed onto each other by generic axis clustering - their own shared axis is re-asserted once more after generic clustering runs (task16).
49. `wall_door_hinge_point` and `wall_door_end_point` are 2 *adjacent* of the red `door_arc` bounding box's own 4 vertices, chosen by combined purple/black evidence along the bbox edges (with orange disambiguating hinge from end) - not searched from mask intersections or arc-geometry inference (task16/task17).
50. Bbox-vertex selection for a door never rejects a cluster, even when every one of its bbox's 4 edges has zero purple/black evidence - only the downstream min-area, no-host-wall, and `200 mm`-floor checks may still reject (task17, supersedes task16's zero-evidence vertex rejection).
51. A `wall_point`/`wall_window_point` follower within axis-alignment tolerance of a red door_arc's `wall_door_hinge_point`/`wall_door_end_point` snaps onto that door anchor's exact coordinate on that axis; the door anchor itself is never moved by alignment (task17).
52. A door's hinge/end hosting search prefers a wall edge whose own running direction matches the hinge-to-end vector's orientation over a merely-closer but orientation-incompatible wall (task17).
53. A red `door_arc` bbox is rejected outright, before vertex selection, when its long/short side ratio exceeds `2:1` (task18).

## 18. Task14 Debugging Notes

task14 debugged the v008 implementation against `specs/vectorization_must_rules.md` using the required test case (`outputs/vectorization/v008/iteration5_run3`). The must-rules and this spec were already correct; the following implementation bugs in `src/vectorization` caused observed output to violate them and were fixed:

- `primitives/scale.py`: scale resolution required the winning red-`door_arc` voting group to cover >= `min_scale_confidence_for_metric` of *all* clusters (including ordinary noise) before reporting `scale_status="estimated"` at all, collapsing scale to `"unknown"` whenever several clusters were noisy even though a clean winning group existed - contradicting SS5 priority item 5 ("unknown ... only if no usable red door_arc cluster exists"). `confidence` is now reporting metadata only; resolution itself only requires a usable winning group. The same over-gating was removed from `snap_to_module_mm` and from the window/door creation checks in `point_detection.py` - once scale is resolved/estimated, mm conversion and object creation proceed regardless of confidence.
- `point_detection.py` `_detect_door_points`: a red `door_arc` cluster paired with a fragmented/tiny `door_origin` component (implausibly narrow projected width) was rejected outright instead of falling back to arc-geometry inference, per SS9.3.1's forceful-inference rule. `RejectedEvidence` records for scale-blocked/too-narrow/non-cardinal-axis door rejections were also attributed to the wrong component (`door_origin`/`origin_id`, often `None`) instead of the originating red `door_arc` cluster, breaking the per-cluster metrics report (SS15).
- `point_detection.py` `_hinge_snap_to_wall`: picked the host wall by raw nearest-distance only, unreliable at corners/junctions; now reuses the same orientation/overlap/remainder-scored `select_host_wall_for_opening` window hosting already uses.
- `point_detection.py` `build_wall_skeleton_graph`: per-chain wall thickness used the *whole wall component's* overall `minAreaRect` short axis, which is meaningless (and can be hundreds of px) for a large non-rectangular component such as a full outer-wall loop - this wildly inflated the final wall polygon's buffer width. Replaced with a local thickness sampled from a distance transform along each chain's own pixels.
- `point_connection.py` `build_wall_edges`: only ever connected a wall edge between a skeleton chain's own pre-existing two endpoints, never splitting at a window/door point hosted mid-chain (the common case when the CNN wall mask has no real pixel gap at the opening) - leaving that point with no wall edge at all (SS3/SS14 "host every final window and door on wall topology"). Chains are now split at any window/door point that projects onto them. A real wall free end recognized as "near opening evidence" (SS9.1) but just outside the tight node-match tolerance is now also connected to that opening point via a dedicated, narrowly-scoped fallback.
- `point_connection.py` `validate_graph`: flagged the *correct* topology (a window/door point covered by one wall edge plus one opening edge) as `opening_point_multiple_edges`, and never checked for the real failure mode (zero wall edges = floating). Coverage is now counted per edge type; a new `floating_window_point`/`floating_door_point` issue catches genuine floating openings.
- `wall_geometry.py` `window_edges_to_primitives`: window thickness was half the host wall's own pixel thickness, which only equals SS14's required `100 mm` when the host wall happens to be the (no longer universal) `200 mm` module - now uses the fixed `100 mm` constant directly whenever scale is resolved.

A follow-up pass (still task14, after also checking the previously-unexamined `sample_005`) found two further bugs of the same class, both rooted in how an opening's host wall is found and reconnected:

- `point_detection.py` `select_host_wall_for_opening`: measured distance from the opening's pixel **centroid** to each candidate wall, instead of the minimum distance from any of the opening's own pixels. A tall/elongated opening (e.g. a window spanning most of a wall's height) commonly has its host wall split into two short skeleton chains - one ending just above the opening, one starting just below it - and the centroid sits roughly equidistant from both, often beyond `max_wall_dist`, even though the opening's own top/bottom extremity is only a few px from one of them. This was the direct cause of `sample_005` having zero windows despite a clearly real, visible window in the source image. Fixed via a new `_min_pixel_distance_to_wall` helper used for all hosting decisions (windows and doors).
- `point_connection.py` `build_wall_edges`: once an opening's host wall is fixed at detection time, reconnecting it back into the wall graph still relied on a fixed pixel tolerance (`opening_match_tolerance_px`) to guess which nearby point belonged to which chain - which fails for the same reason above (the real gap between an opening's boundary and its host chain's literal pixel endpoint can be far larger than any one fixed tolerance, without being unbounded). `point_detection.py` now records the exact `host_wall_edge_id` on every window/door point it creates (`GraphPoint.host_wall_edge_id`), and `build_wall_edges` uses that exact reference - building each chain's ordered list of stops (its own natural endpoints plus every point hosted on it, by id, regardless of how far outside `[0, chain_length]` the point's projection falls) - instead of re-guessing the relationship by coordinate distance. The geometric/distance-based fallback remains only for points with no recorded host edge (hand-built test fixtures).

Remaining, not-fully-resolved for the required test cases (documented per task14 acceptance criterion 6, not further pursued to avoid pipeline redesign):

- A few window/door points on the most heavily fragmented wall corners still end up with no wall edge at all, because *no* wall chain - not even an extrapolated/extended one - comes within any reasonable distance of that specific point. Confirmed by direct measurement in `sample_005`: the nearest candidate is itself a disconnected 1-2px skeleton noise fragment with no further neighbor for tens of px in the relevant direction. This is a genuine gap in the CNN wall mask / skeleton, not a fixed-tolerance or coordinate-guessing problem (those were the two bugs fixed above) - closing it fully would require either stitching disconnected skeleton fragments into one continuous host line before projecting, or accepting a host wall whose own pixel evidence doesn't reach the opening at all. Left as a visible, correctly-flagged (`floating_window_point`/`floating_door_point`) failure rather than silently hidden, per the observable-failure rules.
- A small number of final wall polygon edges are within a few degrees of axis rather than exactly orthogonal, where a long wall run is split across more than one chained skeleton segment and `point_alignment.py`'s per-pair axis snapping doesn't compound transitively across the whole run.

## 19. Task15 Debugging Notes

task15 merged `1_wall_point`/`2_wall_point`/`3_wall_point`/`4_wall_point` into one generic `wall_point` for wall-graph construction, raised the axis-alignment tolerance to `1000 mm`, and lowered the red `door_arc` recognition threshold so almost any surviving red cluster becomes a door (see `## 17` items 41-46). Continuing to debug the required test case (`outputs/vectorization/v008/iteration5_run3`) against `specs/vectorization_must_rules.md` after that change found the following further implementation bugs, all variants of the same root issue task14 already named (an opening's true host wall can be a wall fragment other than its own immediate skeleton neighbor):

- `point_detection.py` `_hinge_snap_to_wall`/`select_host_wall_for_opening`: door hinge hosting reused the window-hosting tie-break, which scores a candidate wall by how well the *opening's own pixel shape* orientation matches the wall's orientation. That assumption holds for a window (a thin rectangle aligned with its host wall) but not for a door_arc (a swing-wedge shape uncorrelated with which wall it sits on) - a short, incidentally diagonal skeletonize corner-noise stub could outscore the real, much longer wall right next to it, then get its candidate point projected (unclamped) far past its own real pixel extent. Added `_select_door_host_wall`: disqualifies a nearest candidate under `MIN_PLAUSIBLE_DOOR_HOST_LENGTH_PX` (10px) in favor of a longer one within the same ambiguity window, falling back to the window-style score only when both candidates are plausibly real wall fragments (e.g. two stubs flanking the door, one each side).
- `point_detection.py` `_detect_door_points`: a door's hinge and far/end point were always hosted on the *same* wall edge - whichever `_hinge_snap_to_wall` picked using the whole arc's pixel shape - even though a door commonly sits between two separate wall stub fragments, one on each side. When the far point's projection onto that shared edge fell outside the edge's own `[0, 1]` span, it is now independently re-targeted (by plain nearest-distance) to whichever real wall edge it actually lands within range of; only the recorded host id is retargeted, never the coordinate itself, so an already-exact `700`/`900 mm` snap is never perturbed by the new host's own pixel-rounded line angle.
- `point_connection.py` `build_wall_edges`: chained a skeleton edge's stops (its own two endpoints plus every window/door point hosted on it) by checking every consecutive pair against that *edge's own* `dir_from_start`/`dir_from_end` labels - correct only for the pair literally at the edge's two true endpoints. A host point legitimately extrapolated beyond the edge's real extent (same scenario as above) sits outside that two-label relationship, so a pair involving it now derives its required direction from that pair's own coordinates instead, falling back to the original (well-tested) edge-label check whenever both stops are within the edge's real `[0, seg_len]` span.
- `point_connection.py` `_connect_axis_aligned_points`: the task15 axis-bridging pass connected *any* two same-axis, corridor-evidence-backed wall-participating points - including, once a door's hinge and end legitimately land on the same aligned axis (the normal case, being the two ends of one straight opening span), the door's own two points. That created a `wall` edge running directly across the door's own gap, parallel to its `door_origin` edge - the door would have rendered with a solid wall polygon over its own opening. A new `_same_opening_pair` guard skips a pair sharing a source component id whose point types are a window pair or a hinge/end pair.
- `point_connection.py` `validate_graph`: required every window *and* door point to individually carry >= 1 `wall` edge, but only window hosting is phrased that way in the must-rules (rule 78: "window **endpoints** must connect... each"); door hosting (rule 82) is phrased at the door-origin level. A door legitimately sitting at the end of a wall run has only one side (hinge or end) continuing into more wall topology - the other side alone having no further wall edge is no longer flagged as `floating_door_point` as long as its paired hinge/end point has one.
- `run_mask_to_vector.py`: passed the *total* `door_arc` component count (after pixel-area cleanup only) as `accepted_door_arc_count` to `validate_points`, so a cluster legitimately rejected afterwards by rule 51 (too far from any plausible wall/door geometry even after fallback, must-rule 17's `200 mm` floor) was double-counted as a `door_count_mismatch` even though the rejection was itself correct. Now subtracts components point_detection.py itself logged as rejected (`class_name == "door_arc"` in `rejected_evidence`) before comparing.

These fixes brought `sample_003` and `sample_004` to zero `metrics.json` validation issues. `sample_005` still reports one `floating_window_point` and one floating door pair (`doorpt_4`) - confirmed (per task14's already-documented limitation, re-verified, not newly introduced by any task15 fix above) to be a genuine sparse-wall-mask gap: the real free wall end on that side of each opening is itself missing from the skeleton graph (suppressed as "near opening evidence" with no wall chain to take its place), not a host-selection or reconnection bug. Left as a visible, correctly-flagged failure per the observable-failure rules rather than pursued further (would require stitching disconnected skeleton fragments or accepting a host whose evidence doesn't reach the opening - out of task15's scope).

## 20. Task16 Debugging Notes

task16 made two deliberate behavior changes (not bug fixes) plus the bug fixes those changes exposed:

**Axis alignment** (`point_alignment.py`): generic clustering (## 11.1/11.2) dropped the task15 wall-pixel-corridor evidence requirement and the wall-only 15px cap entirely - every searched point type now clusters onto a shared axis purely by single-axis coordinate distance, within the full `1000 mm` tolerance, independent of the other axis. Implementing this surfaced two real bugs, not just config changes:

- A door's hinge and its own end (or a window's own two points) commonly differ by less than `1000 mm` on *both* axes - one axis only because they intentionally already share it (the wall-facing axis), the other because door/window widths are themselves well under a meter. Running the x-clustering pass and the y-clustering pass independently let each side of the pair join *different* unrelated groups, which can force both axes equal between hinge and end and collapse the door to a single point (zero width). Fixed by excluding an opening's own pair from matching each other in the generic clustering pass (`_same_opening_pair`, mirroring the existing guard of the same name in `point_connection.py`) and by re-running the exact opening-axis assertion once more *after* generic clustering, restoring cardinality regardless of how either side was nudged individually.
- `_opening_pair_groups` grouped exclusively by `source_component_ids`, but component ids are assigned independently per mask class (`components.py`) - a window component and a door_arc component can legitimately share the same id number. This let an unrelated window pair and door pair collide into one bogus group of 4, silently no-op'ing the opening-axis assertion for both (`len(group) != 2`) and leaving one of them diagonal. Fixed by keying the group by `(category, sorted(source_component_ids))` instead of the id alone - a pre-existing bug, latent until task16's added post-clustering re-assertion pass made it visible.

**Door generation** (`point_detection.py`): `_detect_door_points` was rewritten so `wall_door_hinge_point`/`wall_door_end_point` come directly from 2 of the red `door_arc` bounding box's 4 vertices - the edge with the strongest combined purple+black evidence (widened only perpendicular to itself, so adjacent edges' bands don't bleed into each other and tie), with orange evidence picking which of that edge's 2 endpoints is the hinge. This fully replaces the old orange/purple-intersection search, arc-geometry inference, paired-door-origin search, and bespoke host-wall tie-break (`_select_door_host_wall`) - all removed. Downstream width-module snapping, the `200 mm` floor check, and per-point host-wall hosting (`nearest_wall`, unbounded search radius) are unchanged from task15.

Both changes were verified against the unit suite (rewriting `TestForcefulDoorInference`'s 6 tests whose premises were specific to the retired door mechanism or to the old evidence-gated/capped alignment - all other tests, including the door/window/wall geometry and connection suites, passed unmodified) and against the required real samples in `outputs/vectorization/v008/iteration5_run3`:

- `sample_003` and `sample_004`: zero `metrics.json` validation issues, with door counts now matching every accepted red `door_arc` component (previously several clusters were silently dropped by the old mechanism's stricter pairing requirements).
- `sample_005`: one remaining `floating_window_point` (`winpt_1_a`). Confirmed by direct reproduction with alignment fully bypassed that this is **not** caused by either task16 change - `winpt_1_a`'s own coordinate is untouched by clustering in this run. It is a latent bug in `select_host_wall_for_opening`'s window-hosting choice (assigns `host_wall_edge_id` to a geometrically unrelated, disconnected skeleton stub far from the window's true location) that the now-stricter post-clustering checks simply expose rather than mask. Same class of issue as task14/task15's already-documented `sample_005` limitation; left unresolved as out of task16's two-item scope (axis alignment, door generation) rather than pursued into window-hosting logic.

## 21. Task17 Debugging Notes

task17 made the vectorization process door-first: red `door_arc` bboxes are trusted anchors the rest of the graph builds from, rather than one input among many resolved by mutual, symmetric clustering.

**Door point selection** (`point_detection.py`): vertex selection (## 9.3) no longer rejects a cluster for missing evidence at all - `select_door_hinge_end_from_bbox` always returns a pair, even when every edge scores 0 (task16 had a `best_score <= 0` rejection; task17 dropped it per "Red Bbox Assumption" - "do not overcomplicate red-bbox acceptance with many early rejection conditions"). Per-vertex scores reported in `DoorCandidateRecord`/metrics are the same edge-band score attributed to both of an edge's vertices (confirmed as acceptable - edge-based and per-vertex scoring are the same selection described two ways).

**Door-anchored alignment** (`point_alignment.py`): `wall_point`/`wall_window_point` ("followers") first cluster among themselves the same distance-only way task16 introduced, then any follower within tolerance of a door anchor (`wall_door_hinge_point`/`wall_door_end_point`) on a given axis snaps onto that anchor's exact value, overriding whatever the follower-only pass decided - door anchors are never moved. The axis-alignment tolerance reverted to `500 mm` (task15's `1000 mm` experiment is retired).

**Two real bugs found while verifying against the required samples**, both about door hinge/end ending up unconnectable to any wall topology ("not reliably aligned with wall lines" per task17's own framing of the prior failure):

- An initial attempt added a `host_wall_direction` parameter to `select_door_hinge_end_from_bbox` that restricted candidate edges to the orientation of whichever wall `nearest_wall` found closest to the bbox's *centroid*, intending to satisfy "the selected adjacent pair should be the bbox edge with the closest alignment to the host wall." This was the wrong place to enforce that: `nearest_wall`'s nearest-by-raw-distance pick is frequently a short, unrelated skeleton noise fragment rather than the real host wall, so restricting vertex selection to *its* orientation overrode an already-correct, well-evidenced selection more often than it fixed a wrong one (verified directly - removing the restriction fixed one required sample's floating door while leaving the others at zero issues; keeping it broke a previously-correct door in a different sample). Reverted; vertex selection relies solely on its own local purple/black evidence (## 9.3), which already concentrates on the true host-wall-facing edge in practice.
- The real bug was in **hosting**, not selection: once hinge/end are chosen, each is hosted onto the nearest real wall edge (`nearest_wall`) independently - but a door's hinge/end attachment direction is always along the same axis the hinge-to-end vector itself shares (the door's own gap continues its host wall's line), and `point_connection.py`'s `build_wall_edges` only connects a point into a chain whose direction label actually matches that attachment. Plain nearest-by-distance hosting could pick an orientation-incompatible wall edge that happened to be marginally closer than the real, orientation-matching one, silently producing an unconnectable (floating) hinge/end. Fixed via `_nearest_wall_matching_orientation`: prefer the nearest wall edge whose own running direction matches the door's orientation, falling back to plain nearest-by-distance only when no matching-orientation wall exists within range.

These fixes brought `sample_003` and `sample_004` to zero `metrics.json` validation issues, with every accepted red `door_arc` component producing exactly one door. `sample_005` retains the same single `floating_window_point` already documented in task15/task16's notes (confirmed, by direct reproduction with alignment bypassed entirely, to be a latent `select_host_wall_for_opening` window-hosting bug unrelated to any task17 change) - left unresolved as out of task17's scope.

## 22. Task18 Debugging Notes

task18 added one geometric acceptance floor ahead of door vertex selection (## 9.3): a red `door_arc` bbox whose long/short side ratio exceeds `2:1` is rejected outright (`RejectedEvidence(kind="unresolved_door_arc_aspect_ratio")`), before any purple/black/orange evidence is even looked at. A real 90-degree door arc's bbox is close to square (its two sides are both roughly the door's own radius), so a markedly elongated bbox (e.g. `1:3`) is not a plausible arc regardless of nearby evidence - this is a shape floor on the red cluster itself, the same class of check as the existing minimum-component-area floor, not an evidence-based rejection (must-rule 53; `doors.max_bbox_aspect_ratio` in config, default `2.0`).

Verified against all three required samples (`outputs/vectorization/v008/iteration5_run3`): every accepted red `door_arc` bbox across all 17 doors already has a ratio between `1.05:1` and `2.00:1` (the `2.00:1` case landing exactly on the inclusive boundary), so this floor changed no observable output for the required samples - it is a guard against future/other input, not a fix for anything currently failing. Test suite fixtures that previously used deliberately elongated rectangles to stand in for an "arc" (several at roughly `3.5:1`) were widened to a realistic, roughly-square shape; one test relying on extreme elongation to represent a "weak" cluster was changed to represent weakness via smaller area instead, since aspect ratio is now a separate, independent floor.

## 23. Task19 Debugging Notes

task19 made two changes to `build_debug_overlay` (## 15), both purely about what gets drawn - no change to detection, alignment, connection, or `metrics.json` content.

**Rejected/unresolved evidence is no longer drawn.** The gray bbox/centroid-box rendering loop over `rejected_evidence` was removed from the overlay entirely (along with its legend row and the now-unused `REJECTED_COLOR` constant) - it still flows into `metrics.json` exactly as before (`build_metrics`'s `rejected_evidence` parameter and the `rejected_by_kind` summary are untouched), since `rejected_evidence` is debug/metrics-only information per must-rule 107, and the overlay image is more readable without it. `build_debug_overlay` keeps the `rejected_evidence` parameter for call-site stability but no longer uses it (`del rejected_evidence` documents the intent inline).

**`wall_door_hinge_point`/`wall_door_end_point` are no longer re-drawn in the door candidate's confidence color.** Before this task, the door-candidate loop drew a second, larger (radius-5) circle around each hinge/end point in the candidate's confidence color (red for `door_confidence >= 0.75`, yellow otherwise) - since most candidates in the required samples are high-confidence, this larger red ring visually dominated the smaller radius-3 orange/purple `POINT_COLORS` ring drawn for every other final point type, making doors look like they had "red circles" instead of reading the same way every other point type does. Fixed by dropping that redraw - the bbox rectangle and hinge-to-end connector line still use the confidence color (so accepted-vs-low-confidence candidates remain visually distinguishable per must-rule 108/113), but the points themselves are only ever drawn once, in their own orange/purple color, by the existing per-point-type loop.

Verified by regenerating all three required samples (`outputs/vectorization/v008/iteration5_run3`) and visually inspecting `debug_overlay.png`: no gray rejected-evidence boxes remain anywhere, hinge/end markers read as small orange/purple circles across all 17 doors (including sample_005's 7 doors, the busiest case), and `metrics.json` content/validation issues are unchanged from before this task.

## 24. Completion Criteria

This spec is complete when v008 can consume `segformer_b0_run3` 7-class output and produce strict orthogonal architectural SVG geometry for walls, windows, and doors.

The implementation is not complete if it:

- depends on retired 5-class assumptions
- traces contours as final geometry
- exports diagonal or 45-degree wall geometry
- creates doors without red arc evidence
- creates final windows or doors that are not hosted on wall topology
- emits debug or unidentified visible groups in `vector.svg`

