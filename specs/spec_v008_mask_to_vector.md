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

Component primitive definitions are in `spec_v007_component_primitives.md`. This v008 spec controls reconstruction order, topology, validation, and implementation behavior under `src/vectorization`.

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

### 9.3 Door Points

Search door points from red `door_arc` components first.

Rules:

- each accepted red `door_arc` component requires one `wall_door_hinge_point`
- each accepted red `door_arc` component requires one `wall_door_end_point`
- red pixels define the existence and count of doors
- orange and purple pixels infer the hinge and endpoint locations
- if red pixels exist but purple door-origin evidence is missing or too noisy, infer the door-origin segment from the nearest host wall and the orange/red swing evidence
- if no red pixels exist, reject the door even when purple or orange evidence exists
- purple door-origin evidence without a red arc is debug-only

The hinge should prefer the orange/purple intersection when present. If that intersection is missing, infer the hinge from the red arc geometry and nearest plausible host wall.

### 9.3.1 Forceful Inference Rule (task13)

A connected red `door_arc` cluster is a door, always, once it survives component cleanup. Purple/orange/black evidence refines hinge/end geometry but never decides whether the red cluster is a door:

- `wall_door_hinge_point` and `wall_door_end_point` must each lie within `200 mm` of the red cluster's bounding box (the point may sit just outside the box, not only inside it). Scale must be resolved before this check runs.
- the hinge point is the location with the highest combined proximity to the available subset of `[red, orange, purple, black]` evidence; if all four are unavailable, fall back to the largest available subset (down to black/red alone via arc-geometry + host-wall inference).
- the end point is the location with the highest combined proximity to the available subset of `[red, purple, orange]`; if door-origin evidence cannot be paired, infer the end point from the red cluster's own bounding-box geometry projected onto the host wall.
- fragmented, missing, or noisy purple/orange evidence must lower the resulting `door_confidence` but must not delete the door.
- a red cluster may be rejected only when it is below the minimum component area, or when no plausible wall evidence exists anywhere near it (i.e. no plausible door geometry can be inferred even after fallback).

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

### 11.1 Axis Alignment

If two or more points have nearly equal `x` coordinates, they may share one vertical axis.

If two or more points have nearly equal `y` coordinates, they may share one horizontal axis.

The default tolerance is `500 mm` after scale resolution. If scale is unavailable, use a configured pixel fallback only for graph construction and mark metric alignment decisions as scale-blocked.

After alignment:

- points on the same vertical axis have exactly equal `x`
- points on the same horizontal axis have exactly equal `y`
- final edges are horizontal or vertical only

### 11.2 Alignment Evidence

An axis alignment must be supported by semantic evidence:

- wall pixels along the axis
- window pixels along the axis
- door-origin pixels along the axis
- endpoints from the same connected component
- endpoints from a known door or window pair

Do not align unrelated points only because their coordinates are close.

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
- rejected candidate evidence
- aligned axes
- wall/window/door-origin graph edges
- door hinge choices
- inferred door origins
- scale estimate and confidence
- validation failures
- every red `door_arc` candidate bounding box, its inferred `wall_door_hinge_point`/`wall_door_end_point`, and its supporting nearby red/orange/purple/black evidence, with low-confidence inferred points rendered differently from high-confidence ones (task13)

`metrics.json` must additionally include, under the scale diagnostics (task12 SS1):

```txt
red_arc_bbox_long_edges_px
red_arc_px_to_mm_candidates
red_arc_selected_modules_mm
selected_px_to_mm
scale_source
scale_rejected_outliers
```

and one record per accepted red `door_arc` cluster (task13 "Metrics Requirements"), so it is obvious whether the cluster became a door and how its hinge/end points were inferred:

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
```

Rejected evidence and low-confidence/forced-inference door candidates belong only in `debug_overlay.png` and `metrics.json`, never in `vector.svg`.

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

## 18. Task14 Debugging Notes

task14 debugged the v008 implementation against `specs/vectorization_must_rules.md` using the required test case (`outputs/vectorization/v008/iteration5_run3`). The must-rules and this spec were already correct; the following implementation bugs in `src/vectorization` caused observed output to violate them and were fixed:

- `primitives/scale.py`: scale resolution required the winning red-`door_arc` voting group to cover >= `min_scale_confidence_for_metric` of *all* clusters (including ordinary noise) before reporting `scale_status="estimated"` at all, collapsing scale to `"unknown"` whenever several clusters were noisy even though a clean winning group existed - contradicting SS5 priority item 5 ("unknown ... only if no usable red door_arc cluster exists"). `confidence` is now reporting metadata only; resolution itself only requires a usable winning group. The same over-gating was removed from `snap_to_module_mm` and from the window/door creation checks in `point_detection.py` - once scale is resolved/estimated, mm conversion and object creation proceed regardless of confidence.
- `point_detection.py` `_detect_door_points`: a red `door_arc` cluster paired with a fragmented/tiny `door_origin` component (implausibly narrow projected width) was rejected outright instead of falling back to arc-geometry inference, per SS9.3.1's forceful-inference rule. `RejectedEvidence` records for scale-blocked/too-narrow/non-cardinal-axis door rejections were also attributed to the wrong component (`door_origin`/`origin_id`, often `None`) instead of the originating red `door_arc` cluster, breaking the per-cluster metrics report (SS15).
- `point_detection.py` `_hinge_snap_to_wall`: picked the host wall by raw nearest-distance only, unreliable at corners/junctions; now reuses the same orientation/overlap/remainder-scored `select_host_wall_for_opening` window hosting already uses.
- `point_detection.py` `build_wall_skeleton_graph`: per-chain wall thickness used the *whole wall component's* overall `minAreaRect` short axis, which is meaningless (and can be hundreds of px) for a large non-rectangular component such as a full outer-wall loop - this wildly inflated the final wall polygon's buffer width. Replaced with a local thickness sampled from a distance transform along each chain's own pixels.
- `point_connection.py` `build_wall_edges`: only ever connected a wall edge between a skeleton chain's own pre-existing two endpoints, never splitting at a window/door point hosted mid-chain (the common case when the CNN wall mask has no real pixel gap at the opening) - leaving that point with no wall edge at all (SS3/SS14 "host every final window and door on wall topology"). Chains are now split at any window/door point that projects onto them. A real wall free end recognized as "near opening evidence" (SS9.1) but just outside the tight node-match tolerance is now also connected to that opening point via a dedicated, narrowly-scoped fallback.
- `point_connection.py` `validate_graph`: flagged the *correct* topology (a window/door point covered by one wall edge plus one opening edge) as `opening_point_multiple_edges`, and never checked for the real failure mode (zero wall edges = floating). Coverage is now counted per edge type; a new `floating_window_point`/`floating_door_point` issue catches genuine floating openings.
- `wall_geometry.py` `window_edges_to_primitives`: window thickness was half the host wall's own pixel thickness, which only equals SS14's required `100 mm` when the host wall happens to be the (no longer universal) `200 mm` module - now uses the fixed `100 mm` constant directly whenever scale is resolved.

Remaining, not-fully-resolved for the required test case (documented per task14 acceptance criterion 6, not further pursued to avoid pipeline redesign):

- A few door hinge/end points on the most heavily fragmented wall corners (several near-duplicate sub-pixel skeleton chains from `skeletonize` noise) still end up with no wall edge, because those chains' own endpoints don't resolve to any final point either - a deeper skeleton-graph fragmentation issue at corners, not a single rule violation.
- A small number of final wall polygon edges are within a few degrees of axis rather than exactly orthogonal, where a long wall run is split across more than one chained skeleton segment and `point_alignment.py`'s per-pair axis snapping doesn't compound transitively across the whole run.

## 19. Completion Criteria

This spec is complete when v008 can consume `segformer_b0_run3` 7-class output and produce strict orthogonal architectural SVG geometry for walls, windows, and doors.

The implementation is not complete if it:

- depends on retired 5-class assumptions
- traces contours as final geometry
- exports diagonal or 45-degree wall geometry
- creates doors without red arc evidence
- creates final windows or doors that are not hosted on wall topology
- emits debug or unidentified visible groups in `vector.svg`

