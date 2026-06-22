# Spec v008: Strict 7-Class Mask-to-Vector Reconstruction

## 0. Purpose

This spec defines the strict reconstruction process that converts `segformer_b0_run3` 7-class CNN predictions into architectural SVG vector output.

The vector output must be an architectural abstraction with metric dimensions whenever scale can be resolved or estimated safely. It must not be a pixel contour trace.

Pipeline:

```txt
7-class CNN prediction
-> class masks
-> scale resolution
-> wall topology
-> hosted windows and doors
-> floor reconstruction
-> SVG output
```

Component definitions are in `spec_v007_component_primitives.md`.

JSON output is intentionally out of scope for v008 and belongs to v009 after vectorization is reliable.

## 1. Scope

This spec covers:

```txt
reading 7-class prediction masks
decoding RGB preview masks when needed
resolving or estimating scale
building outer and inner wall vectors
splitting walls for windows and doors
generating door origin, leaf, and arc geometry
generating floor geometry
exporting SVG with scale metadata
```

This spec does not cover:

```txt
CNN retraining
semantic mask generation from SVG
editing checkpoint files
room graph construction
JSON export
DXF export
Grasshopper integration
interactive correction UI
```

## 2. Input Classes

The only active input class mapping is:

| ID | Class | Meaning |
|---:|---|---|
| 0 | background | non-floorplan pixels |
| 1 | floor | floor/interior evidence, lower confidence |
| 2 | wall | wall evidence, high confidence |
| 3 | window | window evidence, high confidence |
| 4 | door_arc | door swing arc evidence |
| 5 | door_leaf | door panel evidence |
| 6 | door_origin | wall-aligned door threshold evidence |

Do not use the retired 5-class mapping in active v008 code:

```txt
0 background
1 wall
2 opening
3 room
4 icon
```

If a 5-class mask is passed to this pipeline, raise a clear incompatible-input error.

## 3. Input Files

The vectorizer should support:

```txt
class-ID PNG masks
RGB prediction preview PNGs using the run3 palette
```

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

If RGB values do not match the configured palette, fail clearly unless an explicit tolerance is configured.

## 4. Core Design Rules

The reconstruction must be strict.

Required behavior:

```txt
use wall/window/door evidence as high-confidence topology
treat floor evidence as lower-confidence fill evidence
snap final vectors to multiples of 45 degrees
normalize components to common metric dimensions when scale is known
host all windows and doors on walls
split wall segments at all hosted openings
preserve topology over pixel contour detail
```

Forbidden final behavior:

```txt
free-floating doors
free-floating windows
door arcs traced as irregular polygons
wall contours exported as jagged polygons
floor islands from noisy floor pixels
arbitrary-angle final wall fragments
pixel-unit export labeled as metric
arbitrary pixel-geometry fallback for an explicitly millimeter-scale rule
  (inner-wall outer-loop attachment, window minimum width, door module
  width) when scale cannot be resolved or estimated (task10) - the sample
  must be recorded as scale-blocked for that rule instead
```

## 5. Scale Resolution

Before metric export, resolve scale using the rules from spec v007.

Priority:

```txt
1. explicit metadata
2. SVG or dimension metadata
3. clustered door-origin widths
4. clustered wall thicknesses
5. fallback to pixel units
```

Door width is the first practical fallback because standard doors are common and visible in the new 7-class output. However, use multiple confident doors and cross-check against wall thickness.

Recommended algorithm:

1. Measure candidate door-origin lengths in pixels.
2. Cluster lengths by similar pixel size.
3. Fit clusters to common door modules: `700 mm`, `900 mm` (task10: `600 mm`
   and `800 mm` are not valid door modules - scale voting and the final
   door-width snap, SS9.1, both use the same two-module set).
4. Measure wall thickness clusters in pixels.
5. Fit wall clusters to `100 mm` and `200 mm`.
6. Choose the scale with the best combined door and wall consistency.
7. Mark scale as `estimated` unless explicit metadata confirms it.

If door and wall estimates conflict strongly, keep pixel units and report scale conflict.

Door-origin lengths used for this clustering are measured directly from
the cleaned `door_origin` mask's connected components (long axis per
component) - scale must be resolved before doors are hosted, since several
downstream rules (inner-wall outer-loop attachment, window minimum width,
door module snap) require a resolved/estimated scale to run at all.

## 6. Reconstruction Order

The vectorizer must build geometry in this order:

```txt
0. decode and clean masks
1. build wall topology (outer loop, then inner walls, pixel-space)
2. resolve scale (door-origin lengths measured directly from the mask,
   wall thickness samples) - task10: moved here, before openings, because
   several opening/wall rules below are millimeter-only
3. attach inner-wall endpoints to the outer loop when within tolerance (mm)
4. generate windows
5. generate doors
6. snap walls to 45 degrees, re-project hosted openings
7. split walls at hosted windows/door-origins
8. generate floor
9. export SVG
```

Wall topology must be built before openings.

Openings must modify wall topology.

Floor must be generated after walls because wall topology is more reliable than floor pixels.

## 7. Wall Building

Wall vectors are built from:

```txt
wall pixels
window pixels
door_origin pixels
door_leaf pixels
door_arc pixels
```

Wall pixels are primary wall evidence. Window and door pixels are opening evidence that indicate interruptions in walls.

### 7.1 Outer Wall

Build the outer wall first.

Requirements:

```txt
closed curve
continuous straight-line segments
only multiples of 45 degrees
derived from exterior wall evidence
robust to window and door gaps
no dangling endpoints
no tiny contour notches
```

Use opening evidence (window, door_arc, door_leaf, door_origin masks) to
bridge across wall gaps while preserving the opening location for later
splitting.

The outer wall must never be derived from the floor/background border.
Floor is the lowest-accuracy class (SS10), so it must not be unioned into
the evidence used to trace the envelope contour - doing so lets the
envelope silently follow wherever floor evidence happens to end rather
than the actual wall pixels, and can also cause the wall band near the
envelope to be detected a second time as a spurious "duplicate" inner wall
just inside the outer loop.

The outer wall represents the building envelope, not a raw contour.

### 7.2 Inner Walls

Build inner walls after the outer wall.

Requirements:

```txt
individual line segments or snapped polylines
not forced closed
only multiples of 45 degrees
connected to other walls where evidence supports connection
trimmed or split by openings
```

Inner wall extraction should use skeleton or centerline-style evidence
rather than filled polygon contours for the underlying topology, even
though the final rendered output is a filled polygon (SS12, spec v007 SS9).

A dangling inner-wall endpoint that lies within a small tolerance of
another wall's line (outer loop or another inner wall) must be snapped or
extended onto it rather than left disconnected, when the evidence implies
a real connection.

Once the outer wall is built, only the spatial band traced along the
outer polygon is erased before inner-wall extraction runs (task10) - not
the whole connected wall component the band touches. Interior partition
walls frequently fuse to the exterior wall in one continuous CNN-predicted
blob; erasing the entire touched component (the task09 behavior) destroys
that interior wall's only evidence along with the outer wall's. Erasing
only the band itself keeps the outer wall from resurfacing as a duplicate
inner wall while preserving every interior wall branch, including ones
that touch the exterior wall.

The inner-wall candidate mask is the union of `wall` pixels and
`door_origin` (purple) pixels - not `door_arc`/`door_leaf`/`window`. This
bridges the gap a doorway leaves in an interior wall's mask, the same way
opening evidence already bridges gaps for the outer loop (SS7.1), so an
inner wall is not falsely cut short or dropped at a doorway. The resulting
segment that "tunnels" through a door is trimmed back to the real wall
span once the door is hosted on it, via the same wall-splitting step used
for any other opening (SS9.1).

If one or both endpoints of an inner-wall segment fall within
`walls.inner_attach_outer_threshold_mm` (default 500mm, real architectural
distance using the resolved/estimated scale - no pixel fallback) of the
outer wall loop, project that endpoint onto the nearest outer wall segment
and snap it there. This never moves the outer wall loop itself. If scale
cannot be resolved or estimated at all, this rule does not run for that
sample and the sample is recorded as scale-blocked (`inner_wall_outer_attach_mm`)
rather than substituting an arbitrary pixel threshold.

An inner wall ending at a door or window opening boundary instead of
another wall is a normal, valid ending (most interior walls sit a short
distance from an opening) - it is not unresolved or incomplete, and does
not need the outer-loop attachment rule applied to that end. Walls remain
strongly biased toward 90-degree relationships; short freestanding stubs,
islands/partial partitions, and explicit diagonal walls are allowed
exceptions when the source evidence clearly supports them.

Wall centerline segments that share an endpoint must be merged into one
connected polyline before they are offset into a polygon, so a corner or
junction gets a single clean mitred join instead of two independently
flat-capped rectangles overlapping (task09, spec v007 SS9). A segment is
only left with its own flat end cap where it genuinely ends - a free end,
or one of three-or-more segments meeting at a junction (which cannot be
represented as a single connected polyline).

### 7.3 Wall Thickness

Classify wall thickness per segment.

Required metric modules:

```txt
100 mm
200 mm
```

When scale is known:

```txt
measured segment thickness -> nearest module
```

When scale is unknown:

```txt
keep measured pixel thickness
data-scale-status="unknown"
```

## 8. Window Generation

Window pixels are high-confidence opening evidence.

For each window candidate:

1. Find the nearest host wall centerline.
2. Project window evidence onto the host wall.
3. Locate the two transition points where wall evidence changes to window evidence.
4. Split the host wall at those two points.
5. Remove or suppress the wall segment between the two split points.
6. Insert a `WindowPrimitive` between the same endpoints.

Hard requirements:

```txt
window endpoints must connect to wall segment endpoints
window segment replaces a wall segment
window cannot float away from the wall
window orientation follows the host wall
window length may snap to common window modules when scale is reliable
```

If a window cannot be hosted on a wall, export it only as debug evidence, not as a final window.

The window's offset width is independent of the host wall's thickness
(task09): 50mm each side, 100mm total - exactly half the wall's 200mm
total, regardless of that particular wall's own measured thickness.

A window's hosted width must be at least `windows.min_width_mm` (default
300mm), using the resolved/estimated architectural scale - no pixel-only
fallback. If scale cannot be resolved or estimated, the window is recorded
as scale-blocked rather than silently accepted at an arbitrary pixel
width (task10).

### 8.1 Opening-Near-Corner Host Selection

A window or door opening is always hosted on exactly one wall - never
split or straddled across two walls, and never reduced to a degenerate
near-zero-length wall stub on either side (task10). When an opening's
evidence sits near where two wall segments meet (e.g. a corner or
T-junction), and the obvious host wall would leave a near-zero-length
remainder on one side after splitting, evaluate both candidate host walls
and push the opening fully onto whichever one has the higher hosting
probability - more overlapping/aligned evidence, better orientation match,
and a larger non-degenerate remainder after the split. This rule applies
to both windows and the door-origin hosting step (SS9.1).

## 9. Door Generation

Door generation uses three CNN classes:

```txt
door_origin (purple)
door_leaf (orange)
door_arc (red)
```

Door count and location are driven solely by red `door_arc` connected
components (task10) - door_origin and door_leaf are never sufficient on
their own to create a door. If no `door_arc` evidence exists at all, no
door is generated, regardless of how much door_origin/door_leaf evidence
exists nearby.

### 9.1 Door Origin and Hinge Detection

Per red `door_arc` connected component:

1. Skip components below `doors.min_door_arc_component_area`.
2. Look for the orange (`door_leaf`)/purple (`door_origin`) intersection
   near this arc group (within `doors.hinge_intersection_tolerance_px`) -
   this is the preferred hinge candidate.
3. If that intersection is missing and `doors.hinge_arc_inference_enabled`
   is true, find a provisional host wall for the arc's own evidence
   (SS8.1 corner-safe selection) and infer the hinge from the arc
   evidence's oriented bounding box corner closest to that wall - the
   swing wedge's pivot corner. If no provisional host wall is reachable,
   the arc group is debug-only (`unresolved_door_arc`).
4. Snap the hinge candidate onto the nearest wall, outer or inner, within
   `doors.hinge_snap_to_wall_max_dist_px`. No reachable wall -> debug-only
   (`unresolved_door_hinge`).
5. Find the door_origin (purple) evidence paired with this hinge and
   project it onto the host wall to get the far endpoint. Orange hinge
   evidence without a paired purple far point is debug-only
   (`unresolved_door_hinge`) - it never becomes a door (Door Pairing Rule,
   SS9.1.1).
6. If scale cannot be resolved/estimated with sufficient confidence, the
   evidence is debug-only (`unresolved_door_scale_blocked`) - never
   generate a pixel-sized door.
7. Snap the hinge-to-far-point distance to the nearest of
   `doors.door_width_modules_mm` (`700 mm` or `900 mm` only - `600 mm` and
   `800 mm` are not valid door modules for this pipeline), preserving the
   hinge position and detected orientation while rescaling the far point.
8. Replace the wall segment between the (now module-snapped) origin
   endpoints with a `DoorOriginPrimitive`.

Hard requirements:

```txt
door origin endpoints must connect to wall endpoints
door origin replaces a wall segment
door origin orientation follows the host wall
door origin cannot float away from a wall
```

#### 9.1.1 Door Pairing Rule

The orange hinge marker and the purple door-origin far/end marker are a
required pair. If both are detected, exactly one door is generated from
that pair. An unpaired orange marker, or an unpaired purple marker (purple
door_origin evidence with no matching red arc group), never generates a
door on its own - it appears only in `debug_overlay.png` and `metrics.json`,
never in `vector.svg`.

Door origin (and door leaf and door arc, SS9.2/9.3) render as thin
symbolic SVG lines/arc, not closed filled polygons - only wall and window
are offset into polygons (task09 supersedes task08's polygon decision for
the door primitives).

### 9.2 Door Leaf

Once the door origin is created, generate the leaf procedurally.

Requirements:

```txt
leaf starts at one endpoint of the door origin
leaf is 90 degrees to the door origin
leaf length equals the normalized door width
hinge endpoint is selected using door_leaf and door_arc evidence
swing side is selected using door_leaf and door_arc evidence
```

Do not trace the raw door_leaf pixel contour as final geometry.

### 9.3 Door Arc

Generate a 90-degree arc.

Requirements:

```txt
arc center is the hinge point where door origin and door leaf meet
arc radius equals the normalized door width
arc angle is 90 degrees
arc side follows door_arc evidence
```

Do not export irregular arc contours.

If door_arc evidence is missing but door_origin and door_leaf are confident, generate the best 90-degree arc and mark confidence lower.

## 10. Floor Generation

Floor generation happens after wall and opening topology.

Inputs:

```txt
floor pixels
wall pixels
outer wall loop
```

The CNN floor class has lower accuracy than wall, window, and door classes. Wall topology has priority over floor pixels.

Procedure:

1. Start from the outer wall loop interior.
2. Use floor pixels to confirm interior fill.
3. Use wall pixels to constrain the boundary.
4. If floor pixels are ambiguous or missing, follow the outer wall loop.
5. Snap the floor boundary to multiples of 45 degrees.
6. Remove tiny islands and holes unless they correspond to wall topology.

Hard requirements:

```txt
floor must be one main architectural filled region unless evidence strongly supports multiple buildings
floor boundary follows the outer wall when floor evidence conflicts
floor sits behind walls, windows, and doors
floor must not create jagged contour noise
```

## 11. Topology Rules

Final topology must satisfy:

```txt
outer wall is closed
inner walls are connected or intentionally free-ended
inner wall branches that touch the outer wall in the source mask are
  preserved, not erased along with the outer wall's evidence (task10)
inner wall endpoints within inner_attach_outer_threshold_mm of the outer
  loop are attached to it; an ending at an opening boundary is also valid
all final windows are hosted by walls
all final doors are hosted by walls
every final window meets the configured minimum width (task10)
every final door's count/location is driven by a red door_arc group, and
  its width is exactly one of doors.door_width_modules_mm (task10)
window and door origin segments replace wall segments
door leaf and arc attach to a door origin endpoint
floor follows or is bounded by outer wall topology
```

Topology violations should be visible in debug output and counted in metrics.

## 12. Output SVG

Output folder:

```txt
outputs/vectorization/v008
```

Required files per sample:

```txt
input.png
prediction.png
vector.svg
metrics.json
debug_overlay.png
```

The final SVG must contain only these four groups, in this order:

```txt
floor
wall
window
door
```

No debug group, dashed unresolved-evidence marker, or other unidentified
element is allowed in `vector.svg` (spec v007 SS13). Debug visualization of
unresolved/unhosted evidence lives only in `debug_overlay.png` and the
counts in `metrics.json`.

Drawing order:

```txt
floor
wall
window
door
```

Required final SVG colors (spec v007 SS9-11):

```txt
floor       white   #ffffff
wall        black   #000000
window      blue    #3c78dc
door_origin purple  #a046b4
door_leaf   orange  #eb8c50
door_arc    red     #dc5a5a
```

Wall and window render as closed filled polygons, not stroked lines with
`stroke-width`. Door-origin, door-leaf, and door-arc are thin symbolic
SVG lines/arc (task09); door-arc's center is analytically fixed to the
hinge point for any wall orientation (spec v007 SS11.3).

Required root metadata:

```txt
data-unit="mm" or "px"
data-scale-status="resolved" | "estimated" | "unknown"
data-px-to-mm="..."
data-scale-source="..."
```

## 13. Debug Output

Debug output must show:

```txt
decoded class masks
wall centerlines
outer wall loop
inner wall candidates
window host assignments
door origin host assignments
door hinge choices
floor fill source
scale estimate and confidence
topology errors
```

Failed hosted openings must appear in debug output but not in final `window` or `door` groups, and never inside `vector.svg` at all - only in `debug_overlay.png` and `metrics.json` (SS12).

## 14. Configuration

Expected config file:

```txt
configs/vectorization_v008.yaml
```

Required config sections:

```yaml
input:
  prediction_path:
  palette: run3

scale:
  allow_estimated_scale: true
  door_width_modules_mm: [700, 900]
  wall_thickness_modules_mm: [100, 200]
  min_scale_confidence_for_metric: 0.70

snapping:
  allowed_angles_deg: [0, 45, 90, 135, 180, 225, 270, 315]
  max_angle_snap_error_deg: 12
  min_segment_length_mm: 100

walls:
  build_order: ["outer", "inner"]
  bridge_opening_gaps: true
  connect_gap_px: 20
  ortho_snap_degrees: 20
  diagonal_snap_degrees: 10
  inner_attach_outer_threshold_mm: 500

openings:
  require_host_wall: true
  replace_wall_segment: true
  corner_ambiguity_px: 25
  min_remainder_px: 3

windows:
  min_width_mm: 300

doors:
  require_arc_group: true
  min_door_arc_component_area: 4
  hinge_intersection_tolerance_px: 6
  hinge_snap_to_wall_max_dist_px: 40
  hinge_arc_inference_enabled: true
  door_width_modules_mm: [700, 900]

floor:
  prefer_outer_wall_when_ambiguous: true
```

## 15. Validation Requirements

Tests must cover:

1. 7-class mask decoding.
2. Rejection of retired 5-class masks.
3. Scale estimation from multiple door widths.
4. Scale cross-check from wall thickness.
5. Closed snapped outer wall loop.
6. Inner walls are not forced into closed loops.
7. Windows split and replace wall segments.
8. Door origins split and replace wall segments.
9. Door leaf is perpendicular to origin.
10. Door arc is a 90-degree generated arc.
11. Floor follows outer wall when floor pixels are noisy or incomplete.
12. SVG metadata records unit and scale status.
13. Debug output records topology failures.
14. Outer wall is not derived from floor/background evidence.
15. Outer wall is not duplicated by inner wall extraction.
16. Dangling inner walls connect to the outer loop or other inner walls
    when evidence implies a connection.
17. Wall and window SVG output are closed filled polygons, not
    stroke-thickness lines; door-origin/door-leaf/door-arc are thin
    symbolic lines/arc, with the arc centered on the hinge point and red.
18. `vector.svg` contains no debug group, dashed marker, or retired-class
    (`room`/`icon`/generic `opening`) group.
19. Wall centerline segments sharing an endpoint produce one connected
    polygon body with a clean mitred join, not duplicated/overlapping caps.
20. Only the outer wall's synthetic band (not whole connected wall
    components) is removed before inner-wall extraction, so a connected
    inner-wall branch that touches the outer wall in the source mask
    survives (task10), while the outer wall still does not resurface as a
    duplicate inner wall.
21. Window total width is half the host wall's total width.
22. Ambiguous wall angles default to the nearest cardinal rather than a
    mathematically-nearer but not-explicit diagonal.
23. Inner walls are extracted when an outer loop exists, using a candidate
    mask that includes door_origin (purple) pixels so a doorway gap does
    not break the wall, then trimmed at the door once it is hosted.
24. An inner wall endpoint within `inner_attach_outer_threshold_mm` snaps
    to the outer wall loop; beyond the threshold it is left unchanged.
25. Window and door-origin endpoints coincide exactly with the adjacent
    wall segment endpoints after splitting (no gap, no overlap).
26. Door count and location are determined solely by red door_arc
    connected components; door_origin evidence without a matching arc
    group never creates a door, and no door_arc evidence means no door.
27. Door hinge prefers the orange/purple intersection when present, falls
    back to red-arc-geometry inference plus nearest-wall snapping when
    that intersection is missing, and always ends up snapped to a wall.
28. Orange/purple door markers are a required pair; unpaired evidence
    never becomes a final door and stays debug-only.
29. Door origin/leaf length snaps to exactly 700mm or 900mm when scale is
    resolved/estimated; 600mm and 800mm are never produced.
30. If scale cannot be resolved or estimated, mm-gated rules (inner-wall
    outer-attachment, window minimum width, door module snap) record the
    affected sample as scale-blocked instead of using an arbitrary pixel
    fallback.

## 16. Completion Criteria

This spec is complete when v008 can consume `segformer_b0_run3` 7-class output and produce strict architectural SVG geometry with metric dimensions whenever scale can be resolved or estimated safely.

The implementation is not complete if it still depends on the retired 5-class `opening` / `room` / `icon` assumptions.
