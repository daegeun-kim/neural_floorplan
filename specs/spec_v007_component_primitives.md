# Spec v007: Metric Component Primitives for Neural Floorplan

## 0. Purpose

This spec defines the reusable architectural primitives used after the active 7-class CNN segmentation model.

The primitive layer must convert segmentation pixels into clean, metric, CAD-like architectural components. It must not preserve noisy pixel contours as final geometry.

Pipeline position:

```txt
segformer_b0_run3 prediction
-> metric primitive definitions        <- this spec
-> strict mask-to-vector reconstruction <- spec v008
-> architectural SVG output
```

## 1. Scope

This spec covers:

```txt
primitive object definitions
metric unit model
scale resolution rules
standard architectural dimensions
allowed snapping and normalization
SVG drawing behavior
module organization under src/vectorization
```

This spec does not cover:

```txt
CNN training
semantic mask generation
candidate extraction from segmentation masks
full reconstruction order
room graph inference
JSON export
DXF export
Grasshopper integration
```

Candidate extraction and reconstruction order belong to `spec_v008_mask_to_vector.md`.

JSON export belongs to `spec_v009_cad_json.md` or a later v009 update after SVG vectorization is reliable.

## 2. Active CNN Classes

The active CNN is `segformer_b0_run3`.

It predicts exactly seven classes:

| ID | Class | Vectorization role |
|---:|---|---|
| 0 | background | ignored |
| 1 | floor | lower-confidence floor evidence |
| 2 | wall | high-confidence wall evidence |
| 3 | window | high-confidence window evidence |
| 4 | door_arc | door swing evidence |
| 5 | door_leaf | door panel evidence |
| 6 | door_origin | wall-aligned door threshold evidence |

Do not use the retired 5-class mapping in active primitive code:

```txt
background / wall / opening / room / icon
```

The retired mapping may remain only in files explicitly marked outdated.

## 3. Design Principle

The vector output must be architectural, not pixel-based.

The CNN mask is evidence. It is not the final geometry.

Required transformation:

```txt
segmentation pixels
-> robust evidence measurement
-> snapped metric primitive parameters
-> clean generated architecture
```

Not allowed as final behavior:

```txt
pixel contour tracing
jagged boundary preservation
free-form door/window polygons
floating openings not hosted by walls
pixel-unit output labeled as metric
```

## 4. Coordinate and Unit System

Preferred internal unit:

```txt
millimeter
```

Every primitive must carry scale metadata:

```txt
unit: "mm" or "px"
px_to_mm: float
scale_status: "resolved" | "estimated" | "unknown"
scale_source: string
confidence: 0.0 to 1.0
```

If scale cannot be resolved or estimated safely, export in pixel units and mark scale as unknown.

Do not silently label pixel units as metric.

## 5. Scale Resolution

A raster prediction alone does not guarantee absolute building scale. The scale resolver must use the strongest available evidence.

Scale priority:

```txt
1. explicit dataset or SVG scale metadata, if available
2. dimension annotations parsed from source SVG, if available
3. clustered door-origin widths from the prediction
4. clustered wall thicknesses from the prediction
5. fallback to pixel units
```

Door size is the first practical fallback because doors have relatively standard architectural dimensions. However, scale must not be estimated from one door alone.

Required first implementation:

1. Measure multiple confident `door_origin` segments in pixels.
2. Cluster similar door-origin widths.
3. Fit clusters to common door modules.
4. Cross-check the resulting `px_to_mm` against wall thickness clusters.
5. Accept the estimate only if door and wall evidence are plausible.
6. Record `scale_status="estimated"` unless explicit metadata confirms the scale.

Recommended common metric modules:

| Component | Common dimensions |
|---|---|
| interior wall thickness | 100 mm |
| exterior wall thickness | 200 mm |
| narrow door clear width | 600 mm |
| standard door clear width | 800 mm |
| wide door clear width | 900 mm |
| small window width | 600 mm |
| standard window width | 900 mm, 1200 mm, 1500 mm |

If door and wall estimates conflict strongly, keep pixel units and report a scale conflict in debug output.

## 6. Primitive Types

Required primitive types:

```txt
WallPrimitive
OuterWallLoopPrimitive
InnerWallSegmentPrimitive
WindowPrimitive
DoorOriginPrimitive
DoorLeafPrimitive
DoorArcPrimitive
FloorPrimitive
```

Do not use one generic `OpeningPrimitive` as the active primitive source of truth. The active CNN already separates windows and door components.

Generic opening objects may exist only as temporary debug objects.

## 7. Shared Primitive Fields

Every primitive must expose:

```txt
id
kind
source_class_ids
confidence
geometry
unit
px_to_mm
scale_status
source_evidence_bbox_px
source_evidence_area_px
```

Metric primitives must store final dimensions in millimeters.

Pixel-space source evidence must remain available for debugging.

## 8. Snapping Rules

All final architectural linework must snap to multiples of 45 degrees:

```txt
0 degrees
45 degrees
90 degrees
135 degrees
180 degrees
225 degrees
270 degrees
315 degrees
```

Do not export arbitrary-angle final wall, floor, window, or door-origin geometry.

Arbitrary-angle evidence may appear only in debug output.

Orthogonal directions (0/90/180/270) are the default interpretation, not an
equal alternative to 45-degree diagonals (task09). Priority:

```txt
1. horizontal / vertical, for any angle within a generous threshold of a cardinal
2. explicit 45-degree diagonal, only for angles close to exactly 45/135 degrees
3. otherwise default to the nearest cardinal, even if a diagonal is mathematically nearer
```

Do not convert noisy or ambiguous wall pixels into diagonal walls just
because they are closer to 45 degrees than to a cardinal - 45-degree output
is reserved for evidence that is explicitly, strongly diagonal.

## 9. Wall Primitives

Walls are represented internally by:

```txt
centerline polyline
thickness_mm or thickness_px
wall_type
```

The centerline is an internal construction helper only. Final rendered wall
geometry must be a closed filled polygon, built by offsetting the centerline
half the wall thickness on each side and connecting caps and corners, not a
single SVG line with `stroke-width`.

Before offsetting, wall centerline segments that share an endpoint must be
merged into connected polylines, preserving vertex order (task09). Buffering
each short segment independently and only unioning the results gives every
segment its own end cap, so a shared corner or junction shows up as two
overlapping flat-capped rectangles instead of one continuous body with a
real mitred corner. Merging first means a flat cap only lands where wall
evidence genuinely ends (a free end or a 3+-way junction stub, which cannot
be represented as a single connected polyline), and every other vertex gets
a clean mitred join. The outer wall loop and inner walls render as one
connected polygon system where their buffers touch or overlap, which also
naturally avoids drawing the outer and inner wall as two separate, visibly
duplicated lines.

Final wall fill color: black (`#000000`).

Allowed `wall_type` values:

```txt
outer
inner
unknown
```

Final wall thickness must normalize to common modules when scale is known:

```txt
100 mm
200 mm
```

### 9.1 OuterWallLoopPrimitive

The outer wall must be a closed polyline.

Requirements:

```txt
closed loop
continuous straight segments
45-degree snapping
no dangling endpoints
no tiny contour notches
dominant exterior shell of the building
```

The outer wall loop is generated before inner walls.

### 9.2 InnerWallSegmentPrimitive

Inner walls are individual line segments or snapped polylines.

Requirements:

```txt
not forced closed
45-degree snapping
connected to outer wall or other inner walls when evidence supports it
normalized 100 mm default thickness unless evidence supports 200 mm
```

Dangling inner-wall endpoints must be snapped onto a nearby wall line
(outer loop or another inner wall) when the gap is small enough to imply an
intended connection, rather than left as a visibly disconnected fragment.

## 10. WindowPrimitive

A window is a wall-hosted line segment that replaces part of a wall.

Required fields:

```txt
host_wall_id
start_point
end_point
width_mm or width_px
orientation
confidence
```

Requirements:

```txt
window endpoints must lie on wall topology
window must split the host wall at both endpoints
window segment replaces the wall segment between those endpoints
window must not float independently from a wall
window orientation follows the host wall
```

When scale confidence is sufficient, window length may snap to common window modules.

Final window geometry is a closed filled polygon (the host wall segment's
gap), not a stroked line. Fill color: blue (`#3c78dc`).

The window's own offset is independent of the host wall's thickness
(task09): 50 mm on each side, 100 mm total width - exactly half the wall's
200 mm total width, regardless of the wall's own measured thickness. In
pixel units (no resolved scale), derive the window's px thickness
proportionally from the wall's own measured px thickness (half of it),
rather than reusing the wall's thickness directly, so the 1:2 ratio holds
whether or not metric scale is known.

## 11. Door Primitives

Door output is composed of three related primitives:

```txt
DoorOriginPrimitive
DoorLeafPrimitive
DoorArcPrimitive
```

### 11.1 DoorOriginPrimitive

Door origin is the wall-hosted threshold segment.

Requirements:

```txt
generated from border evidence between wall pixels and door_origin pixels
hosted by a wall
splits the wall at both endpoints
replaces the wall segment between endpoints
normalized to common door widths when scale confidence is sufficient
```

Common door widths:

```txt
600 mm
800 mm
900 mm
```

Final door-origin geometry is a thin symbolic SVG line, not a closed filled
polygon (task09 supersedes task08's polygon decision for this primitive -
only wall and window are offset into polygons; door components stay
symbolic). Stroke color: purple (`#a046b4`).

### 11.2 DoorLeafPrimitive

Door leaf is generated from the door origin.

Requirements:

```txt
starts at one endpoint of the door origin
is perpendicular to the door origin
length equals the normalized door width
chooses swing side using door_leaf and door_arc evidence
does not trace noisy door_leaf pixels
```

Final door-leaf geometry is a thin symbolic SVG line, not a closed filled
polygon (task09 supersedes task08's polygon decision for this primitive).
Stroke color: orange (`#eb8c50`).

### 11.3 DoorArcPrimitive

Door arc is generated from the hinge point.

Requirements:

```txt
center is the hinge point where door origin and door leaf meet
radius equals the normalized door width
arc angle is 90 degrees
arc side follows door_arc evidence when available
```

The arc is generated geometry, not a traced CNN contour. Like door origin
and door leaf, the arc is a thin symbolic SVG primitive (a stroked,
unfilled path), not a closed polygon - only wall and window are offset
into polygons (task09). Stroke color: red (`#dc5a5a`).

The SVG elliptical-arc command admits two valid circle centers for a given
radius and pair of endpoints; the large-arc-flag and sweep-flag must be
computed from the actual hinge/origin-far/leaf-end angles (not guessed from
swing-side alone) so the rendered arc is always centered exactly on the
hinge point, for any wall orientation. Guessing the sweep flag from swing
direction alone is not sufficient and can render the arc mirrored to the
wrong side.

## 12. FloorPrimitive

Floor is a filled polygon.

The CNN floor class has lower accuracy than wall and opening-related classes, so floor geometry must be subordinate to wall topology.

Required evidence:

```txt
floor pixels
wall pixels
outer wall loop
```

Requirements:

```txt
45-degree snapped boundary
follows outer wall loop when floor pixels are ambiguous
does not create isolated floor islands from noisy floor pixels
fills the architectural interior region
sits behind wall, window, and door primitives
```

Floor is not a room-instance graph in this spec.

## 13. SVG Output Behavior

SVG is an output format, not the primitive definition.

The final SVG must contain only these four visible component groups, in
this order:

```txt
floor
wall
window
door
```

Drawing order:

```txt
floor behind everything
walls above floor
windows replace wall portions
door origins replace wall portions
door leaves and arcs above walls
```

No debug group, dashed unresolved-evidence marker, or other unidentified
element may appear in the final SVG. Debug visualization of unresolved or
unhosted evidence belongs exclusively in a separate debug raster
(`debug_overlay.png`) and in `metrics.json` counts - never inside
`vector.svg`.

Required SVG metadata:

```txt
data-unit
data-scale-status
data-px-to-mm
data-scale-source
```

## 14. Module Organization

Expected source organization:

```txt
src/vectorization/
  primitives/
    base.py
    scale.py
    wall.py
    window.py
    door.py
    floor.py
  geometry_rules.py
  export_svg.py
```

The implementation may keep compatible existing module names if tests are updated, but the primitive concepts in this spec must remain intact.

## 15. Validation Requirements

Primitive tests must verify:

1. Scale metadata is stored and exported.
2. Walls normalize to 100 mm or 200 mm when scale is known.
3. Door origins normalize to 600 mm, 800 mm, or 900 mm when scale is known.
4. Door leaf is perpendicular to door origin.
5. Door arc is centered on the hinge point and spans 90 degrees.
6. Windows are wall-hosted and replace a wall segment.
7. Outer wall loop is closed.
8. Inner walls are not forced closed.
9. Floor follows outer wall when floor evidence is ambiguous.
10. Arbitrary final angles are rejected.
11. Wall and window SVG output are closed filled polygons, not stroked
    lines with `stroke-width`; door origin, door leaf, and door arc are
    thin symbolic lines/arc (task09).
12. Door arc SVG output is centered on the hinge point and is not mirrored
    to the wrong side, for any wall orientation.
13. Final SVG colors match: floor white, wall black, window blue,
    door-origin purple, door-leaf orange, door-arc red.
14. No debug group or unresolved-evidence marker appears in the final SVG.
15. Wall centerline segments sharing an endpoint are merged into one
    connected polyline before offsetting, producing a clean mitred join
    rather than overlapping independently-capped rectangles.
16. Window total width is half the host wall's total width (100mm vs
    200mm), independent of the wall's own measured thickness value.
17. Ambiguous wall angles (not close to an exact cardinal or an exact
    45/135 diagonal) snap to the nearest cardinal, not the nearer diagonal.

## 16. Completion Criteria

This spec is satisfied when the primitive system can represent a clean architectural floorplan from 7-class CNN evidence without relying on the retired 5-class opening/room/icon assumptions.
