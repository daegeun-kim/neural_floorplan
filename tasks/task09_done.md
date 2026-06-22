# Task 09 - Fix Wall Connectivity and Inner Wall Duplication

## Context

The current vectorization output still has major wall-topology problems.

The outer wall generation is mostly useful and should not be replaced wholesale. The current issue is mainly that inner-wall extraction tries to vectorize pixels that were already handled by the outer-wall stage, causing duplicate wall geometry.

There is also a connectivity problem: wall centerlines are treated as many individual line segments instead of connected curves. When each short segment is offset separately, corners and junctions become visually incorrect.

This task is a wall-topology correction task. It should refine the existing vectorization pipeline rather than redesign the whole outer-wall recognition strategy.

## Objective

Fix vectorization so the wall system is built as connected architectural curves before polygon generation.

The final output should have:

```txt
connected outer wall curves
inner walls that do not duplicate outer walls
inner walls connected to outer walls where expected
clean wall corners and junctions
only intended visible components
```

## Relationship to Task 08

This task follows `task08_done.md` and supersedes the conflicting parts about door geometry thickness.

Task 08 required door origin and door leaf as closed polygons. That is no longer desired.

For this task:

```txt
wall geometry is polygonal
window geometry is polygonal
door origin is a thin single SVG line
door leaf is a thin single SVG line
door arc is a thin single SVG arc
```

## Component Geometry Rules

Final visible components must use these geometry rules:

| Component | Geometry | Color |
|---|---|---|
| floor | filled polygon | white |
| wall | closed polygon generated from connected wall curves | black |
| window | closed polygon generated from hosted window segment | blue |
| door origin | thin single SVG line | purple |
| door leaf | thin single SVG line | orange |
| door arc | thin single SVG 90-degree arc | red |

No other visible final components are allowed.

## Wall and Window Offset Rules

Only wall and window centerlines should be converted into closed polygons.

Wall offset:

```txt
offset 100 mm left
offset 100 mm right
total wall polygon width = 200 mm
```

Window offset:

```txt
offset 50 mm left
offset 50 mm right
total window polygon width = 100 mm
```

Door geometry must not be offset into polygons.

Door origin, door leaf, and door arc should remain thin symbolic SVG elements.

## Connected Curve Requirement

Wall centerlines must be connected before offsetting.

Current problem:

```txt
straight wall segments are stored as individual line segments
segments that share a vertex are offset separately
shared corners get duplicate caps, overlaps, gaps, or strange corner shapes
```

Required behavior:

```txt
find line segments with shared endpoints
merge them into connected polylines or curves
preserve the vertex sequence
offset the connected curve as one wall body
generate clean polygon joins at shared vertices
avoid internal end caps at connected vertices
```

If two line segments share a point, they should be treated as one connected curve unless there is an explicit reason to keep them separate.

## Outer Wall Definition

Outer wall means:

```txt
any wall that separates indoor space from outdoor/background space
```

Inner wall means:

```txt
any wall that separates indoor space from indoor space
```

Do not reinterpret outer wall as merely the bounding rectangle, and do not reinterpret inner wall as every wall pixel inside a contour.

## Keep Current Outer Wall Strategy

The current vectorization source is reasonably good at recognizing and creating the outer wall.

Do not replace the outer wall generation strategy unless a small targeted change is necessary.

Required outer wall changes:

```txt
connect outer wall line segments into continuous curves
offset connected outer wall curves cleanly
preserve the existing ability to identify the exterior wall envelope
```

Do not use this task to redesign outer wall extraction from scratch.

## Prevent Outer Wall Duplication as Inner Wall

The main inner-wall problem is duplicate conversion.

Current problem:

```txt
outer wall pixels are used to generate the outer wall
the same or overlapping pixels are then processed again as inner walls
the final drawing shows exterior wall geometry twice
```

Required behavior:

```txt
generate outer walls first
record the outer-wall evidence footprint
remove or mask outer-wall evidence before inner-wall extraction
do not allow inner-wall extraction to reuse already-consumed outer-wall pixels
```

Once pixels or segments have been claimed by the outer wall stage, they should not produce inner wall geometry.

## Inner Wall Connectivity Requirement

The vast majority of inner walls should connect to the wall system.

Most inner wall endpoints should terminate at:

```txt
an outer wall
another inner wall
a valid hosted opening condition
```

Do not leave inner walls as floating fragments when the endpoint is near an outer wall or another wall.

Required behavior:

```txt
snap nearby inner-wall endpoints to outer wall curves
extend short endpoint gaps when evidence implies a connection
merge collinear or nearly collinear fragments
create clean T-junctions and L-junctions
preserve free endpoints only when the source evidence clearly stops there
```

Wall topology should be strongly biased toward connection.

## Angle Snapping Priority

Allowed final wall angles remain multiples of 45 degrees:

```txt
0
45
90
135
180
225
270
315
```

However, ambiguous wall pixels should snap to orthogonal geometry first.

Priority:

```txt
1. horizontal / vertical
2. explicit 45-degree diagonal
3. reject or debug ambiguous arbitrary angle
```

Use 45-degree output only when 45-degree evidence is extremely explicit.

Do not convert noisy or ambiguous pixels into diagonal walls when an orthogonal interpretation is plausible.

## Door Rules

Door components should remain symbolic, thin SVG elements.

Requirements:

```txt
door origin is a purple thin single SVG line
door leaf is an orange thin single SVG line
door arc is a red thin single SVG 90-degree arc
door origin is hosted on wall topology
door leaf is perpendicular to door origin
door arc center is the hinge point where door origin and opened leaf intersect
door swing direction must not be reversed
```

Door origin, leaf, and arc should use wall topology for placement but must not be converted into thick wall-like polygons.

## Window Rules

Windows are wall-hosted and polygonal.

Requirements:

```txt
window is hosted by a wall
window replaces the wall portion at that location
window is offset 50 mm left and 50 mm right
window output is a blue 100 mm total-width closed polygon
window endpoints connect cleanly to adjacent wall polygons
```

## Floor Rules

Floor remains lower priority than wall topology.

Floor should:

```txt
fill the inside of the wall-derived envelope
stay behind all other components
not define or duplicate wall geometry
not create visible border artifacts
```

## Required Source Changes

Update vectorization code under:

```txt
src/vectorization
src/vectorization/primitives
configs/vectorization_v008.yaml
tests
```

Expected implementation changes:

```txt
merge shared-endpoint wall line segments into connected curves
offset connected wall curves as one polygon body
create clean joins at connected wall vertices
preserve current outer wall recognition while improving curve connectivity
mask or remove outer-wall evidence before inner-wall extraction
connect inner-wall endpoints to nearby outer walls and inner walls
strongly prefer orthogonal snapping for ambiguous wall evidence
reserve 45-degree snapping for explicit diagonal evidence
offset only walls and windows into polygons
render door origin, door leaf, and door arc as thin symbolic SVG elements
```

## Required Tests

Add or update tests to verify:

1. Shared-endpoint wall segments are merged into one connected curve before offsetting.
2. Connected wall curves produce clean polygon joins without duplicate internal caps.
3. Current outer wall recognition remains active.
4. Outer-wall evidence is removed or masked before inner-wall extraction.
5. Inner-wall extraction does not duplicate outer-wall geometry.
6. Inner wall endpoints snap or extend to nearby outer walls when evidence supports connection.
7. Inner wall endpoints snap or merge to nearby inner walls at plausible junctions.
8. Ambiguous walls snap to horizontal or vertical before 45 degrees.
9. 45-degree wall output appears only for explicit diagonal evidence.
10. Wall polygons are black and 200 mm total width when scale is known.
11. Window polygons are blue and 100 mm total width when scale is known.
12. Door origin remains a purple thin single SVG line.
13. Door leaf remains an orange thin single SVG line.
14. Door arc remains a red thin single SVG 90-degree arc.
15. Final SVG contains no duplicated exterior wall geometry.
16. Final SVG contains no extra visible debug or unidentified components.

## Acceptance Criteria

This task is complete when:

1. Outer walls remain correctly recognized and are connected into clean curves.
2. Inner walls do not reuse or duplicate outer-wall evidence.
3. Wall segments that share vertices become connected curves before polygon generation.
4. Wall polygon corners and junctions are clean.
5. Most inner wall endpoints connect to outer walls or other walls where expected.
6. Ambiguous wall geometry is orthogonal unless 45-degree evidence is explicit.
7. Only walls and windows are offset into polygons.
8. Door origin, door leaf, and door arc are thin symbolic SVG elements.
9. The final vector output no longer has duplicated exterior walls or disconnected wall fragments.
