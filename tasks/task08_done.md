# Task 08 - Rebuild Vectorization Output as Strict Architectural Polygons

## Context

The current vectorization output is very different from the intended architectural drawing.

Observed problems include:

```txt
unidentified orange dotted lines
wall lines drawn in multiple shades
outer wall lines generated from the floor/background border
outer wall geometry duplicated by inner wall extraction
disconnected inner wall segments
door swing arcs reversed
line-based walls with stroke thickness instead of actual wall polygons
extra visual components that are not part of the desired drawing
```

The active CNN is `segformer_b0_run3` with seven classes:

```txt
0 - background
1 - floor
2 - wall
3 - window
4 - door_arc
5 - door_leaf
6 - door_origin
```

The CNN has high wall and opening accuracy, but low floor accuracy. Therefore, vectorization must prioritize wall, window, and door evidence over floor/background borders.

This task is a full vectorization modification task. Do not treat it as a small styling change.

## Objective

Rebuild the vectorization pipeline so the final SVG is a strict architectural drawing made only from the intended components:

```txt
floor
wall
window
door
```

The output should be clean, readable, and architectural. It should not contain unidentified debug-like elements or pixel-border artifacts.

## Final Output Components and Colors

The final `vector.svg` must contain only these visible component types:

| Component | Geometry | Color |
|---|---|---|
| floor | filled polygon | white |
| wall | closed polygons | black |
| window | closed polygons | blue |
| door origin | closed polygon | purple |
| door leaf | closed polygon | orange |
| door arc | arc or closed arc primitive | red |

No other visible components are allowed in the final SVG.

Forbidden in final SVG:

```txt
dotted orange lines
multiple wall shades
grey wall lines
debug centerlines
unidentified primitive groups
line-stroke thickness used as wall geometry
room/icon/opening groups from the retired 5-class pipeline
```

Debug output may be added later, but this task should focus on making the final vectorization process correct first.

## Hard Rule: Wall Construction Comes First

Wall construction must be the first geometric vectorization step.

The wall system must be generated from the high-confidence structural/opening evidence:

```txt
wall pixels
window pixels
door_origin pixels
door_leaf pixels
door_arc pixels
```

Do not build outer walls from the pixel border between floor and background.

Do not use the white/grey floor-background boundary as the source of wall lines.

The floor class may help fill floor area later, but it must not define the outer wall linework.

## Wall Evidence Rule

Wall linework should follow the structural evidence visible in:

```txt
black wall pixels
blue window pixels
purple door_origin pixels
orange door_leaf pixels
red door_arc pixels
```

The goal is to create a clear wall topology first, then place hosted openings into that topology.

Door leaf and door arc evidence may be used to help identify door locations and swing direction, but they must not create arbitrary walls.

## Outer Wall Requirements

The outer wall must be created from wall/opening evidence, not floor-background borders.

Requirements:

```txt
closed curve
continuous straight segments
multiples of 45 degrees only
black wall polygon output
no duplicated outer/inner wall drawing
no visible floor/background border tracing
```

Opening gaps should be bridged during wall topology construction, while preserving opening locations for later replacement by window or door components.

## Inner Wall Requirements

Inner walls must be connected to the outer wall or to other inner walls when the evidence supports connection.

Requirements:

```txt
not isolated unless clearly free-standing in the source
multiples of 45 degrees only
connected to outer walls where wall evidence reaches the envelope
black wall polygon output
same 200 mm total wall width rule as outer walls
```

Disconnected inner wall fragments should be extended, snapped, or merged to nearby wall topology when the evidence implies a connection.

## Wall Polygon Requirement

Walls must not be drawn as single SVG lines with stroke thickness.

Every wall must be an actual closed polygon.

Use the wall centerline only as an internal construction helper.

Final wall geometry:

```txt
centerline
-> offset 100 mm left
-> offset 100 mm right
-> connect caps and corners
-> closed 200 mm total width polygon
```

This 200 mm total wall width applies to both outer walls and inner walls for this task.

Do not output line elements with `stroke-width` to represent wall thickness.

## Window Requirements

Windows must be closed polygons, not line strokes.

Window generation procedure:

1. Detect window evidence.
2. Find the host wall.
3. Locate the transition points between wall evidence and window evidence.
4. Split the wall polygon/topology at those two transition points.
5. Replace the wall portion with a blue closed window polygon.

Requirements:

```txt
window endpoints connect to wall geometry
window replaces a wall segment
window does not float
window output is blue
window output is a closed polygon
```

## Door Requirements

Door output consists of:

```txt
door origin - purple
door leaf   - orange
door arc    - red
```

### Door Origin

Door origin is the wall-hosted threshold segment.

Procedure:

1. Detect door_origin evidence.
2. Find the host wall.
3. Locate the two transition/border points between door_origin evidence and wall evidence.
4. Split the wall topology at those points.
5. Replace the wall segment with a purple closed door-origin polygon.

Requirements:

```txt
door origin connects to wall geometry
door origin replaces a wall segment
door origin does not float
door origin output is purple
door origin output is a closed polygon
```

### Door Leaf

The door leaf must be generated from the door origin.

Requirements:

```txt
door leaf starts at one endpoint of the door origin
door leaf is 90 degrees to the door origin
door leaf length matches the door width
door leaf output is orange
door leaf is a closed polygon, not a stroke-only line
```

### Door Arc Direction

The current door arc direction is reversed and must be fixed.

The arc center must be:

```txt
the point of intersection between the door origin line and the opened door leaf line
```

This point is the hinge point.

The arc must:

```txt
use the hinge point as centroid/center
connect the closed-door origin direction to the opened leaf direction
span 90 degrees
follow the side indicated by door_arc and door_leaf evidence
render red
```

Do not trace noisy door arc pixels as an irregular contour.

## Floor Requirements

Floor should be generated after walls and openings.

The CNN floor class has low accuracy. Floor geometry must follow wall topology when floor pixels are ambiguous.

Requirements:

```txt
floor output is white
floor is behind all other components
floor boundary follows the wall-derived outer envelope
floor must not create the outer wall line
floor must not produce visible grey/background-border artifacts
```

## Allowed Angles

All final wall, window, door-origin, and floor polygon edges must use only multiples of 45 degrees:

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

Arbitrary final angles are not allowed.

## Scale and Metric Rules

Use the scale strategy from the active specs:

```txt
explicit metadata if available
dimension metadata if available
multiple confident door-origin widths
wall thickness cross-check
fallback to pixel units only if scale is unsafe
```

For this task, wall polygons should represent:

```txt
200 mm total wall width
100 mm offset on each side of centerline
```

If scale is unknown, preserve the same conceptual rule in pixel units and mark scale as unknown rather than pretending it is metric.

## Required Source Changes

Update the vectorization source under:

```txt
src/vectorization
src/vectorization/primitives
configs/vectorization_v008.yaml
tests
```

Expected changes include:

```txt
replace line-stroke wall output with closed wall polygons
replace line-stroke window output with closed window polygons
replace line-stroke door origin/leaf output with closed polygons
fix door arc center and direction
remove final output paths that draw unidentified/debug components
make all final wall output black
make outer wall extraction wall-evidence-based instead of floor/background-border-based
connect inner walls to outer wall topology where evidence supports it
update tests to assert the new strict output rules
```

## Required Tests

Add or update tests to verify:

1. Outer wall is not derived from floor/background border.
2. Outer wall is a closed polygon system.
3. Inner walls connect to outer walls when evidence supports connection.
4. Wall SVG output uses black closed polygons, not stroke-thickness lines.
5. Window SVG output uses blue closed polygons.
6. Door origin SVG output uses purple closed polygons.
7. Door leaf SVG output uses orange closed polygons.
8. Door arc SVG output uses red and has the hinge point as center.
9. Door arc direction is not reversed.
10. Final SVG contains no unidentified visible groups or dotted debug elements.
11. Final SVG contains no retired 5-class `room`, `icon`, or generic `opening` output groups.
12. All final edges snap to multiples of 45 degrees.

## Acceptance Criteria

This task is complete when:

1. Final vector output visibly follows wall pixels and opening evidence, not floor/background borders.
2. Outer wall is not drawn twice.
3. Inner walls are connected where expected.
4. All wall geometry is black closed polygons with 200 mm total width.
5. Windows are blue closed polygons replacing wall segments.
6. Door origin is purple, door leaf is orange, and door arc is red.
7. Door arc hinge/center is the intersection of door origin and opened leaf.
8. No line-stroke thickness is used as the final representation for walls, windows, door origin, or door leaf.
9. No extra visible components appear in the final SVG.
10. The implementation still follows the previous valid rules from `spec_v007_component_primitives.md` and `spec_v008_mask_to_vector.md` unless this task explicitly supersedes them.
