# Task 05 - Improve Vector Output With Topology-Aware Generalization

## Context

The CNN raster segmentation output is visually good, but the current vectorization result is too literal and not architecturally convincing enough.

The vectorizer should not aim for pixel-perfect contour tracing. It should use the CNN raster as evidence, then produce a more generalized floorplan representation that makes spatial and architectural sense.

The goal is to move from:

```txt
segmentation pixels -> contours -> SVG
```

toward:

```txt
segmentation pixels -> evidence masks -> spatial primitives -> snapped topology -> architectural SVG
```

This task focuses only on vectorization quality. Do not retrain the CNN model.

## Objective

Improve the vector output so floor, wall, door, and window/opening primitives are spatially generalized, orthogonal where appropriate, and topologically aligned.

The output should look like a plausible architectural abstraction instead of a direct raster boundary trace.

## General Requirements

1. Do not retrain the CNN model.
2. Do not change the 5-class CNN class mapping.
3. Use the CNN segmentation raster as evidence, not as a pixel-perfect vector boundary.
4. Prefer rectilinear, snapped, architecture-like primitives over noisy contours.
5. Preserve the existing vectorization workflow and output compatibility unless a small targeted change is required.
6. Keep implementation scoped to vectorization modules and related tests.

## Floor / Building Footprint Requirements

The floor should represent the footprint or filled background area beneath all floorplan elements.

The floor should be derived from the union of meaningful foreground evidence:

```txt
floor_evidence = wall + opening + room + icon/furniture
```

Requirements:

1. The floor must be a filled polygon.
2. The floor should sit behind every other element in the SVG layer order.
3. Walls, openings, doors, windows, and furniture/icon evidence should appear to sit on top of the floor.
4. The floor should cover the area occupied by walls, openings, doors, windows, rooms, and furniture/icon elements.
5. The floor outline should be generalized into horizontal and vertical segments.
6. The floor outline should avoid diagonal or jagged contour segments.
7. The floor outline should align with nearby exterior wall, opening, or door-origin geometry.
8. The visible floor outline should not appear alone as an exposed line inside the drawing.
9. If the floor has a stroke, that stroke should be hidden under or coincident with wall, opening, window, or door-origin linework.
10. Prefer using a filled floor polygon with no visually dominant standalone outline unless needed for debug.

The intended visual result is:

```txt
floor fill behind all elements
floor boundary covered by architectural linework
no floating floor outline visible by itself
```

## Orthogonal Generalization Requirements

The vectorizer should generalize noisy raster boundaries into rectilinear floorplan geometry.

Requirements:

1. Snap floor footprint segments to horizontal or vertical directions.
2. Snap near-horizontal and near-vertical wall segments to exact horizontal or vertical lines.
3. Align floor edges with dominant exterior wall/opening axes.
4. Merge nearby collinear segments where they represent the same architectural edge.
5. Remove tiny orthogonal fragments that come from raster noise.
6. Avoid preserving small pixel-level notches unless they correspond to a meaningful architectural condition.

## Door Primitive Requirements

Door geometry should be generated parametrically from hosted door evidence instead of copied from the raster contour.

Each door primitive should contain exactly three visible components:

```txt
1. door origin segment
2. door opening segment
3. door swing arc
```

Door component rules:

1. The door origin segment must align with the host wall.
2. The door origin segment must be coincident with or directly hosted on the wall/opening interval.
3. The door opening segment must be perpendicular to the door origin segment.
4. The door opening segment must have the same length as the door origin segment.
5. The door swing arc must be a quarter arc.
6. The swing arc center must be the hinge point where the door origin segment and door opening segment intersect.
7. The arc radius must equal the door origin/opening segment length.
8. The door arc direction must be consistent with the selected swing side.

Door swing-side rule:

1. Determine the door origin from the host wall opening interval.
2. Determine candidate door opening directions as the two perpendicular directions to the host wall.
3. Prefer the opening direction on the side where a nearby perpendicular or adjacent wall condition is closer and makes the door placement architecturally plausible.
4. If the nearby-wall rule is ambiguous, use a deterministic default swing side.
5. The default swing side must be consistent across runs.
6. Do not draw a door arc in the opposite or mirrored direction from the selected opening segment.

The door should never be drawn as an arbitrary blob or contour.

## Window / Opening Requirements

Window and opening primitives must be wall-hosted.

Requirements:

1. Every window/opening must be assigned to a host wall when possible.
2. Project each window/opening interval exactly onto its host wall centerline.
3. The window/opening direction must match the host wall direction.
4. Window/opening endpoints must lie exactly on the host wall line.
5. Split the host wall geometry at window/opening endpoints.
6. The split endpoints should become real wall segment endpoints.
7. Do not leave floating window/opening segments near walls.
8. Do not draw window/opening geometry that is merely close to a wall but not coincident with it.
9. If an opening cannot be confidently hosted, place it in the debug/unresolved layer rather than pretending it is hosted.

The intended topology is:

```txt
wall segment -> opening interval -> wall segment
```

not:

```txt
wall segment with a nearby floating opening line
```

## SVG Layering Requirements

The SVG should be layered so the drawing reads as an architectural floorplan:

```txt
floor fill
rooms / optional fills
walls
windows / openings
doors
debug
```

Requirements:

1. Floor fill should be behind all linework.
2. Wall linework should cover or coincide with the floor outline.
3. Door origin segments should align with wall/opening geometry.
4. Window segments should align with split wall intervals.
5. Debug geometry should remain visually separate and optional.

## Suggested Implementation Areas

Implementation will likely touch the vectorization modules under:

```txt
src/vectorization/
```

Likely areas:

```txt
floor or footprint extraction
wall extraction / snapping
opening projection
wall splitting
door primitive generation
SVG export layering
```

If new helpers are needed, keep them in focused vectorization modules. Do not embed substantial vectorization logic in notebooks.

## Tests / Validation

Add or update tests for the new behavior where practical.

Minimum validation cases:

1. A filled floor polygon is generated from union foreground evidence.
2. Floor polygon boundary is rectilinear.
3. Door origin aligns with the host wall.
4. Door opening segment is perpendicular to the door origin.
5. Door opening segment length equals the door origin length.
6. Door arc is a quarter arc centered at the hinge point.
7. Window/opening intervals are projected onto host wall centerlines.
8. Host walls are split at opening/window endpoints.
9. Unhosted openings remain unresolved/debug instead of floating as final geometry.

## Acceptance Criteria

This task is complete when:

- Vector output is more spatially generalized and less pixel-contour-like.
- The floor is exported as a filled polygon behind all other elements.
- The floor footprint includes wall, opening, room, and icon/furniture evidence.
- The floor outline is rectilinear.
- The visible floor outline is covered by or coincident with wall, opening, window, or door-origin linework wherever possible.
- Door primitives are generated from parametric rules rather than arbitrary contours.
- Door origin, opening segment, and swing arc follow the geometric rules in this task.
- Door swing direction no longer appears opposite or mirrored relative to the chosen opening side.
- Windows/openings are projected exactly onto host walls.
- Host walls are split at window/opening endpoints.
- Floating window/opening geometry is eliminated or moved to debug/unresolved output.
- The SVG layer order supports clear architectural reading.
- The CNN model is not retrained or modified.
