# Task 06 - Regenerate Vectorization Pipeline With Four Final Classes

## Context

The CNN segmentation model outputs exactly five raster classes:

```txt
0 - background
1 - wall
2 - opening
3 - room
4 - icon
```

For vectorization, background should be ignored. The final vector output should contain exactly four architectural classes:

```txt
floor
wall
opening
icon
```

The previous vectorization direction treated `room` as a separate vector output class. That is incorrect for the current workflow.

Class `3 - room` should be interpreted as floor evidence, not as a separate final vector class. The vector SVG should not include a `rooms` group or separate room primitives.

This task supersedes any previous instruction that exports separate room geometry.

## Objective

Regenerate the vectorization source so the final SVG is produced procedurally in exactly this order:

```txt
wall -> floor -> opening -> icon
```

The final SVG drawing order must be:

```txt
floor > wall > opening > icon
```

The vectorizer should be stricter and more procedural than the previous contour-style implementation.

## Final Vector Classes

The final vector output must contain only these classes:

| Vector class | Source raster evidence | Output behavior |
|---|---|---|
| floor | class 3 room, constrained by outer wall | filled polygon behind everything |
| wall | class 1 wall | thick centerline wall segments |
| opening | class 2 opening | wall-trimming openings, doors, and windows |
| icon | class 4 icon | simplified filled furniture/fixture shapes |

Background class `0` must be ignored.

Do not export:

```txt
rooms
room polygons
<g id="rooms">
RoomPrimitive
```

unless a temporary internal helper is unavoidable. If internal helpers exist, they must not appear as final SVG output.

## Required Pipeline Order

The vectorization process must follow this order:

```txt
1. wall
2. floor
3. opening
4. icon
```

Do not generate openings before walls.

Do not generate floor before the outer wall is resolved.

Do not generate furniture/icon shapes before walls, floor, and openings are resolved.

## Step 1 - Wall Generation

Walls must be generated first from raster class `1 - wall`.

Requirements:

1. Ignore openings while generating the initial wall geometry.
2. Generate walls based on the wall raster evidence.
3. Snap wall angles to 45-degree increments.
4. Prefer horizontal and vertical wall output when evidence is near orthogonal.
5. The most outer wall must form a closed rectilinear loop.
6. The outer wall loop should be created before inner walls.
7. Inner walls should be generated after the outer wall loop.
8. Walls should be represented as thick centerline SVG geometry, not filled wall polygons.
9. Wall geometry should be generalized from raster evidence rather than pixel-perfect contours.
10. Wall segments should be merged and cleaned so the result reads as architectural linework.

Closed outer wall requirement:

```txt
The exterior wall centerlines should form a closed rectilinear loop.
```

The wall loop itself is not filled. It remains wall linework.

## Step 2 - Floor Generation

The floor must be generated after the outer wall has been created.

Requirements:

1. The floor represents class `3 - room` as a final floor area, not as room objects.
2. The floor boundary should be generated from the outer wall loop.
3. The floor boundary should be a direct translation of the outer wall curves/segments.
4. The floor must be a filled polygon.
5. The floor must use `stroke="none"`.
6. The floor must sit behind all wall, opening, and icon geometry.
7. The floor must not create a separate visible outline.
8. The floor should visually read as the area enclosed by the outer walls.

The final SVG should include:

```txt
<g id="floor">
```

The final SVG should not include:

```txt
<g id="rooms">
```

## Step 3 - Opening Generation

Openings must be generated after walls and floor.

Openings come from raster class:

```txt
2 - opening
```

This class includes both doors and windows.

Requirements:

1. Openings must be wall-hosted.
2. Openings must trim or split wall geometry.
3. Both doors and windows must trim/split the walls.
4. Opening endpoints must lie exactly on wall centerlines.
5. Floating openings are not allowed in final output.
6. If an opening cannot be hosted on a wall, keep it out of final geometry or place it in debug output only.

### Door Rule

For door openings:

1. Trim the host wall at the door interval.
2. Replace the trimmed wall portion with the door origin segment.
3. The door origin segment must align with the host wall.
4. Generate the door opened segment perpendicular to the door origin.
5. The opened segment length must equal the door origin segment length.
6. Generate a quarter-circle door arc.
7. The arc center must be the hinge point where the door origin and opened segment intersect.
8. Door geometry must be parametric, not copied from raster contours.

### Window Rule

For window openings:

1. Trim or split the host wall at the window interval.
2. Replace the trimmed wall portion with a blue line segment.
3. The window segment should use the same basic geometry style as wall centerline geometry.
4. The window must be coincident with the host wall centerline.
5. The window segment endpoints must be wall split endpoints.

Windows should read as wall-aligned blue line segments.

## Step 4 - Icon / Furniture Generation

Icons come from raster class:

```txt
4 - icon
```

Requirements:

1. Generate icon/furniture geometry after walls, floor, and openings.
2. Export icons as simplified filled shapes.
3. The icon output should be generalized and cleaned.
4. Do not treat icons as floor, wall, or opening geometry.
5. Icon/furniture geometry should sit above floor and below or above linework only if the SVG layering requires it for readability.

Final SVG group:

```txt
<g id="icon">
```

## Final SVG Group Order

The SVG output must use this group order:

```txt
<g id="floor">
<g id="wall">
<g id="opening">
<g id="icon">
```

Do not use plural group names for these final classes.

Do not include:

```txt
<g id="rooms">
<g id="walls">
<g id="openings">
<g id="windows">
<g id="doors">
```

unless they are debug-only and clearly separated from final output.

The final semantic output should be exactly four groups:

```txt
floor
wall
opening
icon
```

## Color / Visual Requirements

Use clear class-specific visual styling:

```txt
floor   - filled polygon, stroke none
wall    - thick dark centerline segments
opening - doors and windows hosted on wall intervals
icon    - simplified filled furniture/fixture shapes
```

Window openings should be blue line segments.

Door openings should use the parametric door symbol:

```txt
door origin segment
door opened segment
quarter arc
```

## Implementation Scope

Regenerate or substantially revise the vectorization source as needed.

Keep the scope inside:

```txt
src/vectorization/
```

and related tests/configs unless another file is strictly required.

The implementation should be procedural and explicit. Avoid giving the vectorizer multiple competing interpretations of the same raster class.

## Testing / Validation

Add or update tests to verify:

1. The final SVG has no `rooms` group.
2. The final SVG has exactly the final semantic groups:

```txt
floor
wall
opening
icon
```

3. SVG group order is floor, wall, opening, icon.
4. Floor is filled and has `stroke="none"`.
5. Wall geometry is generated before opening geometry.
6. Outer wall geometry forms a closed rectilinear loop.
7. Wall angles are snapped to 45-degree increments.
8. Openings trim or split host walls.
9. Door origin segments replace trimmed wall portions.
10. Window openings are blue wall-aligned line segments.
11. Icons are exported as simplified filled shapes.

## Acceptance Criteria

This task is complete when:

- The vectorizer final output uses exactly four classes: floor, wall, opening, icon.
- Background is ignored.
- Class `3 - room` is no longer exported as room geometry.
- Class `3 - room` contributes to floor generation only where needed.
- The SVG does not contain a final `rooms` group.
- Walls are generated first.
- The outer wall centerlines form a closed rectilinear loop.
- Inner walls are generated after outer walls.
- Wall angles are snapped to 45-degree increments.
- Floor is generated from the outer wall loop.
- Floor is filled and has no stroke.
- Openings are generated after walls and floor.
- Doors and windows trim/split host walls.
- Door symbols are parametric and replace trimmed wall portions.
- Window openings are blue line segments coincident with wall intervals.
- Icons/furniture are generated last as simplified filled shapes.
- Final SVG group order is exactly floor, wall, opening, icon.
