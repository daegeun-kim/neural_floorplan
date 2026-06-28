# Task 32 - Fix Phase 4 Door Primitive Regression

## Objective

Fix the Phase 4 door primitive rendering regression observed in:

```txt
outputs/vectorization/phase4_vectorization/1316/final_vector.svg
```

The current Phase 4 output has acceptable first-attempt wall/window behavior, but the door primitive is wrong.

This is not a new geometry problem. Phase 3 already had a good door primitive model. Phase 4 should reuse that primitive contract instead of hand-drawing a simplified door incorrectly.

## Current Failure

In the current Phase 4 `final_vector.svg`, a door is rendered roughly as:

```txt
purple circle at hinge/origin point
red line from snapped_point_0 to snapped_point_1
red arc starting at snapped_point_0
```

Example from sample `1316`, door 0:

```txt
snapped_points:
  (195, 180)
  (228, 180)
```

Current incorrect interpretation:

```txt
red line (195,180) -> (228,180) is treated as the door leaf
arc starts at (195,180)
purple origin is drawn only as a circle
```

Correct interpretation:

```txt
(195,180) -> (228,180) is the door origin / wall-hosted threshold edge
the leaf is a perpendicular line from the hinge
the arc is centered on the hinge
the arc starts at the origin far point and ends at the leaf endpoint
```

## Root Cause

Phase 4 currently reimplemented door SVG drawing locally instead of using the existing Phase 3 primitive math.

Known relevant files:

```txt
src/vectorization/primitives/door.py
src/vectorization/phase4/export_svg.py
```

The good primitive model already exists in:

```txt
src/vectorization/primitives/door.py
```

It defines:

```txt
DoorOriginPrimitive
DoorLeafPrimitive
DoorArcPrimitive
```

Phase 4 should adapt hosted openings into those primitives or exactly preserve their geometry contract.

## Correct Door Primitive Contract

Every accepted hosted door must render as three primitives:

```txt
1. door_origin edge
2. door_leaf line
3. door_arc arc
```

### Door Origin

The two hosted/snapped door points are the door origin edge.

Given:

```txt
snapped_points = [p0, p1]
```

The origin edge is:

```txt
p0 -> p1
```

This should be rendered as a purple door-origin line, not as the red door leaf.

### Door Leaf

The door leaf starts at the hinge point and extends perpendicular to the door-origin edge.

Given:

```txt
hinge_point = one endpoint of the origin edge
origin_far_point = the other endpoint
width = distance(hinge_point, origin_far_point)
orientation_angle = angle from hinge_point to origin_far_point
```

The leaf endpoint is:

```txt
leaf_end = hinge_point + perpendicular_vector(origin_direction, swing_side) * width
```

The leaf should be rendered as an orange door-leaf line.

### Door Arc

The door arc is a 90-degree arc:

```txt
center = hinge_point
radius = width
start = origin_far_point
end = leaf_end
```

The arc must be centered on the hinge point. It must not start at the hinge point.

## Required Implementation

Replace the Phase 4 local simplified door drawing with one of these acceptable approaches:

```txt
Preferred:
  create an adapter from HostedOpening -> DoorOriginPrimitive / DoorLeafPrimitive / DoorArcPrimitive

Acceptable:
  implement equivalent geometry in Phase 4 export code, but keep the exact same primitive contract
```

Do not keep the current simplified local behavior.

Do not draw a purple circle as the only door-origin primitive.

Do not treat the hosted origin edge as the door leaf.

Do not start the door arc at the hinge.

## Hinge Selection

The fix must make hinge selection explicit.

Current hosted door data contains two snapped points but may not preserve which one is the hinge.

Required behavior:

```txt
1. Prefer hinge identity from door detection/hosting if available.
2. If not available, infer hinge from nearby door_leaf evidence.
3. If door_leaf evidence is not available, use a deterministic fallback and record it in final_vector.json/debug metrics.
```

The fallback may choose `snapped_points[0]` as hinge only if the output records that hinge selection was fallback-based.

Future improvement can refine hinge selection, but this task must at least prevent the primitive role swap.

## Swing Side Selection

The swing side must determine which perpendicular direction the leaf uses.

Required behavior:

```txt
1. Prefer door_leaf and door_arc segmentation evidence to choose swing side.
2. If evidence is unavailable or ambiguous, use a deterministic fallback.
3. Record fallback usage in final_vector.json/debug metrics.
```

The door arc must remain centered on the hinge regardless of swing side.

## JSON Requirements

Update `final_vector.json` for each hosted door so the door primitive information is explicit.

Each accepted door should include:

```json
{
  "source_component_id": 0,
  "snapped_points": [[0, 0], [0, 0]],
  "hinge_point": [0, 0],
  "origin_far_point": [0, 0],
  "leaf_end": [0, 0],
  "swing_side": "left|right|fallback_left|fallback_right",
  "width_px": 0,
  "width_mm": 0,
  "primitive_contract": "door_origin_leaf_arc"
}
```

If this schema is too verbose for the main output, equivalent information may be placed under a nested `door_geometry` field. It must still be available for debugging.

## SVG Requirements

`final_vector.svg` must render doors with separate semantic elements:

```txt
door_origin: purple line from hinge_point to origin_far_point
door_leaf: orange line from hinge_point to leaf_end
door_arc: red 90-degree arc from origin_far_point to leaf_end, centered on hinge_point
```

Recommended group structure:

```xml
<g id="door_N" data-type="door">
  <line data-type="door_origin" ... />
  <line data-type="door_leaf" ... />
  <path data-type="door_arc" ... />
</g>
```

Do not render debug hinge circles in `final_vector.svg`.

If hinge markers are useful, draw them only in:

```txt
image_debug_overlay.png
```

## Test Sample

Use at least this sample for visual verification:

```txt
outputs/vectorization/phase4_vectorization/1316/final_vector.svg
```

Door 0 currently has:

```txt
origin edge: (195,180) -> (228,180)
width: 33 px
```

After the fix, it should visibly become:

```txt
purple origin line along the wall opening
orange perpendicular leaf from one endpoint
red quarter arc centered on that same endpoint
```

The same rule should apply to all doors in sample `1316`.

## Tests

Add focused tests for:

```txt
HostedOpening -> door primitive adapter
door origin is rendered as the snapped host edge
door leaf is perpendicular to the origin edge
door arc starts at origin_far_point, not hinge_point
door arc endpoint equals leaf_end
door arc center is the hinge point
fallback hinge selection is recorded when used
```

If direct SVG arc-center testing is awkward, test the intermediate door geometry object before SVG serialization.

## Completion Criteria

This task is complete when:

```txt
1. Phase 4 no longer hand-draws doors with the hosted origin edge as the leaf.
2. Phase 4 reuses or exactly matches the Phase 3 DoorOrigin/DoorLeaf/DoorArc primitive contract.
3. final_vector.svg contains separate door_origin, door_leaf, and door_arc elements.
4. final_vector.json records hinge_point, origin_far_point, leaf_end, and swing_side or equivalent debug geometry.
5. Sample 1316 visually shows correct door primitives.
6. Focused tests pass, or any environment blocker is clearly documented.
```

After completion, rename this file from:

```txt
tasks/task32.md
```

to:

```txt
tasks/task32_done.md
```
