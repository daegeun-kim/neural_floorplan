# Task 17 - Door-First Vectorization From Red Bboxes

## Objective

The current vectorization performance is still too low because the process contains too many small judgments before the most reliable evidence is used.

The red `door_arc` bounding boxes are currently the most accurate recognized geometry. The vectorization process should therefore start from red door bboxes and use them as trusted anchors.

Simplify the process. Do not keep adding minor early rejection rules. Start from the assumption that the red bbox is correct, build the graph from that, then add complexity only where it clearly improves output.

The final output must still satisfy:

```txt
specs/vectorization_must_rules.md
```

## Current Failure

The previous debugging did not fully solve door-point inference.

In the debug overlay:

```txt
wall_door_hinge_point
wall_door_end_point
```

are not reliably visible as two vertices of the red bbox.

The door hinge points are clearly not located on the red bbox vertices, and the door hinge/end pair is not reliably aligned with wall lines.

Required correction:

```txt
For each accepted red bbox, wall_door_hinge_point and wall_door_end_point must be two adjacent vertices of that red bbox.
These two points must also align with the host wall direction.
```

## Door-First Vectorization Process

The vectorization process must always start from doors:

```txt
1. Detect connected red door_arc components.
2. Get each red component bbox.
3. Use the bbox as trusted door geometry.
4. Select two adjacent bbox vertices as wall_door_hinge_point and wall_door_end_point.
5. Use those door points as standard anchors for wall-point and wall-window-point detection.
6. Align points.
7. Connect points only when strict axis and wall-evidence conditions are satisfied.
8. Generate final SVG/debug/metrics.
```

Do not begin by trying to perfectly classify every wall point. Door bboxes are the initial source of truth.

## Red Bbox Assumption

Assume the red bbox is correct.

Do not overcomplicate red-bbox acceptance with many early rejection conditions.

The current red boxes are the highest-accuracy recognition signal, so the source generation should start by trusting them.

This task is intentionally simple:

```txt
red bbox is correct
door points come from red bbox vertices
other points snap toward door-derived axes
connections require same axis plus weak wall-pixel support
```

## Door Point Selection

For every accepted red `door_arc` bbox:

1. Compute the four bbox vertices.
2. Score each bbox vertex as a candidate for:
   - `wall_door_hinge_point`
   - `wall_door_end_point`
3. Scoring must use nearby evidence from:
   - black wall pixels
   - purple door_origin pixels
4. Choose the two highest-scoring adjacent vertices.
5. The selected vertices must be adjacent, not opposite.
6. The selected adjacent pair should be the bbox edge with the closest alignment to the host wall.

The host wall hint comes from:

```txt
black wall pixels
purple door_origin pixels
```

The selected pair should represent the door-origin side of the red bbox.

## Door Anchors Drive Wall Axes

The door bbox vertices are trusted anchors.

Wall points should snap to the axes implied by door vertices.

Do not snap the door vertices toward noisy wall points as the first step.

Required behavior:

```txt
door vertices define reliable x/y axes
nearby wall points and wall-window points snap to those door-derived axes
wall graph construction uses these snapped axes
```

If a wall point is near a door-derived axis, prefer snapping the wall point to that axis.

## Axis Alignment Threshold

Return the axis-alignment threshold to:

```txt
500 mm
```

Do not use the `1000 mm` threshold from the previous experiment.

The goal is not to over-merge distant geometry. The goal is to use door-derived axes as stronger anchors.

## Wall Point And Window Point Detection

After door bbox vertices are established:

```txt
locate generic wall points
locate wall_window_point endpoints
use door points as standard alignment anchors
```

The process should not depend on accurately distinguishing:

```txt
1_wall_point
2_wall_point
3_wall_point
4_wall_point
```

Use generic wall-point behavior internally where helpful.

`wall_window_point`, `wall_door_hinge_point`, and `wall_door_end_point` must still be preserved as special opening-related point types.

## Connection Rules

Connect two points only when both conditions are true:

### 1. Same Axis

The two points must be on the same aligned horizontal or vertical axis.

If two points are not on the same axis, do not connect them to form a wall.

This rule exists to guarantee orthogonal wall lines.

### 2. Wall Pixel Support

There must be black wall-pixel evidence between the two points.

Use a simple low-threshold test:

```txt
only a small hint of continuous black pixels along the interval should allow connection
```

The threshold should be intentionally low.

The point is to prevent arbitrary connections while avoiding the current over-rejection problem.

Do not require perfect continuous wall pixels. A weak but coherent black-pixel line along the interval is enough.

## Door Geometry Rules

For each red bbox:

```txt
the selected adjacent vertices become wall_door_hinge_point and wall_door_end_point
door origin is generated from those two vertices
door leaf and door arc are generated from the door-origin segment
door width must snap to 700 mm or 900 mm
door leaf length must match the snapped width
door arc radius must match the snapped width
```

The door hinge/end points must be visible in `debug_overlay.png` as bbox-vertex-derived points.

## Keep The Pipeline Simple

Avoid adding many small conditional judgments at the beginning of the process.

The previous approach became too complicated because it tried to solve every ambiguity before establishing trusted anchors.

For this task:

```txt
start simple
trust red bbox
derive door points
align other points to door axes
connect only same-axis points with weak wall evidence
then evaluate output
```

Only add extra conditions after the simple version is working and only if they directly improve the observable output.

## Required Debug Overlay Behavior

The debug overlay must clearly show, for every accepted red bbox:

```txt
red bbox
selected wall_door_hinge_point at one bbox vertex
selected wall_door_end_point at an adjacent bbox vertex
door-origin edge between those two vertices
```

The overlay should make it easy to visually confirm that hinge/end points are bbox vertices and are aligned with wall evidence.

## Required Metrics

Metrics must record, for each red bbox:

```txt
red_component_id
red_bbox
all_four_bbox_vertices
selected_hinge_vertex
selected_end_vertex
hinge_vertex_score
end_vertex_score
selected_bbox_edge
host_wall_alignment_score
created_door_candidate
door_width_mm
door_confidence
door_inference_notes
```

Existing metrics may be extended rather than replaced.

## Required Test Image

Use the same primary debugging target:

```txt
C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\outputs\vectorization\v008\iteration5_run3
```

Vectorize using:

```txt
C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\notebooks\run_single_image_run3_vectorization.ipynb
```

or the equivalent source calls invoked by that notebook.

Regenerate and inspect:

```txt
vector.svg
debug_overlay.png
metrics.json
```

## Required Verification

Verify at minimum:

1. Every accepted red bbox creates one door candidate.
2. Every accepted red bbox selects two adjacent bbox vertices.
3. Selected door hinge/end points are visible on red bbox vertices in debug overlay.
4. Selected door hinge/end pair aligns with black wall and/or purple door-origin evidence.
5. Axis alignment threshold is back to `500 mm`.
6. Wall points and wall-window points can snap to door-derived axes.
7. Wall connections are created only between same-axis points.
8. Wall connections require at least weak continuous black-pixel support.
9. Final wall/window/door-origin edges are orthogonal.
10. Door widths snap to `700 mm` or `900 mm`.
11. Door leaf length and door arc radius match the snapped door width.
12. Final SVG/debug/metrics satisfy `specs/vectorization_must_rules.md`.

## Required Tests

Add or update tests for:

```txt
red bbox vertex extraction
adjacent vertex pair selection
wall/purple evidence scoring for bbox vertices
door hinge/end points being bbox vertices
door-derived axes attracting nearby wall/window points
500 mm axis alignment threshold
same-axis-only wall connection
weak black-pixel support allowing wall connection
door width snapping to 700/900 mm
debug/metrics reporting selected bbox vertices
```

## Acceptance Criteria

This task is complete when:

1. Vectorization starts from accepted red bboxes.
2. Door hinge/end points are selected from adjacent red bbox vertices.
3. Door hinge/end points are visible and correct in the debug overlay.
4. Wall/window point alignment uses door-derived axes as anchors.
5. Wall connections are orthogonal and supported by black wall pixels.
6. Door sizes are snapped to `700 mm` or `900 mm`.
7. The required test image output satisfies all observable must rules.
8. Tests cover the new door-first behavior.

## Clarification Guidance

Ask the user before implementation if any of the following are unclear:

```txt
the exact vertex scoring formula for black/purple evidence
how much black-pixel evidence is enough for weak wall support
how to display generic wall points versus legacy 1/2/3/4 wall point names
whether all accepted red clusters, including very tiny but above-threshold clusters, must create doors
```
