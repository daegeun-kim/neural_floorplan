# Task 15 - Simplify Wall Points And Loosen Door Recognition

## Objective

Continue improving the current vectorization pipeline without redesigning it from scratch.

The SVG output has improved through debugging. Keep as much existing code and pipeline structure as possible. Only change the parts needed for the problems listed below.

As before, the final output and debug artifacts must be checked against:

```txt
specs/vectorization_must_rules.md
```

Continue debugging until the generated `vector.svg`, `debug_overlay.png`, and `metrics.json` satisfy all observable must rules.

## Current Problems And Required Fixes

### 1. Wall Point Recognition Is Too Brittle

Current problem:

```txt
The performance of identifying 1_wall_point, 2_wall_point, 3_wall_point, and 4_wall_point is too low.
These four categories are currently too demanding and too sensitive to noisy raster evidence.
```

Required change:

```txt
Merge 1_wall_point, 2_wall_point, 3_wall_point, and 4_wall_point into one generic wall point type internally.
```

After axis alignment:

```txt
connect all compatible points on the same axis
```

This connection logic must include:

```txt
generic wall points
wall_window_point
wall_door_hinge_point
wall_door_end_point
```

`wall_window_point`, `wall_door_hinge_point`, and `wall_door_end_point` are not generic wall points, but they must still participate in wall-axis connection when they lie on the same valid wall axis.

Important:

```txt
Do not require accurate classification into 1/2/3/4 wall-point subtypes before wall graph construction.
```

### 2. Walls Are Not Orthogonal Enough

Current problem:

```txt
Walls are still not consistently orthogonal to each other.
```

Required change:

```txt
Increase the point axis-alignment threshold to 1000 mm.
```

Connection rule:

```txt
If two points are not aligned onto the same axis, do not connect them to form a wall on that axis.
```

The vectorizer should prefer missing/shorter wall connections over creating non-orthogonal wall edges.

Final wall/window/door-origin edges must remain orthogonal.

### 3. Door Width Snapping Still Fails

Current problem:

```txt
Scale recognition from red door bbox inference is mostly accurate, but final door sizes are still not always snapped to 700 mm or 900 mm.
```

Required change:

```txt
Debug the door-width snapping path again.
Ensure every final door-origin width is exactly 700 mm or 900 mm when scale is resolved.
Ensure door leaf length and door arc radius match that snapped door width.
```

This must be visible in the final output and/or metrics.

### 4. Red Door Clusters Are Still Too Often Rejected

Current problem:

```txt
Some doors are still recognized as rejected or unresolved evidence.
The debug overlay currently distinguishes rejected evidence, door_arc accepted with confidence < 0.75, and door_arc accepted with confidence > 0.75.
```

Required change:

```txt
Lower the door recognition threshold much lower.
Almost any connected red pixel group that survives cleanup should be categorized as a door.
```

The intent is:

```txt
red door_arc cluster -> door object
```

Weak purple/orange/black support should reduce confidence, but should not prevent door creation unless there is truly no plausible wall/door geometry after fallback.

## Required Test Image

Use the same primary debugging target from Task 14:

```txt
C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\outputs\vectorization\v008\iteration5_run3
```

Vectorize the image using:

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

## Constraints

Do not redesign the whole vectorization pipeline.

Do not add unrelated vectorization instructions beyond this task and:

```txt
specs/vectorization_must_rules.md
```

Keep as much current code as possible.

Only change the parts needed to:

```txt
simplify wall point handling
improve orthogonal axis alignment
snap final door widths to 700/900 mm
make nearly all accepted red clusters become doors
```

Do not modify CNN training code.

Do not change `specs/vectorization_must_rules.md` unless the user explicitly asks for a rules change.

## Required Verification

At minimum, verify:

1. Wall graph construction no longer depends on accurate 1/2/3/4 wall-point subtype classification.
2. Generic wall points, wall-window points, wall-door hinge points, and wall-door end points can all participate in wall-axis connection.
3. Axis alignment uses `1000 mm` as the configured threshold.
4. No final wall edge is created between points that are not on the same aligned axis.
5. Final wall/window/door-origin edges are orthogonal.
6. Final door-origin widths are exactly `700 mm` or `900 mm` when scale is resolved.
7. Door leaf length equals the snapped door-origin width.
8. Door arc radius equals the snapped door-origin width.
9. Accepted red `door_arc` clusters become door objects except in truly impossible cases.
10. Lower-confidence red-door inference appears in debug/metrics instead of becoming final rejection.
11. Final `vector.svg` still satisfies the visible group, color, and geometry rules.
12. `debug_overlay.png` and `metrics.json` still explain rejected evidence and door candidates.

## Required Tests

Add or update tests to protect the fixed behavior.

Tests should cover:

```txt
generic wall point handling replacing dependency on 1/2/3/4 subtype accuracy
wall connection only on shared aligned axes
1000 mm axis-alignment threshold configuration
door width snapping to exactly 700 mm or 900 mm
door leaf and arc using the snapped door width
red door_arc clusters becoming door candidates even with weak purple/orange evidence
```

## Acceptance Criteria

This task is complete when:

1. The current vectorization source has been minimally updated for the four listed problem areas.
2. The required test image has been regenerated.
3. The generated `vector.svg`, `debug_overlay.png`, and `metrics.json` satisfy all observable rules in `specs/vectorization_must_rules.md`.
4. Door widths are uniform and snapped to `700 mm` or `900 mm`.
5. Accepted red door clusters no longer disappear as rejected/unresolved evidence except in truly impossible cases.
6. Tests cover the changed behavior.
7. Any must-rule that cannot be automatically verified is named in the final report with a clear reason.

## Clarification Guidance

Ask the user before implementation if any of the following are unclear:

```txt
whether the public/exported point type names must remain 1_wall_point/2_wall_point/3_wall_point/4_wall_point for compatibility
whether the generic wall point should appear in debug_overlay/metrics as wall_point or be mapped back to the old names for display
how low the red-door confidence threshold should be numerically
whether tiny red clusters above cleanup threshold should always become doors even if visually suspicious
```
