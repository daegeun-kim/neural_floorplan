# Task 33 - Preserve Door/Window Openings While Resolving Trim Overlap

## Objective

Fix Phase 4 opening interval overlap behavior.

Current issue:

```txt
When a door and window are close on the same wall segment,
their wall-trim intervals can overlap.
This creates weird final vector output.
```

The previous suggested conflict behavior of rejecting one opening is wrong for this project.

If an opening was detected from the 7-class raster and successfully hosted on the wall graph, it is usually a valid architectural opening. The default behavior should preserve both openings and adjust their intervals so the final wall trim has no overlap.

## Source Spec

Follow the updated rule in:

```txt
specs/spec_v008_phase4_vectorization.md
```

Relevant section:

```txt
## 9. Opening Interval Editing
```

## Core Principle

Do not delete a valid detected door/window just because its interval slightly overlaps another valid opening.

Instead:

```txt
1. keep both openings whenever physically possible
2. resolve overlap in 1D wall-chain interval space
3. move or shrink the lower-priority interval away from the higher-priority interval
4. trim walls only after all opening intervals on the chain are non-overlapping
```

Reject an opening only as a last resort when there is no possible non-overlapping placement on the host wall chain.

## Required Interval Representation

For every hosted opening, compute and store:

```txt
host_chain_id
opening_type
source_component_id
t_start
t_end
center_t
width_px
width_mm
confidence
original_t_start
original_t_end
adjusted_t_start
adjusted_t_end
adjustment_reason
```

The conflict solver should operate on these 1D intervals per wall chain.

## Door Vs Window Rule

If a door interval overlaps a window interval:

```txt
always keep the door fixed
always move or shrink the window
```

Reason:

```txt
door geometry is more constrained by hinge/end/origin/arc evidence
window geometry is usually easier to slide slightly along the same host wall
```

The window should be pushed away from the door along the same wall chain until:

```txt
window interval and door interval no longer overlap
```

If the original floorplan has a tiny wall segment between them, preserve that separation in the vector result.

## Door Vs Door Rule

If two door intervals overlap:

```txt
keep the higher-confidence door fixed
move or shrink the lower-confidence door
```

Door confidence should primarily come from:

```txt
red door_arc evidence
```

Secondary evidence:

```txt
orange door_leaf evidence
purple door_origin evidence
graph-hosting quality
```

Do not delete either door unless no valid non-overlapping position remains.

## Window Vs Window Rule

If two window intervals overlap:

```txt
keep the higher-confidence window fixed
move or shrink the lower-confidence window
```

Window confidence should primarily come from:

```txt
blue window evidence
```

Secondary evidence:

```txt
graph-hosting quality
component size/cleanliness
```

Do not merge windows just because they overlap slightly unless they are confirmed fragments of the same blue component or explicitly detected as one fragmented window.

## Minimum Separator

After conflict resolution, adjacent openings should have a small positive wall separator where possible.

Use a configurable separator:

```txt
min_opening_separator_mm = 50
```

When scale is resolved:

```txt
min_separator_px = 50 / px_to_mm
```

When scale is unknown:

```txt
use a small configured pixel fallback
record scale_blocked for metric-based separator decisions
```

The separator should not create unrealistic movement. If the wall segment is too short to preserve the full separator, use the largest feasible non-overlapping separation and record the compromise.

## Adjustment Strategy

For each wall chain:

```txt
1. collect all hosted door/window intervals on that chain
2. sort by interval position
3. detect overlap or separator violations
4. assign priority:
   door fixed over window
   higher confidence fixed over lower confidence for same type
5. move lower-priority interval away from higher-priority interval
6. if movement exceeds available wall-chain bounds, shrink lower-priority interval if allowed
7. if still impossible, mark unresolved and reject only then
8. recompute snapped endpoints from adjusted t_start/t_end
9. trim wall intervals using adjusted non-overlapping intervals
```

## Movement Constraints

An adjusted interval must remain:

```txt
on the same host wall chain
within the valid chain extent
orthogonal with the host wall
at a plausible location near the original semantic component
```

Record adjustment distance:

```txt
adjustment_px
adjustment_mm
```

If adjustment distance becomes too large, flag the opening as low-confidence or unresolved rather than silently moving it far away.

Recommended configurable guard:

```txt
max_opening_adjustment_mm = 200
```

If scale is unknown, use a pixel fallback and record scale-blocked status.

## JSON Requirements

Update `final_vector.json` so every accepted opening records whether it was adjusted:

```json
{
  "opening_type": "door|window",
  "source_component_id": 0,
  "host_chain_id": 0,
  "original_interval": [0.0, 0.0],
  "adjusted_interval": [0.0, 0.0],
  "was_adjusted": false,
  "adjustment_reason": "",
  "adjustment_px": 0.0,
  "adjustment_mm": 0.0,
  "overlap_resolution_priority": "door_fixed|higher_confidence_fixed|not_needed"
}
```

If an opening is rejected only because no feasible de-overlap is possible, record:

```txt
rejection_reason = "no_feasible_non_overlapping_interval"
```

## SVG Requirements

`final_vector.svg` must use adjusted non-overlapping intervals for:

```txt
wall trimming
window placement
door origin placement
door primitive geometry
```

There must be no final wall trim overlap between accepted openings on the same wall chain.

## Debug Overlay Requirements

`image_debug_overlay.png` should show:

```txt
original opening intervals
adjusted opening intervals
opening type
which interval was moved/shrunk
unresolved overlap if any remains
```

This is important because interval adjustment changes geometry from the direct semantic component position. The debug overlay must make that adjustment auditable.

## Tests

Add focused tests for:

```txt
door-window overlap keeps door fixed and moves window
door-door overlap keeps higher-confidence door fixed
window-window overlap keeps higher-confidence window fixed
slight overlap becomes non-overlap with separator
wall trimming uses adjusted intervals
JSON records original and adjusted intervals
opening is rejected only when no feasible non-overlapping placement exists
```

Use synthetic wall-chain intervals for unit tests so these rules can be tested without GPU inference.

## Completion Criteria

This task is complete when:

```txt
1. valid overlapping openings are preserved whenever possible
2. door-window overlap always moves/shrinks the window, not the door
3. same-type overlap moves/shrinks the lower-confidence opening
4. wall trimming uses only adjusted non-overlapping intervals
5. final_vector.json records all interval adjustments
6. image_debug_overlay.png makes interval adjustment visible
7. tests cover the interval de-overlap rules
```

After completion, rename this file from:

```txt
tasks/task33.md
```

to:

```txt
tasks/task33_done.md
```
