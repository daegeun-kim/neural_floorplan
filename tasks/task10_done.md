# Task 10 - Recover Inner Walls and Anchor Openings/Doors to Wall Topology

## Context

The current v008 vectorization is relatively strong at:

```txt
detecting the outer wall loop
hosting windows on walls
hosting doors on walls
applying primitive geometry directly to the plan
```

The main failure is now:

```txt
when an outer wall loop is present, inner walls are rarely or never generated
```

Based on the current code and observed behavior, the most likely cause is
`_erase_claimed_wall_components()` in `src/vectorization/wall_extraction.py`.

The current inner-wall path does this:

```txt
1. Build outer wall loop.
2. Draw a band around the simplified outer polygon.
3. Find every connected wall component touched by that band.
4. Remove the entire connected component.
5. Run skeleton + Hough inner-wall extraction on the remaining wall pixels.
```

This prevents duplicate outer walls, but it also removes valid inner walls
whenever the CNN wall mask connects interior walls to the outer wall in one
continuous black component. In typical floorplans, interior partition walls
often touch exterior walls, so removing the whole connected component erases
the very evidence needed for inner walls.

## Objective

Update vectorization so:

```txt
outer wall detection remains intact
inner walls are recovered as individual segments inside the closed outer loop
inner wall endpoints snap/connect to the outer wall loop when spatially close
windows and door origins split/terminate wall segments cleanly
door count and location are driven by reliable red door-arc evidence
door hinge and origin/leaf geometry are paired and module-snapped
all topology decisions are made in real architectural scale, millimeters
```

This task should refine the current vectorization logic. Do not replace the
outer wall strategy wholesale.

## Spatial Reasoning

Typical architectural floorplans follow these assumptions:

```txt
the outer wall loop is the building envelope
interior partition walls lie mostly inside that loop
interior walls often connect to the exterior wall at one or both ends
door and window openings interrupt wall geometry rather than floating over it
door swing arcs are more reliable door-location evidence than thin threshold/leaf pixels
most walls are orthogonal unless explicit diagonal evidence exists
```

Therefore, after the outer loop is known, inner wall detection should not be
based on whole connected-component removal. It should preserve interior wall
branches while suppressing only the exterior wall band itself.

## Root Cause To Fix

Current likely misbehavior:

```txt
outer wall and inner wall pixels are one connected wall component
outer loop band touches that component
_erase_claimed_wall_components removes the full component
no interior wall pixels remain for skeleton/Hough
inner_walls = []
```

Required correction:

```txt
remove only pixels that spatially belong to the outer wall band
preserve wall pixels inside the outer loop, including branches touching the outer wall
extract inner wall centerlines from the preserved interior wall evidence
```

## Inner Wall Extraction Requirements

After the outer wall loop is created:

```txt
1. Define the interior region of the closed outer loop.
2. Build an inner-wall candidate mask from wall pixels inside the loop.
3. Remove only the outer-wall envelope band, not entire connected components.
4. Preserve interior wall branches even if connected to the outer wall.
5. Skeletonize the preserved inner-wall mask.
6. Extract line segments from the skeleton.
7. Merge collinear fragments.
8. Snap wall angles, strongly preferring 90-degree relationships.
9. Snap/extend endpoints to nearby wall topology.
```

Inner walls should be represented as individual segments inside the closed
outer wall loop.

The inner-wall candidate mask (step 2) is the union of black wall pixels and
purple `door_origin` pixels. This bridges the gap a doorway leaves in the
wall mask the same way the outer loop already bridges openings, so an inner
wall is not falsely cut short or dropped at a doorway. After the inner wall
segment is produced, it is trimmed/split to fit the door exactly like the
existing outer-wall door-splitting path (see "Wall Splitting After
Openings"). `door_arc`/`door_leaf`/window pixels are NOT added to the
inner-wall candidate mask - only wall + `door_origin`.

## Inner Wall Endpoint Attachment Rule

If one or both endpoints of an inner wall segment are close enough to any part
of the outer wall loop, attach that endpoint to the outer wall loop.

Threshold:

```txt
endpoint-to-outer-wall distance < 500 mm
```

Use the estimated/resolved architectural scale for this threshold. Do not add
a pixel fallback path for this rule. The vectorization process should move
toward millimeter-space topology rather than pixel-space geometry.

Suggested config fields:

```yaml
walls:
  inner_attach_outer_threshold_mm: 500
```

If scale cannot be estimated at all, the implementation must mark the sample as
scale-blocked in metrics and avoid silently substituting arbitrary pixel
thresholds for architectural rules.

When an endpoint is near the outer wall:

```txt
project endpoint onto nearest outer wall segment
move endpoint to projected point
create clean T-junction or endpoint contact
do not move the outer wall loop itself
```

## Usually-True Wall Rules With Exceptions

These should be strong biases, not absolute rejections:

```txt
at least one end of most inner wall segments should connect to the outer wall loop
walls are usually 90 degrees to each other
```

Allowed exceptions:

```txt
short freestanding wall stubs if source evidence clearly ends
islands/partial partitions if source evidence supports them
explicit diagonal walls if the evidence is clear
an inner wall ending at a door/window opening boundary instead of another wall
```

Ambiguous geometry should prefer orthogonal output.

An inner wall segment that terminates at a door or window opening (rather
than at another wall or the outer loop) is a normal, valid ending - this is
expected because many interior walls sit a short distance from an opening.
It is not unresolved/incomplete and does not need the 500mm outer-wall
attachment rule applied to that end.

## Door Detection Rule

Door generation must be driven by red door-arc pixels.

Required behavior:

```txt
count connected red door_arc pixel groups
door count is determined solely from red door_arc groups
door location is determined by each red door_arc group
for each red group, infer one door primitive set
if no red door_arc group exists, assume there is no door
```

Do not create doors from purple door_origin evidence alone. Do not create doors
from orange door_leaf evidence alone. Red door_arc connected components are the
standard for door number and door location.

Suggested config:

```yaml
doors:
  require_arc_group: true
  min_door_arc_component_area: 4
```

## Door Hinge Detection Rule

The preferred hinge point is the exact or near-exact intersection between:

```txt
orange door_leaf pixels
purple door_origin pixels
```

Then snap that hinge point to the nearest wall, outer or inner.

Procedure:

```txt
1. For each red door_arc group, find nearby orange door_leaf and purple door_origin evidence.
2. Detect the intersection or overlap/near-overlap between orange and purple evidence.
3. Use that point as the hinge candidate.
4. Snap hinge candidate to nearest wall segment.
5. Pair this snapped hinge with the hosted door origin.
```

Strict fallback:

```txt
if exact/near orange-purple intersection is missing:
  infer hinge from the red arc group geometry and nearest wall
  use the arc's local center/endpoint geometry as the hinge candidate
  constrain the candidate to the nearest plausible host wall
  snap the inferred hinge to that wall
```

Red pixels remain the core standard. Hinge inference is allowed only inside the
spatial neighborhood of a red door_arc group.

Suggested config:

```yaml
doors:
  hinge_intersection_tolerance_px: 6
  hinge_snap_to_wall_max_dist_px: 40
  hinge_arc_inference_enabled: true
```

## Door Pairing Rule

The debug/geometry representation should treat:

```txt
orange hinge marker
purple door-origin far/end marker
```

as a required pair.

Rules:

```txt
orange square and purple circle should always be paired
if both are detected, one door is generated from that pair
do not generate an unpaired orange marker as a door
do not generate an unpaired purple marker as a door
unpaired evidence should go to debug/unresolved output
```

## Door Module Length Rule

For each orange-square / purple-circle pair:

```txt
fix the orange square / hinge position
preserve the detected orientation
adjust the purple point distance from the hinge
snap the door origin/leaf length to one of:
700 mm
900 mm
```

The door origin segment and door leaf segment should use the same snapped
length.

Use architectural scale in millimeters for this rule. Do not add a pixel module
fallback. If the scale resolver cannot estimate a valid px-to-mm factor, record
the door as unresolved/scale-blocked rather than generating a pixel-sized door.

Suggested config:

```yaml
doors:
  door_width_modules_mm: [700, 900]
```

## Window Minimum Length Rule

A window's hosted width must be at least 300 mm. Use the estimated/resolved
architectural scale for this check, the same as the door module rule - do not
add a pixel-only fallback minimum. If scale cannot be estimated, record the
window as scale-blocked rather than silently accepting an arbitrarily small
pixel width.

Suggested config:

```yaml
windows:
  min_width_mm: 300
```

## Opening-Near-Corner Host Selection Rule

A window or door opening is always generated as exactly one door/window
hosted on exactly one wall - never split or straddled across two walls, and
never reduced to a degenerate zero-length wall stub on either side.

When an opening's evidence sits near where two wall segments meet (e.g. near
a corner, or near a T-junction), and splitting the more obvious host wall
would leave a near-zero-length wall remainder on one side:

```txt
evaluate both candidate host walls near the opening
pick the host wall with the higher hosting probability (more overlapping/
nearby evidence, better alignment of the opening's orientation with the
wall's orientation, larger remaining wall length after the split)
push/project the opening fully onto that chosen wall
generate the door or window on that wall only
```

This keeps wall splitting (see "Wall Splitting After Openings") from ever
needing to special-case a zero-length side: the opening is fully owned by
one wall before splitting happens, so both resulting wall remainders are
real, non-degenerate segments (or the opening sits flush against the wall's
true end, leaving no remainder on that side at all, which is expected, not
an error).

## Door-Wall Endpoint Rule

The two ends of every door origin segment must connect to ends of black wall
segments.

Required behavior:

```txt
door origin replaces a wall span
wall is split at both door-origin endpoints
the two resulting wall segment ends must coincide with the two door-origin endpoints
```

This applies to both outer and inner host walls.

## Window-Wall Endpoint Rule

The two ends of every window line segment must connect to ends of black wall
segments.

Required behavior:

```txt
window replaces a wall span
wall is split at both window endpoints
the two resulting wall segment ends must coincide with the two window endpoints
```

This applies to both outer and inner host walls.

## Wall Splitting After Openings

Wall splitting must happen after:

```txt
outer walls are created
inner walls are created
wall endpoints are snapped/connected
windows are hosted
doors are hosted from red arc groups
door/window endpoints are projected onto host walls
```

Then split wall segments at every hosted:

```txt
window interval
door-origin interval
```

The final wall topology must expose wall endpoints at the opening boundaries.

## Implementation Direction

Update:

```txt
src/vectorization/wall_extraction.py
src/vectorization/door_extraction.py
src/vectorization/window_extraction.py
src/vectorization/geometry_rules.py
src/vectorization/primitives/scale.py
src/vectorization/run_mask_to_vector.py
configs/vectorization_v008.yaml
tests/test_vectorization_v008.py
```

Recommended implementation steps:

```txt
1. Replace whole-component outer-wall erasing with spatial outer-band masking.
2. Build an explicit inner-wall candidate mask inside the outer loop.
3. Preserve interior wall branches connected to the exterior wall.
4. Add endpoint projection/snap from inner walls to the nearest outer wall segment.
5. Ensure inner walls are included in the host-wall pool before window/door hosting.
6. Change door extraction to be arc-group-led.
7. Detect hinge from orange/purple intersection near each red arc group.
8. Snap hinge to nearest wall.
9. Pair hinge/origin evidence; unresolved unpaired evidence stays debug-only.
10. Snap door origin/leaf length to 700/900 mm using architectural scale.
11. Split host walls at window and door-origin endpoints so black wall ends meet blue/purple opening ends.
12. Expand debug overlay to distinguish unresolved, paired, snapped, and module-adjusted door evidence.
```

## Required Tests

Add or update tests for:

1. Connected inner wall branches touching an outer wall are not erased.
2. Inner walls inside the outer loop are extracted when an outer loop exists.
3. Only the spatial outer-wall band is removed from inner-wall candidates.
4. Inner wall endpoint within 500 mm of outer loop snaps to the outer loop.
5. Inner wall endpoint beyond the attachment threshold remains unchanged or unresolved.
6. Window endpoints coincide with adjacent wall segment endpoints after splitting.
7. Door-origin endpoints coincide with adjacent wall segment endpoints after splitting.
8. Red door_arc connected components determine door count.
9. Purple door_origin evidence without a matching red arc never creates a door.
10. No red door_arc group means no door is generated.
11. Hinge point is selected from orange/purple intersection before wall snapping when that intersection exists.
12. Missing orange/purple intersection falls back to red-arc-plus-nearest-wall hinge inference.
13. Hinge snaps to nearest outer or inner wall.
14. Orange/purple debug markers are paired; unpaired evidence is debug-only.
15. Door origin and leaf lengths snap to 700/900 mm when scale is estimated/resolved.
16. If scale cannot be estimated, metric-dependent vectorization records scale-blocked output instead of using arbitrary pixel fallback.
17. Orthogonal wall snapping remains preferred for ambiguous wall evidence.
18. Existing outer wall loop detection remains correct.
19. Existing window-to-wall and door-to-wall hosting behavior remains correct.

## Acceptance Criteria

This task is complete when:

1. Outer wall loop detection remains at least as good as the current behavior.
2. Inner walls are generated as individual segments inside the outer loop.
3. Inner wall branches connected to the outer wall are preserved, not erased.
4. Inner wall endpoints attach to the outer wall loop when within 500 mm.
5. Window endpoints connect exactly to black wall segment endpoints.
6. Door-origin endpoints connect exactly to black wall segment endpoints.
7. Door count and location are driven by red door_arc groups.
8. If no red door_arc group exists, no door is generated.
9. Door hinge candidates are detected from orange/purple intersection before snapping when possible.
10. If orange/purple intersection is absent, hinge is inferred from red arc geometry and nearest wall.
11. Door hinge points snap to nearest wall topology.
12. Orange/purple door markers form required pairs.
13. Door origin and leaf lengths snap to 700/900 mm when scale is available.
14. Unpaired or ambiguous evidence appears only in debug output, not as final doors.
15. No architectural rule silently falls back to arbitrary pixel geometry.
16. Walls remain strongly biased toward 90-degree relationships.

## Resolved Clarifications

These decisions are strict for this task:

```txt
door width modules are exactly 700 mm and 900 mm
600 mm and 800 mm are not valid door modules for this task
red door_arc connected components are the sole standard for door count and location
if red door_arc pixels are missing, assume there is no door
if orange/purple hinge intersection is missing, infer hinge from red arc geometry and nearest wall
all vectorization topology and snapping rules should use estimated/resolved millimeter scale
do not add arbitrary pixel fallback geometry for architectural rules
window minimum hosted width is 300 mm (architectural-scale, no pixel-only fallback)
an opening near a wall corner is never split/straddled across two walls or split into a
  zero-length wall stub - it is pushed fully onto whichever of the two candidate host
  walls has the higher hosting probability, then generated on that wall only
inner-wall candidate mask = black wall pixels union purple door_origin pixels (not
  door_arc/door_leaf/window); the resulting inner wall is trimmed to fit the door the
  same way the outer wall is already trimmed to fit a door
an inner wall ending at a door/window opening boundary (instead of another wall or the
  outer loop) is a normal, valid ending, not unresolved/incomplete
debug overlay: orange square = hinge marker, purple circle = door-origin far/end marker;
  paired evidence drawn solid, unpaired/unresolved evidence drawn hollow/dimmed
final vector.svg never contains unresolved/unpaired evidence - that stays confined to
  debug_overlay.png and metrics.json
```
