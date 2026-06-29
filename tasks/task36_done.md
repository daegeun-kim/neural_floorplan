# Task 36 - Double-Swing Door Detection And Shared-Origin Rendering

## Context

Task35 changed door direction scoring so the swing side is inferred from red
door-arc pixels on either side of the adjusted door origin segment. This fixes
ordinary single-swing doors, but it still assumes every detected door must choose
exactly one opening side.

Some floorplans contain doors that open to both sides from the same origin edge.
In the 7-class segmentation, this can appear as red door-arc evidence on both
sides of the same purple/origin segment. The current source cannot represent
this case because it forces a single side.

This is not always an error. When two door detections exist side by side along
the same door origin and their evidence indicates opposite swing sides, the
correct vector result may be one shared-origin double-swing door:

```txt
one door origin edge
one fixed hinge/origin relationship
leaf + arc on side A
leaf + arc on side B
```

## Objective

Extend Phase 4 vectorization to classify and render doors with evidence on both
sides of the origin segment.

The pipeline must decide between:

```txt
1. ignore one weak/duplicate door detection
2. keep two separate single-swing doors
3. merge into one double-swing shared-origin door
```

This task must update both:

```txt
src/...
notebooks/phase4_vectorization.ipynb
```

The notebook is the manual testing surface and must expose the updated behavior.

## Part A - Detect Two-Sided Door Evidence

For every finalized door origin segment, keep the Task35 red-side metrics:

```txt
red_side_positive_count
red_side_negative_count
red_side_selected
```

Add a two-sided evidence check:

```txt
positive side is supported if red_side_positive_count >= min_side_pixels
negative side is supported if red_side_negative_count >= min_side_pixels
two_sided_red_evidence = positive_supported and negative_supported
```

Use a ratio guard so a tiny amount of noise on the weaker side does not create
a false double-swing door:

```txt
weaker_side / stronger_side >= min_double_swing_ratio
```

Suggested initial thresholds:

```txt
min_side_pixels: tune from existing door masks, start with 8-15 pixels
min_double_swing_ratio: start with 0.25-0.40
```

These thresholds should be constants or config values, not magic numbers hidden
inside one scoring expression.

## Part B - Detect Paired Door Components Along The Same Origin

Two separate door detections may represent one double-swing door if they share
the same wall-hosted origin segment.

Consider two doors as a double-swing candidate when:

1. Their adjusted origin segments are collinear and overlap strongly.
2. Their snapped wall endpoints are nearly identical, or one segment is almost
   contained in the other.
3. Their hinge endpoint evidence selects the same physical hinge/origin endpoint,
   or their hinge endpoints are within a small tolerance.
4. Their red evidence selects opposite sides of the same origin line.
5. Their purple/origin evidence belongs to the same local origin region or
   strongly overlaps.

Do not merge doors just because they are nearby. They must share the origin edge
relationship.

## Part C - Classification Decision

For each possible two-sided or paired-door case, classify into one of:

```txt
single_swing
double_swing_shared_origin
separate_single_swing_doors
ignored_duplicate
```

Decision guidance:

- If one side has strong red/orange evidence and the other side is weak noise,
  keep `single_swing`.
- If two detections have nearly the same origin but one has much lower evidence
  and does not add a real opposite-side arc, mark the weaker one as
  `ignored_duplicate`.
- If two detections share the same origin edge and have opposite-side red arc
  evidence with comparable support, merge them into
  `double_swing_shared_origin`.
- If two detections are adjacent along a wall but their origin segments are not
  the same, keep them as `separate_single_swing_doors`.

The decision should be based on the score metrics, not on hard-coded sample
coordinates.

## Part D - Double-Swing Geometry

Add a geometry representation that can express one origin segment with arcs and
leaves on both sides.

Required geometry fields:

```txt
door_type: "single_swing" | "double_swing_shared_origin"
origin_p0
origin_p1
hinge
swing_sides: ["positive", "negative"] for double-swing
leaf_end_positive
leaf_end_negative
arc_positive
arc_negative
source_door_ids
```

For a double-swing door:

- Render only one shared door origin edge.
- Render one leaf and one arc on each side.
- The two leaves/arcs should be symmetric around the shared origin segment
  unless the raster evidence strongly indicates different sizes.
- The final wall trim should still have only one opening gap for the shared
  origin edge, not two overlapping gaps.

## Part E - SVG, JSON, And Debug Overlay

Update SVG output so double-swing doors draw correctly:

```txt
origin edge once
leaf + arc on side A
leaf + arc on side B
```

Update `final_vector.json` to record:

```txt
door_type
classification
source_door_ids
red_side_positive_count
red_side_negative_count
double_swing_ratio
double_swing_decision_reason
merged_from_doors
ignored_duplicate_ids
```

Update debug overlay so double-swing decisions are visible:

- label double-swing doors distinctly
- show both selected swing sides
- show merged/ignored source detections where practical

## Part F - Interval Editing

Double-swing merged doors must not create duplicate or overlapping wall trim
intervals.

When two detections are merged into one shared-origin double-swing door:

1. Compute one final origin interval.
2. Trim the wall once at that interval.
3. Render both door swings from the same interval.

The interval de-overlap logic from Task33/Task34 must still run against the
merged opening set.

## Part G - Tests

Add or update tests for:

1. Red evidence on only one side remains `single_swing`.
2. Weak noise on the opposite side does not become double-swing.
3. Strong red evidence on both sides of one origin becomes
   `double_swing_shared_origin`.
4. Two door detections with the same origin and opposite sides merge into one
   double-swing door.
5. Two nearby but different origin segments remain separate doors.
6. Merged double-swing doors create one wall trim interval.
7. SVG output draws one origin edge and two leaves/arcs.
8. JSON records source door ids and classification reason.
9. Notebook summary exposes double-swing count.

## Part H - Notebook Update

Update `notebooks/phase4_vectorization.ipynb` so it uses the new source pipeline
and reports:

```txt
door_count
single_swing_door_count
double_swing_door_count
ignored_duplicate_door_count
door_evidence_fallbacks
```

The notebook should continue generating the existing Phase 4 outputs:

```txt
input.png
image_segmentation.png
image_debug_overlay.png
graph_pred.svg
graph_pred.json
graph_overlay.png
graph_overlay_orthogonal.png
final_vector.svg
final_vector.json
```

## Acceptance Criteria

- Phase 4 no longer forces every door to choose exactly one swing side.
- Valid two-sided door evidence can become one double-swing shared-origin door.
- Duplicate weak detections can be ignored without deleting valid separate doors.
- Nearby doors are merged only when they share the same origin edge relationship.
- Double-swing SVG uses one origin edge with leaves/arcs on both sides.
- Wall trimming uses one interval for a merged double-swing door.
- `final_vector.json` and debug overlay explain the classification decision.
- `notebooks/phase4_vectorization.ipynb` is updated with the new behavior.
- Phase 4 tests pass.
