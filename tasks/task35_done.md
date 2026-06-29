# Task 35 - Red-Side Door Direction Scoring And Flat-Ended Opening Rendering

## Context

Task34 improved Phase 4 vectorization, but two issues still remain in sample outputs:

1. Door swing direction and hinge endpoint are still sometimes wrong.
2. Window and wall segments still overlap slightly at both window ends.

The current door direction scoring is still too dependent on generated primitive hypotheses:

```txt
red door_arc mask      -> sampled along a generated 90 degree arc
orange door_leaf mask  -> sampled along a generated leaf line
```

This misses stronger evidence already present in the 7-class segmentation raster. The raster output is reliable for doors: red arc/fill pixels, orange leaf pixels, and purple origin pixels are usually clear.

The window overlap appears to be caused by rendering/buffering that increases segment length as well as width. Opening segments must get thicker only perpendicular to the segment, not longer along the segment.

## Objective

Update Phase 4 vectorization so door direction is inferred directly from segmentation raster evidence, and rendered opening segments have flat endpoints that do not extend beyond their adjusted trim nodes.

This task must update both:

```txt
src/...
notebooks/phase4_vectorization.ipynb
```

The notebook is the manual test surface and must call the updated source pipeline.

## Part A - Door Swing Side From Red Pixels

Replace the current primary swing-side scoring with a red-pixel side test.

For each door after its two wall-attached origin points are finalized:

1. Let `p0` and `p1` be the adjusted door origin segment endpoints.
2. Use the associated local red `door_arc` component mask, cropped to the door's component bbox plus a small margin.
3. For every red pixel in that local component, compute which signed side of the line `p0 -> p1` it lies on.
4. Count or area-score red pixels on both sides.
5. Select the swing side with the larger red-pixel support.

Important requirements:

- The side score must use the original 7-class segmentation raster evidence, not only generated arc samples.
- Use component-local red evidence or a tightly expanded bbox so nearby doors do not contaminate the score.
- Account for image coordinates where `y` increases downward.
- Ignore pixels too close to the origin line only if they create instability; document the tolerance if used.
- Arc-sampling may remain only as a secondary tie-breaker or debug metric, not the primary decision.

## Part B - Hinge Endpoint From Orange Pixels

Use orange `door_leaf` pixels as the primary hinge endpoint evidence.

After choosing the swing side from red pixels:

1. Compare support for `p0` as hinge vs `p1` as hinge.
2. Use orange-pixel proximity to each endpoint and/or orange support inside candidate leaf bands.
3. Pick the hinge endpoint whose candidate leaf is better supported by orange pixels.

Suggested scoring:

```txt
hinge_score(endpoint) =
  weighted orange pixels near endpoint
  + orange pixels inside a narrow candidate leaf corridor from endpoint toward selected swing side
```

Notes:

- The leaf corridor should be based on the door opening side chosen from red pixels.
- Orange pixels should be taken from the local door region when possible.
- Purple `door_origin` pixels should validate the origin segment and may help break ties, but should not override strong red/orange evidence.
- Fall back only when red and orange evidence are genuinely absent or ambiguous.

## Part C - Door Debug Data

Add explicit evidence fields to `final_vector.json` and debug overlay output.

At minimum, record per door:

```txt
red_side_positive_count
red_side_negative_count
red_side_selected
orange_hinge_p0_score
orange_hinge_p1_score
hinge_selected
swing_source
hinge_source
fallback_used
```

The debug overlay should make it possible to inspect:

- which side red evidence selected
- which endpoint was selected as hinge
- whether fallback was used

## Part D - Flat-Ended Window And Door-Origin Rendering

Opening segments must not extend past their adjusted trim endpoints.

Fix all rendering or geometry generation that lengthens a window or door-origin segment along its own direction.

Requirements:

- Window primitives must have flat endpoints.
- Door origin primitives must have flat endpoints.
- If SVG lines are used, use `stroke-linecap="butt"` for these hosted opening segments.
- Do not use `stroke-linecap="square"` or `stroke-linecap="round"` for hosted opening segments that must align to wall trim endpoints.
- Prefer explicit rectangle/polygon geometry if it is clearer:
  - rectangle centerline endpoints must equal the adjusted opening endpoints
  - width expands only perpendicular to the segment
  - length must not exceed the endpoint-to-endpoint distance

This is especially important when a window is very close to a door: the final window primitive and wall trim gap must share the same exact endpoints after de-overlap adjustment.

## Part E - Tests

Add or update Phase 4 tests for:

1. Red pixels on one side of a door origin segment choose that swing side.
2. Orange pixels closer to `p0` select `p0` as hinge.
3. Orange pixels closer to `p1` select `p1` as hinge.
4. Nearby red pixels from another door do not affect the local door decision.
5. Arc-sampling cannot override strong red-side evidence.
6. Window SVG/geometry does not extend beyond its adjusted endpoints.
7. Door-origin SVG/geometry does not extend beyond its adjusted endpoints.
8. Door/window trim endpoints exactly match the final rendered opening endpoints.

## Part F - Notebook Update

Update `notebooks/phase4_vectorization.ipynb` so running the notebook uses the new behavior.

The notebook output should still generate:

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

The notebook summary should expose the new door evidence metrics where practical, especially fallback counts.

## Acceptance Criteria

- Door opening side is selected primarily by red-pixel side count from the 7-class segmentation.
- Door hinge endpoint is selected primarily by orange leaf evidence.
- Existing generated-arc scoring is secondary only.
- Window and door-origin visual segments have flat ends and do not lengthen beyond adjusted trim endpoints.
- Final SVG, final JSON, wall trim gaps, and debug overlay all use the same adjusted opening endpoints.
- `notebooks/phase4_vectorization.ipynb` runs through the updated source pipeline.
- Phase 4 tests pass.
