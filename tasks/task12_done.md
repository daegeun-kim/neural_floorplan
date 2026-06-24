# Task 12 - Fix Door-Based Scale and Point Recognition

## Objective

Improve the current v008 point-graph vectorization by fixing two major failure areas:

```txt
1. scale estimation
2. point recognition
```

The current output recognizes too many `1_wall_point` endpoints and treats red door-arc evidence too weakly. Red door-arc clusters should become the strongest evidence for both scale and door point inference.

Use the active spec as context:

```txt
specs/spec_v008_mask_to_vector.md
```

## 1. Scale Rule Update

The most accurate scale source should be the red `door_arc` class.

### Required Behavior

For each connected red `door_arc` pixel cluster:

```txt
1. Find the connected component.
2. Compute its bounding box.
3. Take the long edge of the bounding box.
4. Treat that long edge as a candidate standard door width.
5. Evaluate both allowed modules:

   700 mm
   900 mm

6. Estimate px_to_mm candidates from:

   px_to_mm = 700 / red_cluster_bbox_long_edge_px
   px_to_mm = 900 / red_cluster_bbox_long_edge_px
```

When multiple red clusters exist, compute candidate scales per cluster and module, then combine them robustly.

Recommended aggregation:

```txt
cluster or vote over px_to_mm candidates
choose the scale supported by the most red clusters
use median of the winning candidate group
```

Reject only obvious outliers.

### Scale Priority

The updated scale priority should be:

```txt
1. explicit metadata if available
2. red door_arc cluster bounding-box long edge = 700 mm
3. door_origin evidence as secondary check/debug evidence
4. wall thickness as weak secondary check/debug evidence
5. unknown scale only if no usable red door_arc cluster exists
```

Do not let noisy wall connected-component thickness override red door-arc scale.

### Required Metrics

Add scale diagnostics to `metrics.json`:

```txt
red_arc_bbox_long_edges_px
red_arc_px_to_mm_candidates
red_arc_selected_modules_mm
selected_px_to_mm
scale_source
scale_rejected_outliers
```

## 2. Point Recognition Rule Update

The current point recognition creates too many `1_wall_point` endpoints. Tighten the conditions for each point type.

## 2.1 `1_wall_point` Rule

A `1_wall_point` is a true free-standing end of a wall.

It should be detected only when:

```txt
the black wall pixel region ends like a peninsula
and the end is surrounded only by background/floor pixels
and no window or door evidence touches or sits immediately near that wall end
```

In mask-color terms:

```txt
valid surrounding classes near a 1_wall_point:
- background / grey
- floor / white or off-white

invalid surrounding classes near a 1_wall_point:
- window / blue
- door_arc / red
- door_leaf / orange
- door_origin / purple
```

If a black wall endpoint touches or is near blue/orange/purple/red evidence, it should not become `1_wall_point`.

## 2.2 `wall_window_point` Rule

If black wall pixels touch blue window pixels, the endpoint should be considered a `wall_window_point`, not a `1_wall_point`.

Required behavior:

```txt
if wall evidence touches window evidence:
  detect wall_window_point at the transition
```

Each final window should still have exactly two compatible `wall_window_point` endpoints.

## 2.3 `wall_door_hinge_point` Rule

If black wall pixels touch or nearly touch orange and purple door evidence near a red door-arc cluster, detect a `wall_door_hinge_point`.

Required behavior:

```txt
if wall evidence is near red door_arc evidence
and nearby orange door_leaf evidence exists
and nearby purple door_origin evidence exists or can be inferred:
  detect wall_door_hinge_point
```

Red `door_arc` evidence is the strongest driver for door recognition. Orange and purple pixels refine the hinge/end location.

## 2.4 `wall_door_end_point` Rule

For every accepted red `door_arc` cluster, infer one matching `wall_door_end_point`.

The hinge point and end point must be spatially tied to the red cluster.

Required distance rule:

```txt
wall_door_hinge_point must be within 200 mm of the red door_arc cluster bounding box
wall_door_end_point must be within 200 mm of the red door_arc cluster bounding box
```

Scale inference must run before point recognition so this threshold can be applied in millimeters. This does not mean the point must be inside the red bbox. It may lie just outside it, but it must be within 200 mm of the bbox boundary.

## 2.5 Red Door-Arc Evidence Rule

All red `door_arc` connected components must be treated as primary evidence.

Current failure:

```txt
red pixels are being counted mostly as rejected evidence
```

Required behavior:

```txt
red door_arc clusters should drive:
- scale estimation
- door existence
- door count
- hinge/end search area
- door origin inference
- door leaf/arc generation
```

Rejected red evidence should only happen when:

```txt
the red cluster is below minimum area
or no plausible nearby wall exists
or no plausible door geometry can be inferred even after fallback
```

Do not reject red evidence merely because purple/orange evidence is noisy.

## 3. Implementation Targets

Likely files to update:

```txt
src/vectorization/scale.py
src/vectorization/primitives/scale.py
src/vectorization/point_detection.py
src/vectorization/debug.py
src/vectorization/run_mask_to_vector.py
configs/vectorization_v008.yaml
tests/test_vectorization_v008.py
```

Do not change unrelated CNN training code.

## 4. Required Tests

Add or update tests for:

1. Red `door_arc` bbox long edge resolves scale as `700 mm`.
2. Red `door_arc` bbox long edge can also resolve scale as `900 mm`.
3. Multiple red clusters use robust candidate voting/median scale.
4. Red scale beats conflicting wall-thickness evidence.
5. `1_wall_point` is detected only when surrounded by background/floor.
6. Wall endpoint touching blue becomes `wall_window_point`, not `1_wall_point`.
7. Wall endpoint near red/orange/purple becomes door-related point, not `1_wall_point`.
8. Red door clusters are not rejected only because purple evidence is noisy.
9. Each accepted red cluster produces one hinge/end pair when plausible wall evidence exists.
10. `wall_door_hinge_point` lies within 200 mm of the red cluster bbox.
11. `wall_door_end_point` lies within 200 mm of the red cluster bbox.
12. Metrics include red-arc scale diagnostics.

## 5. Acceptance Criteria

This task is complete when:

```txt
scale is estimated primarily from red door_arc bbox long edges
wall thickness no longer dominates or breaks scale when red evidence exists
1_wall_point count is reduced to true free wall ends
wall-window contacts become wall_window_point
door-adjacent wall ends become wall_door_hinge_point / wall_door_end_point
red door_arc clusters are primary accepted evidence, not mostly rejected evidence
debug metrics clearly explain scale candidates and rejected red clusters
```

## 6. Clarification Questions

Resolved decisions from user clarification:

1. Red door-arc scale inference must allow both `700 mm` and `900 mm`.
2. Scale inference runs before point recognition; door hinge/end proximity to the red bbox should use `200 mm`, not `10 px`.
3. The neighborhood radius for deciding whether a `1_wall_point` is surrounded only by background/floor is left to Claude's implementation judgment.
