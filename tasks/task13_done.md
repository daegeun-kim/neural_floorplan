# Task 13 - Make Red Door-Arc Clusters Guaranteed Door Objects

## Objective

Fix the current door recognition logic so every connected red `door_arc` pixel cluster is recognized as a door object.

The current failure is that red clusters are being used for scale but are still rejected later when purple/orange pairing or endpoint checks fail. This is backwards.

New rule:

```txt
if a connected red door_arc cluster exists, it is a door, always.
```

After a red cluster creates a door object, the vectorizer must infer the door's hinge and end points from nearby semantic evidence. Purple/orange/black evidence refines the door geometry but must not decide whether the red cluster is a door.

## Source Of Truth

Use:

```txt
specs/spec_v008_phase3_mask_to_vector.md
tasks/task12.md
```

This task supersedes the part of the current implementation that rejects red clusters as failed door evidence.

## Required Door Recognition Process

For each connected red `door_arc` cluster:

```txt
1. Create one door candidate immediately.
2. Use the red cluster for door count.
3. Use the red cluster for door location.
4. Use the red cluster bounding box for scale inference.
5. Infer one wall-door hinge point.
6. Infer one wall-door end point.
7. Generate door origin, door leaf, and door arc from those two points.
```

A red cluster should only be removed before door creation if it is below the configured minimum red-component area.

## Scale Rule

For each red cluster:

```txt
red_cluster_bbox_long_edge_px = max(bbox_width, bbox_height)
```

Infer scale from the red cluster bbox long edge.

Allowed door modules:

```txt
700 mm
900 mm
```

Evaluate both modules and choose the globally most consistent scale across red clusters.

Once scale is inferred, all other metric rules follow that scale.

## Wall-Door Hinge Point Inference

Each red door object must have exactly one `wall_door_hinge_point`.

Search for the hinge point near the red cluster.

Preferred evidence:

```txt
red pixels
orange door_leaf pixels
purple door_origin pixels
black wall pixels
```

The hinge point should be the point with the highest combined proximity to all four pixel types:

```txt
[red, orange, purple, black]
```

If one or more of those four pixel types is missing near the red cluster, use the point with the highest proximity to the largest available subset.

Rules:

```txt
prefer candidates supported by all 4 classes
if all 4 are unavailable, prefer candidates supported by 3 classes
if 3 are unavailable, prefer candidates supported by at least 2 classes
force a hinge point if a red cluster exists and plausible nearby wall evidence exists
```

The hinge point must be near the red cluster bounding box using the metric threshold from `task12.md`:

```txt
maximum distance from red bbox = 200 mm
```

## Wall-Door End Point Inference

Each red door object must have exactly one `wall_door_end_point`.

Search for the end point near the same red cluster.

Preferred evidence:

```txt
red pixels
purple door_origin pixels
orange door_leaf pixels
```

The end point should be the point with the highest combined proximity/intersection to:

```txt
[red, purple, orange]
```

If one of the three pixel types is missing near the red cluster, use the point with the highest support from the remaining available types.

Rules:

```txt
prefer candidates supported by all 3 classes
if all 3 are unavailable, prefer candidates supported by at least 2 classes
force an end point if a red cluster exists
```

The end point must be near the red cluster bounding box:

```txt
maximum distance from red bbox = 200 mm
```

## Forceful Inference Rule

Red evidence is authoritative.

If a red cluster exists, do not reject the door because:

```txt
purple evidence is fragmented
purple evidence is too short
orange evidence is fragmented
orange/purple intersection is imperfect
the snapped 700/900 mm endpoint is slightly different from noisy purple pixels
```

Instead:

```txt
create the door
infer the best hinge point
infer the best end point
record confidence/debug details in metrics
```

The final door may have lower confidence, but it must still exist.

## Rejected Red Evidence

A red cluster may become rejected evidence only if:

```txt
the red component is below minimum area
or no plausible wall evidence exists anywhere near it
```

Do not reject red evidence only because door-origin or door-leaf evidence is weak.

## Metrics Requirements

Add or update metrics so each red cluster reports:

```txt
red_component_id
red_bbox
red_bbox_long_edge_px
created_door_candidate: true/false
scale_candidate_px_to_mm
hinge_candidate_support_classes
end_candidate_support_classes
hinge_distance_to_red_bbox_mm
end_distance_to_red_bbox_mm
door_confidence
door_inference_notes
```

The metrics should make it obvious whether a red cluster became a door and how its hinge/end points were inferred.

## Debug Overlay Requirements

The debug overlay should show:

```txt
red door candidate bbox
inferred wall_door_hinge_point
inferred wall_door_end_point
supporting nearby red/orange/purple/black evidence
low-confidence inferred points differently from high-confidence points
```

Do not put low-confidence or rejected evidence in `vector.svg`. It belongs in debug overlay and metrics.

## Implementation Targets

Likely files:

```txt
src/vectorization/point_detection.py
src/vectorization/scale.py
src/vectorization/primitives/scale.py
src/vectorization/debug.py
src/vectorization/run_mask_to_vector.py
configs/vectorization_v008.yaml
tests/test_vectorization_v008.py
```

Do not modify CNN training code for this task.

## Required Tests

Add or update tests for:

1. Every accepted red `door_arc` connected component creates one door candidate.
2. Door count equals accepted red cluster count.
3. Red cluster bbox long edge participates in scale inference.
4. Red cluster is not rejected because purple evidence is fragmented.
5. Red cluster is not rejected because orange evidence is fragmented.
6. Red cluster with red/orange/purple/black nearby creates a hinge point.
7. Hinge point prefers combined proximity to red/orange/purple/black.
8. If one hinge evidence class is missing, hinge uses the strongest available subset.
9. End point prefers combined proximity to red/purple/orange.
10. If one end evidence class is missing, end uses the strongest available subset.
11. Door hinge and end points are within 200 mm of the red bbox.
12. Metrics report one record per red cluster.
13. Debug overlay identifies each red cluster as a door candidate.

## Acceptance Criteria

This task is complete when:

```txt
connected red clusters are guaranteed door objects
door count equals accepted red cluster count
red clusters drive scale, location, hinge/end search, and door generation
weak purple/orange evidence lowers confidence but does not delete the door
wall_door_hinge_point and wall_door_end_point are forcefully inferred when needed
red clusters no longer appear mostly as rejected evidence
```

