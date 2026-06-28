# Task 34 - Enforce Phase 4 Pipeline Contract In SVG/JSON/Notebook

## Objective

Fix the remaining Phase 4 vectorization contract violations observed after Task 33.

The first graph-to-vector implementation is promising, but three issues show that the code path is not obeying the strict step-by-step process:

```txt
1. wall trim moved after interval adjustment, but window/door primitives did not move
2. door hinge/swing direction still uses fallback instead of red/orange/purple raster evidence
3. wall corners are still sometimes disconnected before buffering
```

This task is not a small visual patch. It must make the source pipeline, final SVG, final JSON, and notebook all follow the same authoritative geometry state.

## Source Spec

Follow:

```txt
specs/spec_v008_phase4_vectorization.md
```

Important sections:

```txt
## 7. Door Generation
## 9. Opening Interval Editing
## 10. Wall Polygon Generation
## 15. Notebook Contract
```

## Non-Negotiable Rule

The vectorization process must follow the strict sequence:

```txt
1. detect/opening endpoints
2. host endpoints on graph
3. resolve opening interval overlap
4. insert adjusted opening nodes
5. trim wall intervals at adjusted opening nodes
6. replace the trimmed wall interval with the final door/window primitive
7. connect remaining wall graph chains
8. buffer connected chains into final walls
9. export SVG/JSON/notebook output from the same adjusted geometry state
```

If the wall trim uses adjusted endpoints but the window/door primitive still uses old endpoints, the implementation is wrong.

## Part A - Propagate Adjusted Opening Geometry To Final Primitives

### Current Failure

Task 33 adjusts intervals inside `trim_wall_intervals()`.

Example from sample `1316`:

```txt
window source_component_id=3
original interval moves to adjusted interval
wall trim uses adjusted interval
```

But `final_vector.svg` still draws the window from the original `HostedOpening.snapped_points`.

This creates:

```txt
wall moved
window not moved
```

That violates the core rule:

```txt
trimmed wall interval must be replaced by the window/door segment at the same adjusted endpoints
```

### Required Fix

After interval resolution, create an authoritative adjusted opening list.

Required concept:

```txt
HostedOpening
-> interval conflict resolution
-> AdjustedOpening / adjusted HostedOpening
-> wall trimming
-> final window/door SVG
-> final_vector.json
-> debug overlay
```

`build_final_svg()` and `build_final_vector_json()` must use adjusted openings, not the pre-adjustment `hosted_doors` / `hosted_windows`.

Acceptable implementation options:

```txt
Option A:
  create adjusted HostedOpening objects with updated snapped_points/width

Option B:
  create a new FinalOpening dataclass used by SVG/JSON/debug export

Option C:
  add a clear resolver function:
    apply_adjusted_intervals_to_hosted_openings(trimmed_graph, hosted_openings, aligned_edges)
```

Whichever approach is used, there must be exactly one final geometry source of truth after interval adjustment.

### Required Output Behavior

For each adjusted opening:

```txt
wall trim endpoints == final primitive endpoints
```

For windows:

```txt
final window line endpoints == adjusted opening gap endpoints
```

For doors:

```txt
door_origin endpoints == adjusted opening gap endpoints
door_leaf and door_arc use the adjusted door_origin endpoints
```

## Part B - Door Direction Must Use 7-Class Raster Evidence

### Current Failure

Door primitive shape was fixed in Task 32, but direction/orientation is still not fixed.

Observed output still records:

```txt
hinge_source = fallback_pt0
swing_source = fallback
swing_side = fallback_left
```

This means Phase 4 still does not use the reliable 7-class raster evidence for door direction.

### Required Fix

Door hinge and swing direction must be inferred from the segmentation masks:

```txt
red    = door_arc
orange = door_leaf
purple = door_origin
```

The red `door_arc` component is the primary swing-side evidence.

For each accepted door:

```txt
1. start from the adjusted door_origin segment endpoints p0/p1
2. test hinge=p0/end=p1 and hinge=p1/end=p0
3. for each hinge assignment, test both swing sides
4. score each candidate against the local red/orange/purple evidence
5. choose the highest-scoring candidate
```

### Red Pixel Swing Rule

The door should open toward the side where the local red door_arc pixels are concentrated.

Use component-local evidence:

```txt
1. use the red door_arc component associated with that door
2. or use only red pixels inside a slightly expanded bbox around that component
3. do not count global red pixels from nearby doors
```

Recommended scoring:

```txt
primary:
  count red door_arc pixels on each side of the door_origin segment
  prefer the side with more red pixels

secondary:
  overlap generated arc/quarter-swing region with the red component
  overlap generated leaf line/band with orange door_leaf pixels
  overlap origin line/band with purple door_origin pixels

penalty:
  generated leaf/arc on side with little/no red evidence
  generated leaf/arc crossing wall pixels
  using fallback when red/orange/purple evidence is present
```

### Primitive Flipping

The implementation must allow door primitives to flip.

Do not force all doors to open to one side.

Acceptable approaches:

```txt
1. one door primitive class that accepts hinge endpoint and swing side
2. two generated primitive variants, left/right, selected by evidence
3. four generated hypotheses: p0-left, p0-right, p1-left, p1-right
```

The selected final door must record:

```txt
hinge_source = "red_orange_purple_evidence"
swing_source = "red_door_arc_side"
```

Fallback is allowed only when component-local red/orange/purple evidence is missing or ambiguous.

## Part C - Wall Corner Connection Before Buffering

### Current Failure

Some corners are still disconnected, and buffered segments show capped/disconnected corners.

This suggests the wall buffering stage is still receiving floating line segments whose endpoints are visually close but not topologically identical.

### Required Fix

Before buffering, build a topology-safe wall graph:

```txt
1. collect remaining trimmed wall edges
2. snap near-identical endpoints within tolerance
3. cluster shared x/y axes after trimming
4. split intersections if introduced
5. rebuild edges from snapped node IDs
6. remove zero-length and duplicate edges
7. only then convert graph edges to LineString chains
8. buffer connected chains
```

Do not rely on raw float equality for `shapely.linemerge()`.

The implementation must explicitly report:

```txt
pre_buffer_node_count
post_snap_node_count
pre_buffer_edge_count
chain_count
disconnected_endpoint_count
```

If a visually connected corner remains disconnected, debug output must identify which endpoints failed to snap and why.

## Part D - Notebook Must Be Updated

The notebook is the manual test surface.

Update:

```txt
notebooks/phase4_vectorization.ipynb
```

Requirements:

```txt
1. notebook imports the current source modules/functions
2. notebook does not contain stale inline copies of the old pipeline logic
3. notebook uses adjusted final openings for SVG/JSON output
4. notebook uses evidence-driven door direction
5. notebook uses topology-safe wall buffering
6. notebook has autoreload enabled or clear restart-kernel instructions
7. Run All produces final_vector.svg/json from the same current source code path
```

This task is incomplete if the source code changes but the notebook still runs an old path.

## Required JSON Fields

Add/verify final JSON records:

```json
{
  "openings": {
    "doors": [
      {
        "original_interval": [0.0, 0.0],
        "adjusted_interval": [0.0, 0.0],
        "snapped_points_original": [[0, 0], [0, 0]],
        "snapped_points_adjusted": [[0, 0], [0, 0]],
        "final_points": [[0, 0], [0, 0]],
        "was_adjusted": false,
        "hinge_point": [0, 0],
        "origin_far_point": [0, 0],
        "swing_side": "left|right|fallback_left|fallback_right",
        "hinge_source": "red_orange_purple_evidence|fallback",
        "swing_source": "red_door_arc_side|fallback"
      }
    ],
    "windows": [
      {
        "original_interval": [0.0, 0.0],
        "adjusted_interval": [0.0, 0.0],
        "snapped_points_original": [[0, 0], [0, 0]],
        "snapped_points_adjusted": [[0, 0], [0, 0]],
        "final_points": [[0, 0], [0, 0]],
        "was_adjusted": false
      }
    ]
  },
  "metrics": {
    "pre_buffer_node_count": 0,
    "post_snap_node_count": 0,
    "pre_buffer_edge_count": 0,
    "wall_chain_count": 0,
    "disconnected_endpoint_count": 0
  }
}
```

Equivalent nested structure is acceptable, but the information must be present.

## Required Debug Overlay

`image_debug_overlay.png` must show:

```txt
original opening interval
adjusted opening interval
final primitive endpoints
door hinge point
selected door swing side
red pixels used for swing scoring
unconnected pre-buffer endpoints, if any
```

This makes it possible to audit whether the final vector output obeys the strict process.

## Tests

Add focused tests for:

```txt
adjusted window interval propagates to final SVG endpoints
adjusted door interval propagates to door_origin endpoints
door hinge/swing candidate scoring picks the side with more local red pixels
door primitive can flip left/right and p0/p1 hinge assignment
fallback hinge/swing is used only when evidence is absent/ambiguous
topology-safe wall graph snaps near-identical corner endpoints before buffering
notebook imports current source path and does not duplicate stale core logic
```

Synthetic tests are preferred for geometry. GPU inference is not required for these unit tests.

## Manual Verification Samples

At minimum, rerun the notebook on:

```txt
outputs/vectorization/phase4_vectorization/1316/image.png
outputs/vectorization/phase4_vectorization/10026/image.png
```

Check:

```txt
1. any moved window is drawn at its adjusted location
2. wall trim and window/door primitive endpoints match
3. door directions are not all fallback_left
4. door directions visually follow red door_arc pixels
5. disconnected wall corners are reduced or explicitly reported
```

## Completion Criteria

This task is complete when:

```txt
1. adjusted opening intervals are the source of truth for wall trim, SVG, JSON, and debug overlay
2. door direction is inferred from local red/orange/purple raster evidence
3. door primitives can flip hinge and swing direction
4. wall chains are topology-snap-cleaned before buffering
5. final_vector.svg/json visibly change where Task 33 previously only changed wall trim
6. phase4_vectorization.ipynb is updated and uses the current source path
7. tests cover adjusted-opening propagation, door evidence scoring, and pre-buffer wall connection
```

After completion, rename this file from:

```txt
tasks/task34.md
```

to:

```txt
tasks/task34_done.md
```
