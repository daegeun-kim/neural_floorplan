# Spec v005: SegFormer Training Run3 With Door/Window Subclasses

## 0. Purpose

This spec replaces the outdated v005 SegFormer training spec.

The previous CNN output classes were:

```txt
0 - background
1 - wall
2 - opening
3 - room
4 - icon
```

For `segformer_b0_run3`, keep the current SegFormer model structure and training approach, but change the output semantic classes.

The main goal is to separate windows and doors before vectorization. Door geometry should also be subdivided so the vectorizer can infer the correct number of nearby doors from semantic evidence.

Do not continue development from `segformer_b0_run1` or `segformer_b0_run2`. Those runs are historical comparison runs only.

`specs/spec_v003_semantic_mask_generation.md` has been refreshed (task07) to match this 7-class scheme — it is no longer a 5-class document.

## 1. Run Name

Use this run identity:

```txt
segformer_b0_run3
```

Required output paths:

```txt
checkpoints/segformer_b0_run3
runs/segformer_b0_run3
features/segformer_b0_run3
```

`run3` must have its own feature cache. Do not reuse old cached feature `.pt` files from:

```txt
features/segformer_b0
```

The old cache files are safe to remove because the previous runs are no longer being developed and the output class count is changing.

Keep old checkpoint folders for `run1` and `run2` so they can still be loaded later for comparison.

## 2. Model Scope

Keep the existing CNN architecture and training workflow.

Allowed:

```txt
- change output class count
- update semantic mask generation
- update class colors
- update configs for run3
- rebuild feature cache
- train a new decoder/output head for run3
```

Not allowed:

```txt
- redesign the SegFormer backbone
- switch to instance segmentation
- continue training from run1 or run2 checkpoints
- use latest.pt instead of best.pt for comparison workflows
```

## 3. New Output Classes

The new output class order must be exactly:

| ID | Class | Meaning |
|---:|---|---|
| 0 | background | ignored / non-floorplan pixels |
| 1 | floor | room or floor surface evidence |
| 2 | wall | wall polygons and structural wall evidence |
| 3 | window | all window geometry collapsed into one class |
| 4 | door_arc | door swing quarter-circle region / arc evidence |
| 5 | door_leaf | opened door panel / leaf segment evidence |
| 6 | door_origin | wall-aligned threshold/origin segment evidence |

Remove the previous `icon` / furniture class from run3.

Do not use:

```txt
opening
room
icon
furniture
fixture
```

as final run3 CNN output classes.

## 4. Class Color Palette

Use a fixed seven-class color palette for saved prediction previews and semantic masks.

Recommended palette:

| ID | Class | RGB |
|---:|---|---|
| 0 | background | `(200, 200, 200)` |
| 1 | floor | `(245, 240, 232)` |
| 2 | wall | `(30, 30, 30)` |
| 3 | window | `(60, 120, 220)` |
| 4 | door_arc | `(220, 90, 90)` |
| 5 | door_leaf | `(235, 140, 80)` |
| 6 | door_origin | `(160, 70, 180)` |

The exact palette can be adjusted only if all related code and tests are updated consistently.

## 5. Original SVG Structure Observed

A sample source file was inspected:

```txt
docs/original_vector/cubicasa5k/cubicasa5k/high_quality_architectural/2/model.svg
```

Relevant observed structure:

```xml
<g id="Wall" class="Wall External">
  <polygon .../>
  <g id="Window" class="Window Regular">
    <polygon .../>
    <g id="Glass">...</g>
    <g id="Panel">...</g>
  </g>
  <g id="Door" class="Door Swing Beside">
    <polygon .../>
    <g id="Threshold">...</g>
    <g id="Panel" class="Panel Left Positive">
      <g id="PanelArea">...</g>
      <path d="M... q... l...Z"/>
    </g>
  </g>
</g>
```

Doors and windows can be nested inside wall groups.

For doors:

```txt
Door/Threshold polygon -> door_origin
Door/Panel path curve  -> door_arc and door_leaf evidence
Door/PanelArea polygon -> do not use directly unless needed to support arc/leaf labeling
```

For windows:

```txt
Window group geometry -> window
Glass and Panel subgeometry -> window
```

## 6. Door Labeling Rules

Train three separate semantic classes for door evidence:

```txt
door_arc
door_leaf
door_origin
```

These are semantic classes, not instance IDs.

The purpose is to let vectorization count separate doors by finding separate arc/leaf/origin components after segmentation.

### door_origin

`door_origin` should label the wall-aligned threshold/origin segment of a door.

Use SVG evidence from:

```txt
Door > Threshold
```

and equivalent door threshold geometry.

The origin is the segment that replaces the trimmed wall opening in vectorization.

`door_origin` is rendered as a **stroked centerline**, not a filled polygon: the Threshold
polygon's bounding box long axis (the door-width direction, running along the wall) becomes
a single line segment through its midpoint, stroked at the same fixed width as `door_leaf`
(see §6's `door_leaf` rendering and the shared stroke-width constant in
`src/generate_semantic_masks.py`). This keeps the rendered width identical across every
door regardless of wall thickness, matching `door_leaf`'s width exactly.

### door_leaf

`door_leaf` should label the opened door panel line/segment.

Use SVG evidence from the straight line portion of:

```txt
Door > Panel > path
```

and equivalent door panel geometry.

For SVG paths such as:

```txt
M ... q ... l ... Z
```

the straight `l` segment corresponds to door leaf evidence.

`door_leaf` is rendered as a **stroked line** (not filled) along that `l` segment, using a
configurable minimum raster stroke width so it remains a learnable, separable class even after
the line is overlaid on the `door_arc` wedge fill. `door_leaf` must be rasterized **after**
`door_arc` so that wherever the leaf stroke and the arc wedge overlap, the pixels are labeled
`door_leaf`, not `door_arc` (see §11 priority order).

### door_arc

`door_arc` should label the quarter-circle swing arc region/evidence.

`door_arc` is rendered as a **filled wedge**: the full closed Panel path (`M ... q ... l ... Z`)
is filled and labeled `door_arc`. This gives the CNN a large, easily-learnable swing-region
area rather than a thin curve.

Use SVG evidence from the closed:

```txt
Door > Panel > path
```

and equivalent swing arc geometry.

The straight `l` segment of that same path is part of the wedge's boundary, but it is labeled
`door_leaf` instead (see below) — `door_leaf` is rendered after `door_arc` so the leaf stroke
is never swallowed by the wedge fill.

## 7. Window Labeling Rules

Keep windows simple.

All window subtypes and internal window parts should collapse into one class:

```txt
window
```

Use SVG evidence from:

```txt
Window
Window Regular
Glass
Panel
```

and equivalent window geometry.

Do not create separate classes for glass, sill, panel, or window subtype in run3.

## 8. Wall Labeling Rules

Keep walls simple.

All wall types should collapse into one class:

```txt
wall
```

Use SVG evidence from:

```txt
Wall
Wall External
Column
Railing
```

only if those elements are intended to participate in structural wall/edge prediction. If uncertain, prioritize `Wall` and `Wall External`.

Do not split exterior and interior walls in run3.

## 9. Floor Labeling Rules

Use class:

```txt
floor
```

instead of the previous `room` class name.

Because semantic segmentation cannot assign overlapping classes to one pixel, floor should be labeled as the visible room/floor surface area behind walls, windows, doors, and other foreground classes.

Rules:

```txt
1. Floor is the far-back semantic surface.
2. Wall, window, door_arc, door_leaf, and door_origin override floor where they overlap.
3. Furniture/icon/fixture elements are not separate run3 classes.
4. Do not let removed furniture/icon classes block floor unless they are intentionally rendered as background/noise.
```

Use SVG `Space` polygons as floor evidence where appropriate:

```txt
Space
Space Kitchen
Space Bath
Space Entry Lobby
Space Outdoor
```

Outdoor spaces may be included only if the existing dataset workflow already treats them as valid floor/space evidence. Do not introduce inconsistent outdoor handling without tests.

## 10. Furniture / Icon Removal

Remove the previous icon class from run3.

Do not train a separate furniture or fixture class in run3.

Furniture and fixture SVG elements should not become target classes:

```txt
FixedFurniture
FixedFurnitureSet
ElectricalAppliance
Sink
Toilet
Closet
Cabinet
Shower
Stove
Refrigerator
```

Recommended handling:

```txt
- ignore furniture/fixture geometry as a target class
- let underlying floor remain floor where possible
- if furniture is visible in input raster, treat it as input noise/context, not a supervised output class
```

## 11. Mask Generation Priority

When rasterizing class masks, overlapping semantic layers must use this priority from back to front:

```txt
background
floor
wall
window
door_origin
door_arc
door_leaf
```

Door classes should remain visible where they overlap walls/openings.

`door_leaf` is the highest-priority door subclass: it is rasterized last so that the leaf
stroke remains visible on top of the `door_arc` wedge fill wherever they overlap (see §6).

## 12. Training Configuration

Create or update the run3 training config with:

```yaml
run:
  version: "run3"
  run_name: "segformer_b0_run3"

image:
  image_size: 512
  num_classes: 7

checkpoint:
  output_dir: "checkpoints/segformer_b0_run3"

logging:
  log_dir: "runs/segformer_b0_run3"

feature_cache:
  cache_dir: "features/segformer_b0_run3"
  force_rebuild: true
```

Do not resume run3 from run1 or run2 checkpoints because the output head class count changed.

## 13. Cache Handling

Old cache files in:

```txt
features/segformer_b0
```

are safe to remove.

Run3 must use:

```txt
features/segformer_b0_run3
```

Keep run3 cache files for future modification and debugging.

The cache must be rebuilt for run3.

## 14. Checkpoint Rules

Run3 checkpoints must be saved under:

```txt
checkpoints/segformer_b0_run3
```

Required checkpoint files:

```txt
best.pt
latest.pt
```

`best.pt` is the checkpoint used for later evaluation, prediction preview, and vectorization comparison.

Do not overwrite run1 or run2 checkpoints.

## 15. Preview Outputs

Prediction previews should show the new seven-class palette.

Preview output path:

```txt
runs/segformer_b0_run3/previews
```

Each preview sample should include at least:

```txt
sample_000_input.png
sample_000_target.png
sample_000_prediction.png
sample_000_overlay.png
```

The target and prediction preview images must use the seven-class run3 palette.

## 16. Metrics

Update class-aware metrics to report per-class IoU for:

```txt
background
floor
wall
window
door_arc
door_leaf
door_origin
```

The vector-ready score should prioritize classes useful for vectorization:

```txt
wall
window
door_arc
door_leaf
door_origin
floor
```

Door subclasses should receive meaningful weight because they are needed to count and reconstruct individual doors.

## 17. Acceptance Criteria

Run3 is complete when:

1. The project has an active `specs/spec_v005_segformer_train.md` for run3.
2. The old outdated v005 spec remains separate and is not used for run3 implementation.
3. Semantic mask generation supports exactly seven classes.
4. The output class order is exactly:

```txt
background, floor, wall, window, door_arc, door_leaf, door_origin
```

5. The previous icon/furniture class is removed.
6. Windows are one semantic class.
7. Walls are one semantic class.
8. Doors are split into `door_arc`, `door_leaf`, and `door_origin`.
9. Door labels are derived from SVG `Door`, `Threshold`, and `Panel` structure where available.
10. Run3 uses `checkpoints/segformer_b0_run3`.
11. Run3 uses `runs/segformer_b0_run3`.
12. Run3 uses `features/segformer_b0_run3`.
13. Old `features/segformer_b0` cache files are not reused.
14. Prediction preview images use the new seven-class palette.
15. Per-class metrics include all seven run3 classes.
16. The CNN architecture remains otherwise unchanged.
