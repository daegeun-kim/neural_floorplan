# Spec: Semantic Mask Generation for Neural Floorplan

## 0. Active Version Notice

This spec has been refreshed (task07) to match the 7-class scheme implemented by `src/generate_semantic_masks.py` and trained as `segformer_b0_run3` (see `spec_v005_segformer_train.md`). Earlier revisions of this document described a 5-class `background/wall/opening/room/icon` mapping — that mapping is retired and must not be treated as current. It survives only inside `spec_v005_segformer_train_outdated.md` as historical context.

## 1. Purpose

Create a preprocessing script that converts CubiCasa5K SVG floorplan annotations into CNN-training-ready semantic masks.

`model.svg` is the semantic source of truth — it is parsed and rasterized into dense pixel labels, never used as the CNN label directly.

The goal is to transform:

```text
model.svg
```

into aligned raster label tensors:

```text
floor_mask.png
wall_mask.png
window_mask.png
door_arc_mask.png
door_leaf_mask.png
door_origin_mask.png
semantic_class_map.png
```

These masks will be used as the ground-truth labels for semantic segmentation training.

This stage does **not** train the CNN and does **not** perform vector reconstruction. It only prepares reliable pixel-level labels from the SVG annotation source.

---

## 2. Project Context

The full Neural Floorplan pipeline is:

```text
1. Dataset loading
2. SVG/raster preprocessing
3. Semantic mask generation
4. Sketch-style augmentation
5. Segmentation CNN training
6. Evaluation
7. Mask-to-vector post-processing
8. Classified JSON / SVG export
```

This spec covers only:

```text
3. Semantic mask generation
```

The CNN will later learn:

```text
floorplan raster image → semantic masks
```

not:

```text
floorplan raster image → SVG text directly
```

The SVG remains the source of truth, but the training label format is rasterized semantic masks.

---

## 3. Input Data

Each CubiCasa sample folder has a structure similar to:

```text
sample_folder/
  model.svg
  F1_original.png
  F1_scaled.png
  F2_original.png
  F2_scaled.png
  model_clean.png
```

For this stage, the required input is:

```text
model.svg
```

Optional input for validation:

```text
model_clean.png
F1_scaled.png
```

`model_clean.png` is the clean raster image rendered directly from `model.svg` in the previous SVG-to-raster stage. `F1_scaled.png` is the original CubiCasa raster image (the real-world scan/listing image), kept aligned to the same SVG-derived target.

---

## 4. Output Data

For each sample folder, create a new subfolder:

```text
sample_folder/
  masks/
    floor_mask.png
    wall_mask.png
    window_mask.png
    door_arc_mask.png
    door_leaf_mask.png
    door_origin_mask.png
    semantic_class_map.png
    mask_metadata.json
```


### Two Raster Inputs Per One SVG

Each sample folder contains both:

```text
model_clean.png
F1_scaled.png
```

Their meaning is:

| File | Meaning |
|---|---|
| `model_clean.png` | Synthetic clean raster directly rendered from `model.svg` |
| `F1_scaled.png` | Original CubiCasa raster image, scaled to align with the SVG coordinate space |

Both files are treated as **two separate training rows** sharing the **same SVG-derived target masks**. This is implemented by `CLEAN_IMAGE_NAMES`/`INPUT_TYPE_MAP` in `src/build_splits.py`.

Conceptually:

```text
Sample A:
X = F1_scaled.png
y = masks/semantic_class_map.png

Sample B:
X = model_clean.png
y = masks/semantic_class_map.png
```

This is not a problem because both raster inputs represent the same annotated floorplan geometry.

The SVG should not be duplicated physically unless needed for a downstream dataset format. Instead, the dataset index duplicates the sample reference.

Example dataset index:

```json
[
  {
    "sample_id": "0001_original_raster",
    "image": "0001/F1_scaled.png",
    "target": "0001/masks/semantic_class_map.png",
    "source_svg": "0001/model.svg",
    "input_type": "original_raster"
  },
  {
    "sample_id": "0001_svg_rendered_clean",
    "image": "0001/model_clean.png",
    "target": "0001/masks/semantic_class_map.png",
    "source_svg": "0001/model.svg",
    "input_type": "svg_rendered_clean"
  }
]
```

Important rule:

```text
One SVG creates one target mask set.
Both raster inputs (original and SVG-rendered) point to the same target mask set.
```

This increases training diversity without manually creating new labels. Evaluation should report metrics grouped by `input_type` (`svg_rendered_clean` vs `original_raster`) — see `spec_v005_segformer_train.md` §"Training Configuration" requirements.


Required outputs:

```text
masks/
  floor_mask.png
  wall_mask.png
  window_mask.png
  door_arc_mask.png
  door_leaf_mask.png
  door_origin_mask.png
  semantic_class_map.png
  mask_metadata.json
```

Historical/optional outputs from earlier 5-class revisions of this spec (no longer generated):

```text
opening_mask.png
room_mask.png
icon_mask.png
furniture_mask.png
space_type_mask.png
instance_map.png
```

---

## 5. Semantic Classes

### Active Classes (run3, 7 classes)

This is the only class scheme produced by the current `src/generate_semantic_masks.py` and consumed by `segformer_b0_run3` (see `spec_v005_segformer_train.md`):

| Class ID | Class Name | Description |
|---:|---|---|
| 0 | background | Empty space / non-floorplan pixels |
| 1 | floor | Room/floor surface evidence (`Space` polygons) |
| 2 | wall | Structural wall geometry |
| 3 | window | All window subtypes collapsed into one class |
| 4 | door_arc | Door swing quarter-circle wedge evidence |
| 5 | door_leaf | Opened door panel/leaf line evidence |
| 6 | door_origin | Wall-aligned door threshold/origin segment evidence |

There is no separate furniture/icon/fixture class in the active scheme — furniture/fixture SVG elements are not rasterized as a target class (see `spec_v005_segformer_train.md` §10).

### Historical Classes (retired, do not implement)

Earlier revisions of this spec described a 5-class `background/wall/opening/room/icon` scheme, later sketched as an 8-class `door/window/room/furniture/fixture/stair/text_or_annotation` expansion. Neither was implemented as described; both are retired in favor of the 7-class table above. They remain documented only in `spec_v005_segformer_train_outdated.md` for historical reference.

---

## 6. Core Concept

The SVG is not used directly as the CNN label. Instead, the SVG is parsed and rasterized into dense pixel labels.

Example:

```text
SVG Space elements          → floor_mask.png
SVG Wall elements            → wall_mask.png
SVG Window elements          → window_mask.png
SVG Door > Threshold         → door_origin_mask.png (stroked centerline)
SVG Door > Panel closed path → door_arc_mask.png (filled wedge)
SVG Door > Panel line segment→ door_leaf_mask.png (stroked line)
```

Then the masks are combined into:

```text
semantic_class_map.png
```

where every pixel stores a class ID.

Example:

```text
0 = background
1 = floor
2 = wall
3 = window
4 = door_arc
5 = door_leaf
6 = door_origin
```

---

## 7. Coordinate and Resolution Rules

The generated masks must match the coordinate system and raster size of `model_clean.png`.

### Required Rule

If `model_clean.png` exists:

```text
mask width  == model_clean.png width
mask height == model_clean.png height
```

If `model_clean.png` does not exist, render masks using the native SVG dimensions.

`F1_scaled.png` is expected to already be scaled to this same coordinate space (hence the name) so it can share the same target mask as a second training row — see §4 "Two Raster Inputs Per One SVG".

### Do Not

Do not resize each mask independently.

### Correct

All outputs from the same SVG must share:

```text
same width
same height
same origin
same scale
same coordinate space
```

This is critical because the CNN requires pixel-level alignment between input image and label mask.

---

## 8. SVG Parsing Strategy

Use Python XML parsing to inspect the SVG structure.

Recommended libraries:

```text
xml.etree.ElementTree
lxml
BeautifulSoup
```

The script should identify semantic categories from SVG group IDs, class names, or element attributes.

The actual observed CubiCasa5K structure (see `spec_v005_segformer_train.md` §5 and the module docstring in `src/generate_semantic_masks.py`) nests doors/windows inside wall groups:

```xml
<g id="Wall" class="Wall External">
  <polygon .../>
  <g id="Window" class="Window Regular">...</g>
  <g id="Door" class="Door Swing Beside">
    <polygon .../>
    <g id="Threshold">...</g>
    <g id="Panel" class="Panel Left Positive">
      <path d="M... q... l...Z"/>
    </g>
  </g>
</g>
<g id="<uuid>" class="Space ...">...</g>
```

The implementation must not assume only one exact naming pattern. It should use a configurable mapping.

---

## 9. Category Mapping

The active mapping, matching `CLASS_IDS`/`APPLY_ORDER` in `src/generate_semantic_masks.py`:

```python
CATEGORY_MAP = {
    "floor": ["space"],                 # Space polygons, including kitchen/bath/entry/outdoor variants
    "wall": ["wall"],                   # Wall / Wall External (Column, Railing only if intended as structural)
    "window": ["window"],               # Window, Window Regular, Glass, Panel — all collapse to one class
    "door_origin": ["door > threshold"],  # door's Threshold polygon -> stroked centerline
    "door_arc": ["door > panel (closed path)"],  # door's Panel closed path -> filled swing wedge
    "door_leaf": ["door > panel (line segment)"],  # door's Panel path straight `l` segment -> stroked line
}
```

Matching should be case-insensitive. Furniture/fixture elements (`FixedFurniture`, `Sink`, `Toilet`, `Closet`, etc.) are intentionally **not** mapped to any class — they are ignored as a target class, not collapsed into an `icon` class (see `spec_v005_segformer_train.md` §10).

Example:

```text
"Wall" → wall
"WALL" → wall
"Space Kitchen" → floor
"Door > Threshold" → door_origin
"Door > Panel" path → door_arc and door_leaf (see §10 below)
"FixedFurniture" → (ignored, no target class)
```

---

## 10. Mask Generation Method

### Option A — Recommended First Method (active)

Create temporary SVG files for each semantic category and render them with CairoSVG.

For each class:

```text
1. Parse original SVG
2. Keep only elements belonging to selected class
3. Remove all unrelated SVG elements
4. Render filtered SVG to PNG
5. Convert rendered PNG to binary mask
```

Example:

```text
filtered_floor.svg  → floor_mask.png
filtered_wall.svg    → wall_mask.png
filtered_window.svg  → window_mask.png
```

`door_origin`, `door_arc`, and `door_leaf` masks are **not** produced by filtering+rendering the original SVG elements directly. They are synthesized as fixed-width stroked lines (`door_origin`, `door_leaf`) or a filled wedge (`door_arc`) derived from the `Door > Threshold` / `Door > Panel` geometry — see `spec_v005_segformer_train.md` §6 for the exact rendering rules and the shared stroke-width constant.

Advantages:
- Keeps SVG coordinate system intact
- Handles paths, polygons, transforms, strokes better than manual geometry parsing
- Consistent with the previous `svg_to_raster.py` workflow

### Option B — Later Advanced Method

Manually parse paths/polygons and rasterize with OpenCV/Shapely.

Use this only if CairoSVG filtering is insufficient.

---

## 11. Binary Mask Rules

Each class-specific mask should be binary:

```text
0 = not this class
255 = this class
```

Example:

```text
wall_mask.png
  white pixels = wall
  black pixels = not wall
```

Use grayscale PNG mode:

```text
L
```

not RGB.

---

## 12. Semantic Class Map Rules

The combined semantic map should be a single-channel image:

```text
semantic_class_map.png
```

Pixel values:

```text
0 = background
1 = floor
2 = wall
3 = window
4 = door_arc
5 = door_leaf
6 = door_origin
```

This is suitable for PyTorch segmentation training with cross-entropy loss.

Expected tensor shape after loading:

```text
[H, W]
```

not:

```text
[H, W, 3]
```

---

## 13. Class Priority / Overlap Resolution

Some SVG elements may overlap after rasterization.

Use this class priority (back to front — matches `APPLY_ORDER` in `src/generate_semantic_masks.py`):

```text
door_leaf > door_arc > door_origin > window > wall > floor > background
```

Reason:
- Floor regions can occupy large filled areas behind everything else.
- Wall, window, and door evidence are more specific and should override floor where they overlap.
- Among door subclasses, `door_leaf` is rasterized last so the leaf stroke stays visible on top of the `door_arc` wedge fill.

Implementation rule:

```python
semantic_map[floor_mask > 0] = 1
semantic_map[wall_mask > 0] = 2
semantic_map[window_mask > 0] = 3
semantic_map[door_origin_mask > 0] = 6
semantic_map[door_arc_mask > 0] = 4
semantic_map[door_leaf_mask > 0] = 5
```

This order (the real `APPLY_ORDER = ["floor", "wall", "window", "door_origin", "door_arc", "door_leaf"]`) makes `door_leaf` the highest priority.

---

## 14. Expected Script

Create:

```text
src/generate_semantic_masks.py
```

The script should process all sample folders under a given dataset root.

Example command:

```powershell
python -m src.generate_semantic_masks "C:\path\to\high_quality_architectural"
```

Optional flags:

```powershell
python -m src.generate_semantic_masks "C:\path\to\high_quality_architectural" --overwrite --verbose
```

---

## 15. CLI Requirements

The script should accept:

```text
root_dir
```

Required positional argument.

Optional arguments:

```text
--overwrite
--verbose
--output-dir-name masks
--classes-v1
--debug-overlays
```

### Example

```powershell
python -m src.generate_semantic_masks `
"C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\docs\original_vector\cubicasa5k\cubicasa5k\high_quality_architectural" `
--verbose
```

---

## 16. Processing Behavior

For each sample folder:

```text
1. Find model.svg
2. Parse SVG
3. Identify semantic groups/elements
4. Render binary masks per class
5. Combine masks into semantic_class_map.png
6. Save mask_metadata.json
7. Skip existing masks unless --overwrite is used
```

If no `model.svg` exists, skip the folder and log a warning.

If semantic classes are missing, still write metadata and log the issue.

---

## 17. Metadata Output

Create:

```text
mask_metadata.json
```

Example:

```json
{
  "source_svg": "model.svg",
  "width": 1024,
  "height": 768,
  "classes": {
    "0": "background",
    "1": "floor",
    "2": "wall",
    "3": "window",
    "4": "door_arc",
    "5": "door_leaf",
    "6": "door_origin"
  },
  "class_pixel_counts": {
    "background": 650000,
    "floor": 275000,
    "wall": 52000,
    "window": 4000,
    "door_arc": 1800,
    "door_leaf": 600,
    "door_origin": 600
  },
  "missing_classes": [],
  "status": "ok"
}
```

Purpose:
- verify masks were generated correctly
- detect empty or broken masks
- help later train/validation filtering

---

## 18. Debug Overlay Output

If `--debug-overlays` is used, create:

```text
masks/debug_overlay.png
```

This should show semantic masks overlaid on `model_clean.png` or a white background.

Suggested colors (matches `DEBUG_COLORS` in `src/generate_semantic_masks.py`):

```text
floor = (245, 240, 232)
wall = (30, 30, 30)
window = (60, 120, 220)
door_arc = (220, 90, 90)
door_leaf = (235, 140, 80)
door_origin = (160, 70, 180)
```

This is only for human inspection, not training.

---

## 19. Validation Requirements

The script must verify:

```text
floor_mask.png exists
wall_mask.png exists
window_mask.png exists
door_arc_mask.png exists
door_leaf_mask.png exists
door_origin_mask.png exists
semantic_class_map.png exists
mask_metadata.json exists
```

Also verify:

```text
all masks have same width and height
semantic_class_map contains only valid class IDs
at least one non-background class exists
```

If a sample has zero wall pixels, mark it as suspicious.

---

## 20. Dataset-Level Summary

After processing all folders, print:

```text
Processed: N
Skipped existing: N
Missing SVG: N
Failed: N
Suspicious: N
```

Also write:

```text
semantic_mask_generation_summary.json
```

at the dataset root.

Example:

```json
{
  "processed": 3732,
  "skipped_existing": 0,
  "missing_svg": 0,
  "failed": 3,
  "suspicious": 42
}
```

---

## 21. Quality Control Strategy

Do not manually inspect all samples.

Instead:

```text
1. Generate all masks automatically
2. Randomly sample 50 debug overlays
3. Manually inspect those overlays
4. Inspect suspicious samples from metadata
5. Fix category mapping if needed
6. Regenerate masks
```

Expected review folders:

```text
quality_check/
  random_overlays/
  suspicious_overlays/
```

---

## 22. Important Non-Goals

This script should not:

```text
train CNN
augment images
convert predicted masks to vectors
clean original raster images
manually fix multi-floor plans
directly generate final SVG output
```

Those belong to later stages.

---

## 23. Relationship to Sketch Augmentation

This script creates the clean labels.

The next stage, sketch augmentation, should take one clean input raster plus the shared semantic labels.

Possible input rasters:

```text
model_clean.png
F1_scaled.png
```

Shared labels:

```text
semantic_class_map.png
class-specific masks
```

and produce augmented pairs:

```text
augmented_image.png
augmented_semantic_class_map.png
```

Augmentation must transform image and masks together.

Example:

```text
rotate image 90 degrees
rotate semantic_class_map 90 degrees
```

Never augment the input without applying the identical transformation to the labels.

---

## 24. Relationship to CNN Training

The CNN training stage will use:

```text
X = model_clean.png, F1_scaled.png, or augmented raster image
y = semantic_class_map.png
```

Both `model_clean.png` and `F1_scaled.png` appear as two rows in the training dataset index, both pointing to the same target mask.


For PyTorch:

```text
X shape: [C, H, W]
y shape: [H, W]
```

The model output:

```text
logits shape: [num_classes, H, W]
```

Loss:

```text
CrossEntropyLoss
```

---

## 25. File Structure After This Stage

Expected result:

```text
high_quality_architectural/
  0001/
    model.svg
    model_clean.png
    F1_scaled.png
    masks/
      floor_mask.png
      wall_mask.png
      window_mask.png
      door_arc_mask.png
      door_leaf_mask.png
      door_origin_mask.png
      semantic_class_map.png
      debug_overlay.png
      mask_metadata.json

  0002/
    model.svg
    model_clean.png
    F1_scaled.png
    masks/
      floor_mask.png
      wall_mask.png
      window_mask.png
      door_arc_mask.png
      door_leaf_mask.png
      door_origin_mask.png
      semantic_class_map.png
      mask_metadata.json

  semantic_mask_generation_summary.json
```

---

## 26. Implementation Notes

### Recommended packages

```text
cairosvg
Pillow
numpy
lxml
pytest
```

Optional:

```text
opencv-python
```

### Environment

Use the existing conda environment:

```text
floorplan-cad
```

Run from project root:

```powershell
conda activate floorplan-cad
python -m src.generate_semantic_masks <root_dir>
```

If Cairo DLL path is required on Windows:

```powershell
$env:PATH = "C:\Users\kdgki\anaconda3\envs\floorplan-cad\Library\bin;" + $env:PATH
```

---

## 27. Testing Requirements

Create:

```text
tests/test_generate_semantic_masks.py
```

Minimum tests:

```text
1. Creates masks from a simple SVG
2. Creates semantic_class_map.png
3. Class map contains expected class IDs
4. Masks are binary
5. Skip behavior works when outputs already exist
6. Overwrite behavior works
7. Missing SVG folder is skipped
8. Metadata JSON is created
9. Debug overlay is created when flag is used
10. All masks share the same dimensions
```

---

## 28. Success Criteria

This stage is complete when:

```text
1. All SVGs can be processed automatically
2. Each sample has class-specific masks
3. Each sample has one semantic_class_map.png
4. Random debug overlays visually match the plan geometry
5. Suspicious samples are listed automatically
6. Tests pass
```

The output should be ready for:

```text
spec_sketch_augmentation.md
```

---

## 29. Practical First Milestone

Do not process all 3,732 plans first.

Start with:

```text
10 sample folders
```

Then inspect:

```text
masks/debug_overlay.png
```

After the logic is correct, process:

```text
100 samples
```

Then finally process:

```text
all high_quality_architectural samples
```

---

## 30. Key Principle

The SVG is the ground-truth source.

The CNN label is not the SVG file directly.

The CNN label is the semantic mask generated from the SVG.

Therefore:

```text
model.svg → semantic masks → CNN labels
```

This makes the raster-to-vector project trainable using standard semantic segmentation.
