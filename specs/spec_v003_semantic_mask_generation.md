# Spec: Semantic Mask Generation for Neural Floorplan

## 1. Purpose

Create a preprocessing script that converts CubiCasa5K SVG floorplan annotations into CNN-training-ready semantic masks.

The goal is to transform:

```text
model.svg
```

into aligned raster label tensors:

```text
wall_mask.png
door_mask.png
window_mask.png
room_mask.png
icon_mask.png
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
```

`model_clean.png` is the clean raster image generated from `model.svg` in the previous SVG-to-raster stage.

---

## 4. Output Data

For each sample folder, create a new subfolder:

```text
sample_folder/
  masks/
    wall_mask.png
    door_mask.png
    window_mask.png
    room_mask.png
    icon_mask.png
    opening_mask.png
    semantic_class_map.png
    mask_metadata.json
```


### Multiple Clean Raster Inputs Per One SVG

Some sample folders may contain both:

```text
model_clean.png
model_clean01.png
```

Their meaning is:

| File | Meaning |
|---|---|
| `model_clean.png` | Synthetic clean raster directly rendered from `model.svg` |
| `model_clean01.png` | Manually verified clean real raster, usually derived from `F1_original.png` |

If both files exist, they should be treated as **two separate training inputs** sharing the **same SVG-derived target masks**.

Conceptually:

```text
Sample A:
X = model_clean.png
y = masks/semantic_class_map.png

Sample B:
X = model_clean01.png
y = masks/semantic_class_map.png
```

This is not a problem because both raster inputs represent the same annotated floorplan geometry.

The SVG should not be duplicated physically unless needed for a downstream dataset format. Instead, the dataset index should duplicate the sample reference.

Example dataset index:

```json
[
  {
    "sample_id": "0001_svg_clean",
    "image": "0001/model_clean.png",
    "target": "0001/masks/semantic_class_map.png",
    "source_svg": "0001/model.svg",
    "input_type": "svg_rendered_clean"
  },
  {
    "sample_id": "0001_real_clean",
    "image": "0001/model_clean01.png",
    "target": "0001/masks/semantic_class_map.png",
    "source_svg": "0001/model.svg",
    "input_type": "manual_real_clean"
  }
]
```

Important rule:

```text
One SVG can create one target mask set.
Multiple verified clean raster inputs can point to the same target mask set.
```

This increases training diversity without manually creating new labels.


Minimum required outputs for the first version:

```text
masks/
  wall_mask.png
  opening_mask.png
  room_mask.png
  icon_mask.png
  semantic_class_map.png
  mask_metadata.json
```

Optional later outputs:

```text
door_mask.png
window_mask.png
furniture_mask.png
space_type_mask.png
instance_map.png
```

---

## 5. Semantic Classes

Use a small number of high-confidence classes first.

### Version 1 Classes

| Class ID | Class Name | Description |
|---:|---|---|
| 0 | background | Empty space / white background |
| 1 | wall | Structural wall geometry |
| 2 | opening | Doors, windows, wall openings |
| 3 | room | Interior room / space regions |
| 4 | icon | Furniture, fixtures, symbols, objects |

### Version 2 Classes

After the first version works, split `opening` and `icon` into more detailed classes.

| Class ID | Class Name |
|---:|---|
| 0 | background |
| 1 | wall |
| 2 | door |
| 3 | window |
| 4 | room |
| 5 | furniture |
| 6 | fixture |
| 7 | stair |
| 8 | text_or_annotation |

Do not start with too many classes. First produce reliable wall/opening/room/icon masks.

---

## 6. Core Concept

The SVG is not used directly as the CNN label. Instead, the SVG is parsed and rasterized into dense pixel labels.

Example:

```text
SVG wall elements → wall_mask.png
SVG door/window elements → opening_mask.png
SVG space/room elements → room_mask.png
SVG furniture/icon elements → icon_mask.png
```

Then the masks are combined into:

```text
semantic_class_map.png
```

where every pixel stores a class ID.

Example:

```text
0 = background
1 = wall
2 = opening
3 = room
4 = icon
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

Possible SVG structures may include:

```xml
<g id="Wall">
<g class="Wall">
<g id="Door">
<g id="Window">
<g id="Space">
<g id="FixedFurniture">
```

The implementation must not assume only one exact naming pattern. It should use a configurable mapping.

---

## 9. Category Mapping

Create a mapping dictionary in the script or a config file.

Example:

```python
CATEGORY_MAP = {
    "wall": ["wall", "walls"],
    "opening": ["door", "doors", "window", "windows", "opening"],
    "room": ["space", "room", "rooms", "kitchen", "bedroom", "bathroom", "livingroom"],
    "icon": ["fixedfurniture", "furniture", "bathtub", "toilet", "sink", "stairs", "appliance"]
}
```

Matching should be case-insensitive.

Example:

```text
"Wall" → wall
"WALL" → wall
"FixedFurniture" → icon
"Door" → opening
```

---

## 10. Mask Generation Method

### Option A — Recommended First Method

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
filtered_wall.svg → wall_mask.png
filtered_opening.svg → opening_mask.png
filtered_room.svg → room_mask.png
```

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
1 = wall
2 = opening
3 = room
4 = icon
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

Use this class priority:

```text
wall > opening > icon > room > background
```

Reason:
- Room regions can occupy large filled areas.
- Walls/openings/icons are more specific.
- If room pixels overlap walls, wall should win.
- If furniture overlaps a room, icon should win.

Implementation rule:

```python
semantic_map[room_mask > 0] = 3
semantic_map[icon_mask > 0] = 4
semantic_map[opening_mask > 0] = 2
semantic_map[wall_mask > 0] = 1
```

This order makes wall the highest priority.

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
    "1": "wall",
    "2": "opening",
    "3": "room",
    "4": "icon"
  },
  "class_pixel_counts": {
    "background": 650000,
    "wall": 52000,
    "opening": 4000,
    "room": 275000,
    "icon": 12000
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

Suggested colors:

```text
wall = black
opening = red
room = light blue
icon = green
```

This is only for human inspection, not training.

---

## 19. Validation Requirements

The script must verify:

```text
wall_mask.png exists
opening_mask.png exists
room_mask.png exists
icon_mask.png exists
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
model_clean01.png
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
X = model_clean.png, model_clean01.png, or augmented raster image
y = semantic_class_map.png
```

If both `model_clean.png` and `model_clean01.png` exist, they should appear as two rows in the training dataset index, both pointing to the same target mask.


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
    model_clean01.png
    masks/
      wall_mask.png
      opening_mask.png
      room_mask.png
      icon_mask.png
      semantic_class_map.png
      debug_overlay.png
      mask_metadata.json

  0002/
    model.svg
    model_clean.png
    masks/
      wall_mask.png
      opening_mask.png
      room_mask.png
      icon_mask.png
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
