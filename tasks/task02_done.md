# Task 02 — Update CNN Input Workflow for Original Raster and SVG-Raster Training

## Context

The project trains a CNN-based floorplan image segmentation model using raster floorplan images as inputs and semantic mask PNGs as labels.

The previous workflow used:

- `model_clean.png` as the main input image.
- `model_clean0.png` as an optional manually verified raster image from the original CubiCasa5K dataset.
- If `model_clean0.png` existed, both `model_clean.png` and `model_clean0.png` were used as separate CNN inputs with the same semantic mask label.
- If `model_clean0.png` did not exist, only `model_clean.png` was used.

This workflow was needed because some SVG-to-raster exports did not fully match the original CubiCasa5K raster image. Some SVG elements were hidden with `display: none`, so direct SVG rasterization could miss valid floorplan geometry.

That issue has now been addressed. The SVG-to-raster export process now makes hidden floorplan-relevant SVG elements visible before exporting. Therefore, `model_clean.png` should now correspond correctly to the original CubiCasa5K raster image.

Because of this, the manually verified `model_clean0.png` workflow is no longer needed.

## Objective

Update the CNN training data pipeline so that each floorplan sample can use two raster inputs:

1. Original CubiCasa5K raster image: `F1_scaled.png`
2. Direct SVG-to-raster export: `model_clean.png`

Both images should use the same semantic mask PNG as the label.

This creates two separate training samples for the same semantic label:

- `F1_scaled.png` → semantic mask
- `model_clean.png` → same semantic mask

## Updated Workflow

### Previous Workflow

For each sample directory:

- Use `model_clean.png` as image input.
- If `model_clean0.png` exists:
  - Also use `model_clean0.png` as an additional image input.
  - Duplicate the same semantic mask label for both image inputs.
- If `model_clean0.png` does not exist:
  - Use only `model_clean.png`.

### New Workflow

For each sample directory:

- Use `F1_scaled.png` as one image input.
- Use `model_clean.png` as another image input.
- Use the same semantic mask PNG label for both inputs.
- Do not use `model_clean0.png`.
- Do not require manually verified raster copies.

The resulting dataset should treat these as two independent image-label pairs:

```txt
F1_scaled.png   -> semantic_mask.png
model_clean.png -> semantic_mask.png