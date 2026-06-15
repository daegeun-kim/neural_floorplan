# Goal

Generate lightly augmented sketch-style raster inputs from clean floor plans while preserving semantic geometry alignment.

This stage improves model robustness while maintaining clear architectural structure for early CNN training.

---

# Input Sources

Use only:

- model_clean.png
- model_clean01.png

Ignore:
- F1_original.png
- F1_scaled.png
- F2_original.png
- F2_scaled.png
- unrelated PNG files

If both:
- model_clean.png
- model_clean0.png

exist in the same folder, treat them as separate training samples sharing the same semantic mask targets.

---

# Inputs

X:
- clean raster floorplan image

y:
- semantic masks
- semantic_class_map.png

Augmentation must be applied identically to:
- image
- masks

to preserve pixel alignment.

---

# Augmentation Philosophy

Version 1 should use only light augmentation.

Goal:
- improve generalization
- simulate mild raster variation
- preserve architectural readability

Do NOT use severe deformation in the first CNN training cycle.

Aggressive augmentation should only be introduced after the CNN produces plausible segmentation outputs.

---

# Version 1 Augmentations (Light)

## Geometry-safe transforms

- horizontal flip
- vertical flip
- 90° rotation
- small random translation
- mild padding variation

## Light raster variation

- slight line jitter
- slight stroke width variation
- mild Gaussian blur
- mild grayscale variation
- light JPEG compression
- light brightness/contrast variation

## Optional

- very subtle paper texture
- slight scan noise

---

# Explicitly Avoid in Version 1

Do not use:
- perspective distortion
- elastic deformation
- large rotation angles
- heavy blur
- severe edge breakage
- random erasing
- topology-breaking transforms

Reason:
architectural topology must remain readable during early training.

---

# Future Version 2 Augmentation

After segmentation results become plausible, optionally add:

- broken edges
- stronger sketch noise
- stronger line wobble
- hand-drawn simulation
- scan artifacts
- paper folds
- stronger compression artifacts
- partial occlusion

Goal:
improve robustness to real-world noisy plans.

---

# Alignment Rule

Every spatial transform applied to the input image must also be applied identically to:
- wall_mask
- opening_mask
- room_mask
- icon_mask
- semantic_class_map

Example:

rotate image 90°
→ rotate all masks 90°

Failure to maintain alignment invalidates the dataset.

---

# Output

Generate augmented dataset pairs:

- augmented_image.png
- augmented_semantic_class_map.png

Optional:
- augmented_wall_mask.png
- augmented_room_mask.png
- augmentation_metadata.json

---

# Acceptance Criteria

- Input image and label masks remain perfectly aligned.
- Augmentation can be enabled/disabled from config.
- Augmentation intensity can be adjusted.
- Preview script exports visual examples.
- Original files are never overwritten.
- Generated outputs preserve architectural readability.

---

# Preview Requirement

Provide a preview/debug script that exports:

- original image
- augmented image
- semantic overlay

for manual inspection before full dataset generation.

---

# Recommended First Milestone

1. Generate 20 augmented samples
2. Manually inspect all outputs
3. Verify:
   - wall continuity
   - room readability
   - mask alignment
4. Then scale to full dataset