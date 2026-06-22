# Neural Floorplan → Classified CAD Workflow

## 0. Purpose of This Document

This document explains the overall goal and project workflow for Neural Floorplan.

It is not a detailed implementation specification. Detailed training rules, model configuration, checkpointing, loss functions, and metric formulas belong in the relevant spec files.

The purpose of this file is to explain:

```text
what the project is trying to achieve
why each stage exists
how each stage connects to the next one
what is currently in scope
what is intentionally left for later
```

---

## 1. Project Goal

The project goal is to convert a raster floorplan into a clean, classified, CAD-like representation.

The intended long-term pipeline is:

```text
raster floorplan image
→ semantic understanding
→ classified masks
→ vector geometry
→ clean CAD-like JSON / SVG
→ later Grasshopper or downstream spatial analysis
```

The current active development stage is the CNN segmentation stage:

```text
raster floorplan image
→ 7-class semantic segmentation map
```

The active model is `segformer_b0_run3` (see `spec_v005_segformer_train.md`). Earlier `run1`/`run2` checkpoints used a 5-class scheme and are kept only for historical comparison.

The vectorization stage comes later, after the CNN output is visually and numerically reliable.

---

## 2. Core Design Principle

The project should not simply maximize pixel-perfect segmentation.

A good result should satisfy both:

```text
1. The predicted pixels are accurate enough to support vectorization.
2. The predicted masks preserve the architectural intention of the plan.
```

This means pixel accuracy still matters, especially for clean or well-drawn plans. A clean input should produce a clean, reliable mask because later CAD conversion depends on it.

At the same time, mean IoU alone is not enough. A model can have lower mIoU on messy raster inputs while still producing a more useful architectural interpretation. Openings are especially important because they control circulation and room connectivity.

The model should therefore be evaluated as a segmentation model and as a future vectorization input.

---

## 3. Current Scope

The current project scope is:

```text
CubiCasa5K raster / SVG data
→ prepared semantic masks
→ SegFormer-based 7-class CNN segmentation (segformer_b0_run3)
→ visual and metric-based evaluation
```

The current CNN model predicts these seven classes:

| Class ID | Class Name |
|---:|---|
| 0 | background |
| 1 | floor |
| 2 | wall |
| 3 | window |
| 4 | door_arc |
| 5 | door_leaf |
| 6 | door_origin |

No additional semantic labels should be introduced at this stage.

Do not add:

```text
hinge labels
corner labels
wall centerline labels
room instance labels
new semantic classes
new external datasets
```

The project size should stay fixed until the current 7-class model is satisfactory.

---

## 4. Dataset Strategy

### 4.1 Primary Dataset

CubiCasa5K is the active dataset.

It is used because it provides both raster floorplan images and SVG-based semantic information. The SVG data allows the project to generate semantic masks that become the training target.

The active data sources are:

```text
F1_scaled.png        original CubiCasa raster image
model_clean.png      SVG-rendered clean raster image
masks/semantic_class_map.png
```

Both `F1_scaled.png` and `model_clean.png` can be used as separate input samples while sharing the same target mask.

This lets the model learn from both:

```text
clean SVG-rendered floorplans
original real raster floorplans
```

The clean image helps the model learn precise semantic structure. The original raster helps the model generalize to messier real inputs.

---

### 4.2 Inactive / Future Datasets

Other datasets such as HouseGAN++, RPLAN, or FloorPlanCAD may be useful later for topology, layout distribution, or CAD-level details.

They are not active in the current workflow.

The current priority is to make the CubiCasa5K-based 7-class CNN pipeline reliable before expanding the dataset scope.

---

## 5. Data Preparation Intent

The purpose of data preparation is to create reliable image-mask pairs.

Each sample should have:

```text
input image:
    F1_scaled.png or model_clean.png

target:
    masks/semantic_class_map.png
```

The target mask is a single class-ID image where each pixel belongs to one of the seven classes.

The separate binary masks are useful for inspection and debugging:

```text
floor_mask.png
wall_mask.png
window_mask.png
door_arc_mask.png
door_leaf_mask.png
door_origin_mask.png
```

The combined training target is:

```text
semantic_class_map.png
```

Background does not need a separate mask file. Background is simply every pixel that is not assigned to floor, wall, window, or one of the door classes.

---

## 6. CNN Segmentation Stage

The CNN stage is responsible for perception.

Its job is to answer:

```text
what semantic class does each pixel belong to?
```

It should learn to recognize:

```text
floor
walls
windows
door_arc
door_leaf
door_origin
background
```

from both clean and original raster floorplans.

The CNN should not directly produce CAD geometry, room graphs, or final SVG output.

The current model direction is:

```text
SegFormer-B0 pretrained backbone
+ custom trainable segmentation head
+ 7-class semantic output
```

The model output is:

```text
[B, 7, H, W]
```

After prediction, the model produces a hard semantic class map for inspection and later use.

---

## 7. Training Intention

The training goal is not only to reduce loss. The model should produce masks that are visually clean, class-consistent, and useful for later vectorization.

The model should perform well on:

```text
clean SVG-rendered floorplans
original CubiCasa raster floorplans
lightly augmented floorplans
```

Clean input accuracy remains important. If a user provides a clear, well-drawn plan, the model should not over-generalize or distort it.

Messy input generalization is also important. If the input is noisy or less clean, the model should still recover the main spatial structure.

---

## 8. Evaluation Intention

The project should keep standard segmentation metrics, but they should not be the only way to judge the model.

Useful metrics include:

```text
train_loss
val_loss
pixel_accuracy
foreground_mIoU
per-class IoU
wall_IoU
window_IoU
door_arc_IoU
door_leaf_IoU
door_origin_IoU
floor_IoU
wall_boundary_F1
door_boundary_F1
```

Pixel accuracy matters because a well-drawn plan should produce a precise mask.

Door and window quality matters because openings affect circulation. A small wall dimension shift may be acceptable, but a misplaced or miscounted door/window can change the architectural interpretation.

For this reason, the best model should be selected with a vector-readiness score that gives more weight to windows and door subclasses than walls (see `spec_v005_segformer_train.md` §16).

The evaluation should also separate results by input type when possible:

```text
clean SVG-rendered input
original raster input
overall validation set
```

This helps answer two different questions:

```text
Can the model accurately segment clean plans?
Can the model generalize to original raster plans?
```

If separating metrics by input type becomes computationally expensive, it can be skipped temporarily, but it should remain part of the intended evaluation design.

---

## 9. Visual Inspection

Metrics alone are not enough.

The training workflow should save a small number of preview samples so the result can be inspected directly.

The preview should show:

```text
input image
ground-truth target
model prediction
overlay or comparison image
```

Only a few samples are needed, usually 3–4 per preview cycle.

The purpose is to check whether the model is learning the actual floorplan structure, not just improving a number.

---

## 10. Checkpointing and Model Safety

The model should be saved safely, but not waste disk space by saving every epoch archive.

The expected saved models are:

```text
latest.pt
best.pt
```

`latest.pt` is used to resume interrupted training.

`best.pt` is used for evaluation and future inference.

Historical epoch archives should not be saved by default unless explicitly enabled.

Model and checkpoint names should include a clear version name, such as:

```text
segformer_b0_v005
```

This prevents overwriting older trained models when the spec or training intention changes.

---

## 11. Hardware Assumption

The active development machine has an NVIDIA RTX 5080 Laptop GPU.

Training should use CUDA/GPU rather than CPU.

CPU training is not the intended workflow. If CUDA is unavailable, the script should warn clearly instead of silently continuing as if the environment is correct.

---

## 12. Boundary Between CNN and Vectorization

The CNN stage and the vectorization stage have different responsibilities.

The CNN stage should produce good semantic evidence:

```text
raster image
→ 7-class semantic mask
```

The vectorization stage, which comes later, will convert masks into geometry:

```text
semantic mask
→ wall lines
→ floor polygon
→ windows
→ doors (from door_arc/door_leaf/door_origin evidence)
→ classified JSON / SVG
```

Architectural logic such as straightening walls, snapping corners, attaching openings to walls, and checking room adjacency belongs mainly to the vectorization stage.

However, the CNN output must still be clean enough for that later process to work. That is why pixel accuracy, door/window quality, and boundary quality are important during CNN evaluation.

**Implementation status note (task07):** the implemented vectorization code (`spec_v007_component_primitives.md`, `spec_v008_mask_to_vector.md`) still targets the retired 5-class scheme and has not yet been updated to consume run3's 7-class output directly — see the "Known Mismatch / Technical Debt" notes in those specs. This file describes the intended long-term boundary, not the current implementation state.

---

## 13. Future Raster-to-Vector Stage

This stage is not active yet, but it remains the long-term target of the project.

The future vectorization stage should take the predicted semantic masks and produce classified geometry.

Expected objects include:

```text
walls
windows
doors (reconstructed from door_arc/door_leaf/door_origin evidence)
rooms
adjacency relationships
```

The future vector process may include:

```text
mask cleanup
contour extraction
wall centerline extraction
line simplification
orthogonal snapping
room polygon reconstruction
opening-to-wall assignment
classified JSON / SVG export
```

This should not be mixed into the current CNN training spec.

---

## 14. Final Output Intention

The final output should be a structured CAD-like representation.

A future JSON output may look like:

```json
{
  "walls": [
    {
      "centerline": [[x1, y1], [x2, y2]],
      "thickness": 150
    }
  ],
  "windows": [
    {
      "line": [[x1, y1], [x2, y2]],
      "host_wall_id": 3
    }
  ],
  "doors": [
    {
      "origin": [[x1, y1], [x2, y2]],
      "opening": [[x1, y1], [x2, y2]],
      "arc_radius": 90,
      "host_wall_id": 5
    }
  ],
  "rooms": [
    {
      "polygon": [[...]],
      "class": "room",
      "area": 12.5
    }
  ],
  "adjacency": [
    ["room_1", "room_2"]
  ]
}
```

This sketch reflects the 7-class CNN evidence (`windows` and `doors` as separate classes, doors reconstructed from `door_arc`/`door_leaf`/`door_origin`). It does not represent a finished schema — the exact JSON schema should be defined later, after the vectorization stage is updated to consume run3 output (see the "Known Mismatch / Technical Debt" notes in `spec_v007_component_primitives.md` and `spec_v008_mask_to_vector.md`). The current implemented `outputs/vectorization/v008` pipeline does not produce this schema; it still emits the older `floor/wall/opening/icon` SVG groups.

---

## 15. Current Project Priority

The current priority is:

```text
make the 7-class CNN segmentation model (segformer_b0_run3) reliable
```

A successful current-stage result means:

```text
1. Clean plans produce accurate masks.
2. Original raster plans produce plausible masks.
3. Windows and door subclasses are recognized reliably.
4. Floor regions remain spatially coherent.
5. Predictions look suitable for later vectorization.
6. Model checkpoints are versioned and safe.
7. Training uses GPU and does not silently fall back to CPU.
```

Only after this stage is satisfactory should the project move into raster-to-vector conversion and topology correction.

---

## 16. Summary

Neural Floorplan is currently a semantic segmentation project with a CAD-generation goal.

The immediate objective is not to produce final CAD geometry. The immediate objective is to train a reliable 7-class floorplan segmentation model (`segformer_b0_run3`) that preserves both pixel-level accuracy and architectural intention.

The long-term objective is:

```text
raster floorplan
→ semantic segmentation
→ clean vector geometry
→ classified CAD-like output
```

The current milestone is complete when the CNN output is accurate, visually plausible, and ready to become the input for a future raster-to-vector pipeline.
