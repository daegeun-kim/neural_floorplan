# Spec: Segmentation CNN Training for Neural Floorplan

## 0. Scope of This Spec

This document defines only the CNN segmentation training design for Neural Floorplan.

This spec covers:

```text
raster floorplan image → 5-class semantic segmentation map
```

This spec does not cover:

```text
semantic mask → vector geometry
semantic mask → topology correction
semantic mask → classified JSON / SVG export
```

Raster-to-vector conversion, topology validation, CAD cleanup, snapping, room graph construction, and final SVG/JSON export belong to later pipeline stages.

This version keeps the project size fixed. Do not introduce additional labels, auxiliary semantic classes, hinge labels, corner labels, door-swing labels, room-instance labels, or new training datasets in this spec.

The model remains a 5-class semantic segmentation model.

---

## 1. Purpose

Train a semantic segmentation model that predicts floorplan component masks from raster floorplan images.

The model should learn:

```text
floorplan raster image → semantic class map
```

not:

```text
floorplan raster image → SVG directly
```

The predicted semantic masks will later be converted into vector geometry through mask-to-vector post-processing, but that process is outside the scope of this training spec.

The CNN stage is successful when it produces accurate and visually plausible 5-class masks that are suitable for later vectorization.

---

## 2. Pipeline Context

The full Neural Floorplan pipeline is:

```text
1. Dataset loading
2. SVG/raster preprocessing
3. Semantic mask generation
4. Sketch-style augmentation
5. Segmentation model training
6. CNN evaluation
7. Mask-to-vector post-processing
8. Classified JSON / SVG export
```

This spec covers only:

```text
5. Segmentation model training
6. CNN evaluation
```

The training script must not generate semantic masks, perform mask-to-vector conversion, create topology graphs, or export final SVG/JSON.

---

## 3. Training Objective

Train a model that receives a clean, original, or lightly augmented floorplan raster image and predicts a pixel-level semantic class map.

Example:

```text
Input X:
F1_scaled.png        (original CubiCasa5K raster — primary real-world input)
model_clean.png      (SVG-rendered raster with all floors visible)
augmented_image.png

Target y:
semantic_class_map.png
```

The output should classify every pixel into one of the 5 semantic classes.

Pixel-level accuracy remains important because a clean, well-drawn input plan should produce a clean semantic mask suitable for vectorization.

However, the best model should not be selected by pixel accuracy or mIoU alone. Opening quality, foreground quality, and boundary quality must also be considered because circulation depends strongly on openings.

---

## 4. Semantic Classes

Use the same class IDs produced by semantic mask generation.

| Class ID | Class Name |
|---:|---|
| 0 | background |
| 1 | wall |
| 2 | opening |
| 3 | room |
| 4 | icon |

The training script must read class definitions from config or metadata. Do not hard-code class definitions in multiple files.

Strict rule:

```text
Do not add new semantic classes in v005.
```

Do not add:

```text
door hinge
corner
junction
door swing
door panel
wall centerline
room instance
```

These may be considered in a future project phase, but they are excluded from this spec.

---

## 5. Dataset Input Rules

Each training sample should be one raster image paired with one semantic target mask.

Valid input rasters:

```text
F1_scaled.png        (original CubiCasa5K raster — primary real-world input)
model_clean.png      (SVG-rendered clean raster)
augmented_image.png  (optional generated during training or preprocessing)
```

Ignore:

```text
F1_original.png
F2_original.png
F2_scaled.png
model_clean01.png
other unrelated png files
```

The target for all valid input rasters is:

```text
masks/semantic_class_map.png
```

The training script must never silently create new labels. If the target mask does not exist, the sample should be skipped with a warning or the script should raise an error, depending on config.

---

## 6. Shared Target Rule

If a sample folder contains both `F1_scaled.png` and `model_clean.png`, treat them as two separate training samples sharing the same semantic mask target.

Example:

```json
[
  {
    "sample_id": "0001_F1_scaled_png",
    "image": "0001/F1_scaled.png",
    "target": "0001/masks/semantic_class_map.png",
    "input_type": "original_raster"
  },
  {
    "sample_id": "0001_model_clean_png",
    "image": "0001/model_clean.png",
    "target": "0001/masks/semantic_class_map.png",
    "input_type": "svg_rendered_clean"
  }
]
```

Do not duplicate the mask file physically. Duplicating the dataset index is enough.

The split files must preserve `input_type` so validation metrics can be reported separately for clean and original rasters.

---

## 7. Model Architecture

Use a pretrained SegFormer backbone as the feature extraction encoder.

Recommended first backbone:

```text
SegFormer-B0
```

Reason:

```text
lightweight
fast iteration
lower VRAM usage
sufficient for first CNN pipeline validation
```

The architecture should consist of:

```text
1. Pretrained SegFormer-B0 backbone / encoder
2. Cached backbone feature extraction stage
3. Custom floorplan segmentation decoder / head
4. Final 5-class semantic classification layer
```

The SegFormer backbone extracts hierarchical visual features from floorplan rasters.

The custom decoder/head is responsible for:

```text
floorplan-specific feature learning
semantic component separation
spatial reconstruction
pixel-level classification
```

The decoder/head should use convolutional layers rather than fully connected layers to preserve spatial structure.

### Architecture recommendation after the updated training intention

The architecture recommendation does not change immediately.

Keep SegFormer-B0 with the existing custom 2-layer CNN decoder/head for v005 because the project should stay compact and the output remains 5 classes.

If opening quality and clean/original validation results plateau, the next recommended change is not new labels. The next recommended change is:

```text
partial fine-tuning of later SegFormer stages
```

not:

```text
new semantic classes
new datasets
larger model first
```

---

## 8. Frozen Backbone Feature Cache Mode

For the initial frozen-backbone training phase, the SegFormer backbone should not run during every training batch.

Use a feature-caching workflow:

```text
1. Load pretrained SegFormer-B0 backbone
2. Freeze backbone parameters
3. Run each training image through the backbone once
4. Save extracted feature tensors to disk
5. Train only the custom decoder/head using cached features
6. Do not call SegFormer forward pass during head-only training
```

This avoids repeatedly running expensive backbone computation during every epoch.

### Cached Feature Storage

Create:

```text
features/
  sample_id.pt
```

Each cached feature file should contain:

```text
SegFormer feature tensors
sample_id
image_path
target_mask_path
input_type
feature_shape
backbone_name
preprocessing_config_hash
```

Cached features must be regenerated only if:

```text
image preprocessing changes
backbone architecture changes
input resolution changes
normalization changes
feature extraction code changes
```

---

## 9. Head-Only Training Mode

During frozen-backbone training:

```text
Input:
cached SegFormer features

Target:
semantic_class_map.png
```

The SegFormer backbone should not execute forward propagation during this stage.

Acceptance criteria:

```text
SegFormer forward pass is not called during head-only training
only decoder/head parameters require gradients
training speed is significantly faster than full forward training
GPU memory usage is reduced compared to end-to-end training
```

---

## 10. Decoder / Head Architecture

The custom decoder/head should consist of:

### Hidden Layer 1

```text
Conv 3×3
256 channels
BatchNorm2d
GELU activation
Dropout2d(0.1)
```

### Hidden Layer 2

```text
Conv 3×3
128 channels
BatchNorm2d
GELU activation
Dropout2d(0.1)
```

### Final Classification Layer

```text
1×1 Conv
outputs 5 semantic classes
```

Output logits shape:

```text
[B, 5, H, W]
```

Do not add auxiliary heads in v005.

---

## 11. Model Output

For an input image:

```text
X shape: [B, C, H, W]
```

The model should output logits:

```text
logits shape: [B, num_classes, H, W]
```

The target mask should be:

```text
y shape: [B, H, W]
```

where each pixel value is the class ID.

The model itself should be saved as PyTorch checkpoints. Do not save probability maps for every validation sample during training.

For visual inspection, save only a small fixed set of 3–4 preview samples at the configured preview interval.

---

## 12. Image Size

Start with:

```text
512 × 512
```

Reason:

```text
manageable GPU memory
fast iteration
sufficient for first segmentation experiment
compatible with RTX 5080 Laptop GPU training
```

Later test:

```text
768 × 768
1024 × 1024
```

Do not start with full-resolution floorplans.

---

## 13. Device Requirement

Training should use the laptop GPU, not CPU.

Required device:

```text
NVIDIA RTX 5080 Laptop GPU
```

The script must check CUDA availability before training.

Required behavior:

```python
torch.cuda.is_available() must be True
```

If CUDA is unavailable, the script should stop with a clear error unless config explicitly allows CPU debugging.

Recommended config:

```yaml
device:
  require_cuda: true
  preferred_name_contains: "NVIDIA"
  allow_cpu_debug: false
```

The script should log:

```text
selected device
CUDA version
GPU name
total GPU memory
mixed precision status
```

CPU training is allowed only for tiny smoke tests if explicitly enabled.

---

## 14. Loss Function

Use a combined loss:

```text
loss = weighted_cross_entropy + dice_weight * dice_loss
```

Recommended first setting:

```text
loss = weighted_CE + 0.5 * DiceLoss
```

Reason:

```text
CrossEntropyLoss supports stable pixel-level classification.
DiceLoss improves global foreground shape and overlap.
Together they balance pixel accuracy and global semantic region quality.
```

DiceLoss should be multiclass Dice computed from softmax probabilities.

Recommended Dice behavior:

```text
include wall, opening, room, icon
exclude background from Dice average by default
```

Background should still be included in CrossEntropyLoss because background classification affects clean mask quality.

---

## 15. Class Imbalance and Class Weights

Floorplan segmentation has class imbalance.

Expected imbalance:

```text
background is dominant
room may be large
walls are thin but visually important
openings are small but architecturally important
icons vary by sample
```

Because the appropriate weights are uncertain, do not hard-code fixed weights as the only behavior.

Use automatic class weights computed from the training split.

Recommended method:

```text
1. Count pixels per class over the training masks.
2. Compute frequency per class.
3. Use inverse square-root frequency weighting.
4. Normalize weights so the mean foreground weight is near 1.
5. Clip extreme weights to avoid unstable training.
6. Apply an opening-priority multiplier.
```

Recommended formula:

```text
base_weight[c] = 1 / sqrt(freq[c] + epsilon)
normalized_weight = base_weight / mean(base_weight[foreground_classes])
final_weight = normalized_weight * priority_multiplier[c]
final_weight = clip(final_weight, min_weight, max_weight)
```

Recommended priority multipliers:

| Class | Multiplier | Reason |
|---|---:|---|
| background | 0.50 | prevent background dominance |
| wall | 0.80 | wall pixel perfection is less important than overall CAD intention |
| opening | 1.80 | circulation and door/window placement are high priority |
| room | 1.00 | room area matters for spatial layout |
| icon | 1.00 | useful but secondary |

Recommended clipping:

```text
min_weight: 0.10
max_weight: 5.00
```

The computed weights must be saved into:

```text
training_summary.json
checkpoint metadata
```

Config should allow either:

```yaml
class_weights:
  mode: "auto"
```

or:

```yaml
class_weights:
  mode: "manual"
  values: [0.5, 0.8, 1.8, 1.0, 1.0]
```

Default should be:

```text
auto
```

---

## 16. Metrics

Log the following core metrics:

```text
train_loss
val_loss
mean IoU
foreground_mIoU
per-class IoU
pixel accuracy
foreground pixel accuracy
learning rate
epoch time
```

Required per-class IoU:

```text
background_IoU
wall_IoU
opening_IoU
room_IoU
icon_IoU
```

Required boundary metrics:

```text
wall_boundary_F1
opening_boundary_F1
```

Boundary metrics must be implemented immediately in v005.

Boundary F1 should be computed from the existing 5-class masks. It must not require new labels.

Recommended boundary extraction:

```text
binary_mask → boundary = mask XOR erode(mask)
```

Recommended tolerance:

```text
2 pixels at 512 × 512
```

Boundary precision/recall:

```text
precision = predicted boundary pixels within tolerance of target boundary / predicted boundary pixels
recall    = target boundary pixels within tolerance of predicted boundary / target boundary pixels
F1        = 2 * precision * recall / (precision + recall)
```

If boundary computation significantly slows training, compute it only during validation, not during every training batch.

---

## 17. Input-Type Metrics

Validation metrics must be reported both overall and separated by input type.

Required groups:

```text
overall validation
svg_rendered_clean validation
original_raster validation
```

For each group, log:

```text
pixel_accuracy
foreground_pixel_accuracy
foreground_mIoU
background_IoU
wall_IoU
opening_IoU
room_IoU
icon_IoU
wall_boundary_F1
opening_boundary_F1
vector_ready_score
```

This allows the model to be evaluated for both:

```text
clean well-drawn plan accuracy
messy/original raster generalization
```

If grouped metrics significantly slow validation, keep overall metrics mandatory and grouped metrics configurable. The default should still enable grouped metrics.

---

## 18. Primary Model Selection Metric

Do not select `best.pt` using only `val_mIoU`.

Primary checkpoint metric:

```text
val_vector_ready_score
```

The score should emphasize openings more than walls because openings strongly affect circulation.

Recommended formula:

```text
val_vector_ready_score =
    0.25 * pixel_accuracy
  + 0.25 * opening_IoU
  + 0.15 * opening_boundary_F1
  + 0.15 * foreground_mIoU
  + 0.10 * room_IoU
  + 0.05 * wall_IoU
  + 0.05 * icon_IoU
```

Rationale:

```text
pixel_accuracy remains important for clean vector output
opening_IoU receives high weight because circulation depends on openings
opening_boundary_F1 checks whether openings are spatially usable
foreground_mIoU checks global semantic quality excluding background dominance
room_IoU checks spatial area layout
wall_IoU is included but not over-weighted because slight wall dimension shifts may not change CAD intention
icon_IoU is included but secondary
```

If `opening_boundary_F1` is unavailable due to early debugging, use fallback:

```text
val_vector_ready_score_fallback =
    0.30 * pixel_accuracy
  + 0.30 * opening_IoU
  + 0.15 * foreground_mIoU
  + 0.10 * room_IoU
  + 0.10 * wall_IoU
  + 0.05 * icon_IoU
```

The checkpoint metadata must record which metric was used.

---

## 19. Dataset Split

Create explicit split files:

```text
splits/train.json
splits/val.json
splits/test.json
```

Do not randomly split differently every run unless using a fixed seed and saving the split.

Recommended split:

```text
train: 80%
val: 10%
test: 10%
```

For early debugging, create:

```text
splits/debug_train.json
splits/debug_val.json
```

with a very small subset.

The test set should not be used for checkpoint selection.

---

## 20. Config File

Create or update:

```text
configs/train_segformer_b0.yaml
```

Recommended config:

```yaml
run:
  version: "v005"
  run_name: "segformer_b0_v005_opening_weighted"

paths:
  dataset_root: "path/to/high_quality_architectural"
  train_index: "splits/train.json"
  val_index: "splits/val.json"
  test_index: "splits/test.json"

image:
  image_size: 512
  num_classes: 5

model:
  name: "segformer_b0"
  pretrained: true
  frozen_backbone: true
  use_cached_features: true
  decoder_version: "v005_head_256_128"

training:
  epochs: 50
  batch_size: 4
  learning_rate: 0.00006
  weight_decay: 0.01
  num_workers: 4
  mixed_precision: true
  seed: 42

loss:
  name: "weighted_ce_plus_dice"
  ce_weight: 1.0
  dice_weight: 0.5
  dice_exclude_background: true

class_weights:
  mode: "auto"
  method: "inverse_sqrt_frequency"
  priority_multipliers:
    background: 0.50
    wall: 0.80
    opening: 1.80
    room: 1.00
    icon: 1.00
  min_weight: 0.10
  max_weight: 5.00

metrics:
  compute_boundary_f1: true
  boundary_tolerance_px: 2
  compute_grouped_by_input_type: true
  primary_metric: "val_vector_ready_score"
  vector_ready_score:
    pixel_accuracy: 0.25
    opening_IoU: 0.25
    opening_boundary_F1: 0.15
    foreground_mIoU: 0.15
    room_IoU: 0.10
    wall_IoU: 0.05
    icon_IoU: 0.05

checkpoint:
  output_dir: "checkpoints/segformer_b0_v005"
  save_best: true
  save_latest: true
  save_epoch_archives: false
  keep_last_n_archives: 0
  monitor: "val_vector_ready_score"
  mode: "max"
  resume_from: "auto"

device:
  require_cuda: true
  preferred_name_contains: "NVIDIA"
  allow_cpu_debug: false

logging:
  log_dir: "runs/segformer_b0_v005"
  save_preview_every_n_epochs: 5
  preview_sample_count: 4
  save_probability_maps_for_previews: false
```

---

## 21. Required Scripts

Create or update:

```text
src/train_segmentation.py
```

Recommended helper files:

```text
src/dataset.py
src/models.py
src/losses.py
src/metrics.py
src/checkpointing.py
src/config.py
```

The PyTorch model file should be organized with version naming. Do not overwrite previous model definitions silently.

Recommended model file organization:

```text
src/models/
  __init__.py
  segformer_b0_v004.py
  segformer_b0_v005.py
```

or:

```text
src/models.py
```

with explicit class names:

```python
SegFormerB0HeadV004
SegFormerB0HeadV005
```

The v005 model class should keep the same 5-class output but use the updated training objective, loss, metrics, and checkpoint naming.

Do not overwrite old checkpoints or old model files. Use a new checkpoint directory:

```text
checkpoints/segformer_b0_v005/
```

---

## 22. Data Loading

The PyTorch Dataset should return:

```python
{
    "image": image_tensor,
    "mask": mask_tensor,
    "sample_id": sample_id,
    "image_path": image_path,
    "mask_path": mask_path,
    "input_type": input_type,
}
```

Image tensor:

```text
float32
shape [3, H, W]
normalized
```

Mask tensor:

```text
long
shape [H, W]
values are class IDs
```

Important:

```text
image resize uses bilinear interpolation
mask resize uses nearest-neighbor interpolation
```

Never use bilinear interpolation for class masks.

---

## 23. Normalization

For pretrained SegFormer, use ImageNet normalization unless the selected library specifies otherwise.

Typical:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

If using grayscale floorplans, convert to 3-channel RGB before feeding into the pretrained backbone.

---

## 24. Augmentation During Training

For v005, keep augmentation controlled.

Allowed training-time augmentation:

```text
horizontal flip
vertical flip
90-degree rotation
small brightness/contrast change
very mild blur
minor line darkness variation
```

Avoid for v005:

```text
heavy line jitter
large broken-edge augmentation
large arbitrary rotations
elastic deformation
perspective transform
random erasing that destroys openings
```

Reason:

```text
The model must still perform strongly on clean, well-drawn floorplans.
```

Generalization is important, but not at the cost of losing clean-plan precision.

---

## 25. Small-Subset Smoke Test

Before full training, the script must run on a small subset.

Example:

```text
20 training samples
5 validation samples
2 epochs
```

Command:

```powershell
python -m src.train_segmentation --config configs/train_segformer_b0.yaml --debug
```

Acceptance for smoke test:

```text
script starts
CUDA device is detected unless CPU debug is explicitly enabled
dataset loads correctly
model forward pass works
loss computes
backward pass works
checkpoint saves
validation runs
boundary metrics compute
preview prediction saves
```

This must pass before full training.

---

## 26. Overfit Test

Before a full run, train on 5 samples.

Goal:

```text
model should overfit and produce plausible masks
```

Command:

```powershell
python -m src.train_segmentation --config configs/train_segformer_b0.yaml --overfit 5
```

If it cannot overfit 5 samples, there is likely a bug in:

```text
mask generation
class IDs
model output size
loss function
image/mask alignment
feature cache alignment
```

---

## 27. Checkpointing Requirements

Training may take a long time, so checkpointing is mandatory.

The script must save:

```text
checkpoints/segformer_b0_v005/latest.pt
```

for resume support.

The script must save:

```text
checkpoints/segformer_b0_v005/best.pt
```

whenever validation metric improves.

Do not save a permanent checkpoint archive for every epoch by default.

Default behavior:

```text
latest.pt is overwritten after each epoch
best.pt is overwritten only when val_vector_ready_score improves
epoch_XXX.pt archives are disabled by default
```

Optional archive behavior:

```yaml
save_epoch_archives: false
keep_last_n_archives: 0
```

If epoch archives are enabled, keep only the most recent N archive files.

---

## 28. Best Model Definition

Primary best model rule:

```text
best.pt = checkpoint with highest val_vector_ready_score
```

Fallback rule:

```text
if val_vector_ready_score is unavailable:
    best.pt = checkpoint with highest fallback vector-ready score
```

Debug-only fallback:

```text
if no IoU or boundary metrics are available:
    best.pt = checkpoint with lowest val_loss
```

The checkpoint metadata must clearly record which metric was used.

Example:

```json
{
  "best_metric_name": "val_vector_ready_score",
  "best_metric_value": 0.7421,
  "best_epoch": 17
}
```

---

## 29. Checkpoint Contents

Each checkpoint file must contain:

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict
epoch
global_step
best_metric_value
best_metric_name
config
class_mapping
class_weights
random_seed
training_history
model_version
run_name
```

Minimum PyTorch checkpoint structure:

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    "epoch": epoch,
    "global_step": global_step,
    "best_metric_value": best_metric_value,
    "best_metric_name": best_metric_name,
    "config": config,
    "class_mapping": class_mapping,
    "class_weights": class_weights,
    "random_seed": seed,
    "history": history,
    "model_version": "v005",
    "run_name": run_name,
}
```

---

## 30. Resume Training Rule

The training script must support resuming.

Config:

```yaml
checkpoint:
  resume_from: "auto"
```

Behavior:

```text
If resume_from == "auto":
    if checkpoints/segformer_b0_v005/latest.pt exists:
        load latest.pt and continue training
    else:
        start from pretrained SegFormer / cached features
```

Manual resume:

```powershell
python -m src.train_segmentation --config configs/train_segformer_b0.yaml --resume checkpoints/segformer_b0_v005/latest.pt
```

or:

```powershell
python -m src.train_segmentation --config configs/train_segformer_b0.yaml --resume checkpoints/segformer_b0_v005/best.pt
```

For continued training, use:

```text
latest.pt
```

For evaluation or inference, use:

```text
best.pt
```

unless explicitly testing a different checkpoint.

---

## 31. Latest vs Best

Use different files for different purposes:

| File | Purpose |
|---|---|
| `latest.pt` | continue interrupted training |
| `best.pt` | evaluation, inference, deployment |
| `epoch_XXX.pt` | optional historical backup only if enabled |

Important:

```text
latest.pt is not always the best model.
best.pt is not always the most recent model.
```

---

## 32. Training History

Save training history continuously.

Required file:

```text
checkpoints/segformer_b0_v005/training_history.csv
```

Required columns:

```text
epoch
train_loss
val_loss
val_vector_ready_score
val_mIoU
val_foreground_mIoU
pixel_accuracy
foreground_pixel_accuracy
background_IoU
wall_IoU
opening_IoU
room_IoU
icon_IoU
wall_boundary_F1
opening_boundary_F1
clean_pixel_accuracy
clean_foreground_mIoU
clean_wall_IoU
clean_opening_IoU
clean_opening_boundary_F1
original_pixel_accuracy
original_foreground_mIoU
original_wall_IoU
original_opening_IoU
original_opening_boundary_F1
learning_rate
checkpoint_saved
best_updated
```

Also save:

```text
checkpoints/segformer_b0_v005/training_summary.json
```

with the latest training state.

---

## 33. Preview Predictions

Save visual prediction previews every configured interval.

Default:

```yaml
save_preview_every_n_epochs: 5
preview_sample_count: 4
```

Preview outputs:

```text
runs/segformer_b0_v005/previews/epoch_005/
  sample_001_input.png
  sample_001_target.png
  sample_001_prediction.png
  sample_001_overlay.png
  sample_002_input.png
  sample_002_target.png
  sample_002_prediction.png
  sample_002_overlay.png
  sample_003_input.png
  sample_003_target.png
  sample_003_prediction.png
  sample_003_overlay.png
  sample_004_input.png
  sample_004_target.png
  sample_004_prediction.png
  sample_004_overlay.png
```

Purpose:

```text
verify that model is learning
catch label alignment bugs
catch class-color mapping bugs
inspect wall/opening/room quality
compare clean vs original raster behavior
```

The preview set should be fixed across epochs so improvements are visually comparable.

Do not save predictions for the full validation set during training.

---

## 34. Logging

Required:

```text
console logs
training_history.csv
training_summary.json
preview images
```

Optional:

```text
TensorBoard
Weights & Biases
```

Console log example:

```text
Epoch 07/50
train_loss=0.7421
val_loss=0.6812
val_vector_ready_score=0.6234
val_foreground_mIoU=0.5841
pixel_accuracy=0.9412
opening_IoU=0.4128
opening_boundary_F1=0.5069
best_val_vector_ready_score=0.6234
saved latest.pt
updated best.pt
```

---

## 35. Evaluation During Training

At the end of each epoch:

```text
1. Run validation.
2. Compute val_loss.
3. Compute pixel accuracy and foreground pixel accuracy.
4. Compute per-class IoU.
5. Compute foreground_mIoU.
6. Compute wall_boundary_F1 and opening_boundary_F1.
7. Compute val_vector_ready_score.
8. Compute grouped metrics by input_type if enabled.
9. Save latest.pt by overwriting the previous latest.pt.
10. Update best.pt if val_vector_ready_score improves.
11. Save 3–4 preview predictions if scheduled.
12. Update training_history.csv.
13. Update training_summary.json.
```

---

## 36. Inference Script

Create later:

```text
src/predict_segmentation.py
```

Expected command:

```powershell
python -m src.predict_segmentation `
--checkpoint checkpoints/segformer_b0_v005/best.pt `
--input path/to/floorplan.png `
--output outputs/predictions/
```

For inference, default to:

```text
best.pt
```

not:

```text
latest.pt
```

The inference output may save the hard 5-class prediction mask and visual preview. Full raster-to-vector output is not part of this spec.

---

## 37. Evaluation Script

Create later:

```text
src/evaluate_segmentation.py
```

Expected command:

```powershell
python -m src.evaluate_segmentation `
--checkpoint checkpoints/segformer_b0_v005/best.pt `
--split splits/test.json
```

The test set should not be used for checkpoint selection.

---

## 38. Output Directory Structure

Expected structure:

```text
neural_floorplan/
  configs/
    train_segformer_b0.yaml

  splits/
    train.json
    val.json
    test.json
    debug_train.json
    debug_val.json

  checkpoints/
    segformer_b0_v005/
      latest.pt
      best.pt
      training_history.csv
      training_summary.json

  runs/
    segformer_b0_v005/
      previews/
        epoch_001/
        epoch_005/
        epoch_010/

  features/
    sample_id.pt

  src/
    train_segmentation.py
    dataset.py
    losses.py
    metrics.py
    checkpointing.py
    config.py
    models/
      __init__.py
      segformer_b0_v005.py
```

---

## 39. Acceptance Criteria

The v005 training stage is complete when:

```text
1. Training script runs on a small subset.
2. CUDA GPU is detected and used by default.
3. Model forward/backward pass works.
4. Weighted CrossEntropyLoss computes correctly.
5. DiceLoss computes correctly.
6. Combined loss computes correctly.
7. Validation loop computes loss, IoU, pixel accuracy, foreground_mIoU, and boundary F1.
8. val_vector_ready_score is computed.
9. latest.pt is saved by overwriting the previous latest checkpoint.
10. best.pt is updated when val_vector_ready_score improves.
11. Per-epoch archive checkpoints are not saved by default.
12. Training can resume from latest.pt.
13. Training history is saved to CSV.
14. Training summary is saved to JSON.
15. 3–4 preview predictions are saved at the configured interval.
16. The model can overfit 5 samples.
```

---

## 40. Non-Goals

This training spec should not:

```text
generate semantic masks
create SVG output
perform mask-to-vector conversion
perform topology correction
build room adjacency graphs
infer door swing direction
create new semantic classes
create new labels
modify original dataset files
save full validation predictions every epoch
save permanent epoch checkpoints by default
```

Those belong to other project stages or future specs.

---

## 41. Practical First Milestone

First v005 implementation should support:

```text
SegFormer-B0
5 semantic classes
512 × 512 images
cached frozen-backbone feature mode
custom 2-layer CNN decoder/head
weighted CrossEntropyLoss + DiceLoss
automatic class weights
opening-weighted vector-ready score
wall/opening boundary F1
clean vs original raster validation metrics
latest.pt checkpoint
best.pt checkpoint
training_history.csv
training_summary.json
3–4 preview outputs
CUDA GPU requirement
```

After this works, possible later improvements are:

```text
partial SegFormer fine-tuning
larger image size
larger SegFormer backbone
stronger but controlled augmentation
improved raster-to-vector post-processing
```

Do not add new classes or new labels until the 5-class CNN reaches a satisfactory level.

---

## 42. Final Principle

The CNN training stage is successful if the model can produce accurate and plausible 5-class semantic masks.

Pixel accuracy remains important because clean, well-drawn plans must convert into clean vector output.

Opening quality receives extra emphasis because openings control circulation and affect whether a later vectorized plan makes architectural sense.

The best model should therefore balance:

```text
pixel accuracy
opening quality
foreground segmentation quality
room area quality
wall quality
icon quality
boundary usability
```

The full project remains:

```text
raster image
→ 5-class semantic segmentation
→ mask-to-vector conversion
→ topology validation/correction
→ classified JSON / SVG export
```

This spec covers only the CNN segmentation stage.
