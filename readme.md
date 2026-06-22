# Neural Floorplan to Classified CAD

A supervised deep-learning pipeline that converts raster floor plans — or color-coded sketches — into per-pixel semantic segmentation masks, and (as planned next steps) into clean classified vector geometry exported as structured JSON.

The pipeline addresses a practical data problem: architectural drawings in the wild exist as images, not editable CAD files. This project builds the ML infrastructure to turn those pixel representations back into typed, machine-readable geometry — walls with centerlines and thickness, openings attached to host walls, room polygons with inferred type labels.

---

## Pipeline

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Dataset loading — CubiCasa5K, `high_quality_architectural` subset | ✅ done |
| 2 | SVG / raster preprocessing — CubiCasa SVG annotations rendered to `model_clean.png` (spec_v002) | ✅ done |
| 3 | Semantic mask generation — per-class masks (floor, wall, window, door_arc, door_leaf, door_origin, background) (spec_v003) | ✅ done |
| 4 | Sketch-style augmentation — flip, rotate 90°, translate ±10px, blur, brightness (spec_v004) | ✅ done |
| 5 | Segmentation model training — SegFormer-B0, frozen backbone, custom FloorplanDecoder, AdamW, active run `segformer_b0_run3` (spec_v005) | ✅ done |
| 6 | Evaluation — loss and mIoU logged per epoch; best checkpoint saved by val mIoU (spec_v006) | ✅ done |
| 7 | Component primitives — parametric Wall/Opening/Door/Window/Room/Floor/Icon primitive classes (spec_v007) | 🔶 implemented, targets retired 5-class scheme — not yet updated for run3's 7 classes |
| 8 | Mask-to-vector post-processing — contour extraction, primitive fitting, SVG export (spec_v008) | 🔶 implemented, same 5-class mismatch as stage 7 — see spec_v008 "Known Mismatch / Technical Debt" |
| 9 | Classified CAD-like JSON export — walls, openings, room polygons with type labels (spec_v009) | 🔲 planned |

---

## Model Architecture

**Backbone:** SegFormer-B0 (`nvidia/mit-b0`), fully frozen. Pretrained ImageNet weights are loaded from HuggingFace Transformers. The backbone runs once per image to extract four multi-scale feature maps, which are cached to disk. The frozen forward pass is not called during training epochs.

**Decoder:** Custom `FloorplanDecoder` — trainable from scratch.
- 1×1 projection per backbone stage (channel sizes [32, 64, 160, 256] → 256)
- Upsample all stages to stage-0 resolution (H/4 × W/4), element-wise sum
- Hidden Layer 1: Conv 3×3 → 256 ch, BatchNorm2d, GELU, Dropout2d(0.1)
- Hidden Layer 2: Conv 3×3 → 128 ch, BatchNorm2d, GELU, Dropout2d(0.1)
- Classification: 1×1 Conv → 7 classes
- Bilinear upsample → [B, 7, 512, 512]

**Active run:** `segformer_b0_run3` — `background, floor, wall, window, door_arc, door_leaf, door_origin`. Earlier `segformer_b0_run1`/`segformer_b0_run2` checkpoints used a 5-class scheme (`background, wall, opening, room, icon`) and are kept only for historical comparison (see `specs/spec_v005_segformer_train.md`).

**Training config (from `configs/train_segformer_b0_run3.yaml`):**
- Input size: 512 × 512 px
- Classes: 7 (background, floor, wall, window, door_arc, door_leaf, door_origin)
- Loss: CrossEntropy
- Optimizer: AdamW (lr = 6e-5, weight decay = 0.01)
- Scheduler: CosineAnnealingLR
- Batch size: 4 · Epochs: 50 · Mixed precision: enabled · Seed: 42

---

## Dataset

**Source:** [CubiCasa5K](https://github.com/cubicasa/cubicasa5k) — ~5,000 residential floor plans with SVG vector annotations. `high_quality_architectural` subset used throughout.

**Data-quality strategy:** Each sample uses two training rows sharing one SVG-derived target mask:
- `F1_scaled.png` — the original CubiCasa raster, scaled to align with the SVG coordinate space.
- `model_clean.png` — a raster rendered directly from `model.svg`, guaranteeing exact raster↔mask alignment because both originate from the same source.

---

## Tech Stack

| Component | Library / Tool |
|-----------|---------------|
| Language | Python 3.11 |
| Deep learning | PyTorch (torch.amp mixed precision) |
| Model | HuggingFace Transformers — SegFormer |
| Raster processing | OpenCV, Pillow |
| Vector geometry | Shapely (planned, stages 7–8) |
| Testing | pytest |
| Lint / format | ruff |
| Environment | conda `floorplan-cad` |

---

## Setup

```bash
conda create -n floorplan-cad python=3.11
conda activate floorplan-cad
pip install -e .
```

**Run tests:**
```bash
pytest
```

**Format and lint:**
```bash
ruff format .
ruff check .
```

**Train (active run3 — frozen backbone + custom decoder, 7 classes):**
```bash
python -m src.train_segmentation --config configs/train_segformer_b0_run3.yaml
```

---

## Spec-driven workflow

Development follows a spec-driven workflow. Each feature is defined in a versioned spec file under `/specs` before any code is written. One spec is worked at a time; code is committed only after tests and lint pass. Experiment outputs (checkpoints, cached features, preview images) are kept out of Git unless explicitly approved.

Spec history: `spec_v002_svg_to_raster` · `spec_v003_semantic_mask_generation` (7-class, active) · `spec_v004_sketch_augmentation` · `spec_v005_segformer_train` (active, `segformer_b0_run3`) / `spec_v005_segformer_train_outdated` (historical 5-class) · `spec_v006_evaluation` · `spec_v007_component_primitives` · `spec_v008_mask_to_vector` (v007/v008 still target the retired 5-class scheme — see their "Known Mismatch / Technical Debt" notes) · `spec_v009_cad_json` (planned)
