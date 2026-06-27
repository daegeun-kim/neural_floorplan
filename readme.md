# Neural Floorplan To Classified CAD

This project explores how to convert raster floor plans into clean, classified, CAD-like geometry.

The current work has two major tracks:

```txt
1. semantic understanding:
   raster/SVG floorplan -> 7-class semantic mask

2. vector reconstruction:
   semantic mask or SVG-derived raster -> wall graph / CAD-like geometry
```

The segmentation track is working reasonably well. The vectorization track has now settled on a pretrained Raster-to-Graph inference pipeline for Phase 4 wall graph extraction.

---

## Current Direction

The project is currently settled on **Phase 4 Raster-to-Graph inference**:

```txt
CubiCasa model_clean.png
-> true white-padding preprocessing, long edge to 512 px
-> pretrained Raster-to-Graph checkpoint inference
-> generous graph generation + validity scoring/filtering
-> mask-and-rerun multistart recovery
-> merged orthogonal wall graph
-> later door/window attachment and CAD-like output
```

No Raster-to-Graph fine-tuning is planned for the current version because the adjusted inference method is producing satisfactory wall graphs.

This direction is documented in:

```txt
specs/spec_v005_raster2graph.md
specs/spec_v010_raster2graph_modifications.md
specs/vectorization_phase_history.md
```

---

## Pipeline

| Stage | Description | Status |
|---:|---|---|
| 1 | Dataset loading: CubiCasa5K `high_quality_architectural` subset | done |
| 2 | SVG/raster preprocessing: render CubiCasa SVG annotations to aligned rasters | done |
| 3 | Semantic mask generation: 7-class masks for background/floor/wall/window/door components | done |
| 4 | Sketch-style augmentation | done |
| 5 | SegFormer-B0 segmentation training, active run `segformer_b0_run3` | done |
| 6 | Segmentation evaluation and preview generation | done |
| 7 | Component/vector primitive experiments | historical / superseded |
| 8 | Mask-to-vector experiments | phase history, still useful for lessons learned |
| 9 | SVG-derived wall graph label generation (`masks/wall_graph.json`) for QA/reference | available / optional |
| 10 | Pretrained Raster-to-Graph inference from preprocessed `model_clean.png` | current / settled |
| 11 | Classified CAD-like JSON/SVG export from graph + openings | future |

---

## Vectorization Phase History

Vectorization has gone through four conceptual phases.

### Phase 1 - 5-Class Segmentation To Line Segments

```txt
5-class segmented raster
-> pixel/line extraction
-> vector output
```

Problem: the five-class scheme did not distinguish doors and windows clearly. Opening evidence was too ambiguous, so vector accuracy was extremely low.

Related output folders:

```txt
outputs/vectorization/v008/iteration1_run1_failed
outputs/vectorization/v008/iteration2_run1_failed
outputs/vectorization/v008/iteration2_run2_failed
```

### Phase 2 - 7-Class Segmentation With Door/Window Hints

```txt
7-class segmented raster
-> richer semantic pixel evidence
-> rule/CV vectorization
```

This removed furniture-like targets and added:

```txt
window
door_arc
door_leaf
door_origin
```

Goal: provide stronger spatial hints for doors, windows, and circulation rather than pixel-perfect raster conversion.

Problem: direct pixel/line conversion still struggled with wall topology and clean CAD reconstruction.

Related output folders:

```txt
outputs/vectorization/v008/iteration3_run2_failed
outputs/vectorization/v008/iteration4_run3_failed
```

### Phase 3 - 7-Class Segmentation To Point-Based Vectorization

```txt
7-class segmented raster
-> component cleanup
-> point recognition
-> axis alignment
-> graph edges
-> SVG/debug/metrics
```

This attempted to identify wall points, window points, and door hinge/end points, then connect them into a graph.

Problem: point recognition and spatial logic were too brittle for simple CV/rule-based extraction.

Related output folder:

```txt
outputs/vectorization/v008/iteration5_run3
```

### Phase 4 - Pretrained Raster-To-Graph Inference

```txt
-> model_clean.png
-> crop content bbox
-> add true 20% white padding
-> long edge to 512 px on white canvas
-> pretrained Raster-to-Graph checkpoint
-> generous autoregressive graph inference
-> hard/soft graph validity scoring
-> mask-and-rerun multistart recovery
-> merge-on-intersection and light post-merge filtering
-> wall graph JSON/SVG/overlays
```

This is the current settled direction. Instead of training a new model, the project uses the official Raster-to-Graph checkpoint and adapts preprocessing, thresholds, candidate scoring, recovery, and graph cleanup around this project's `model_clean.png` inputs.

Output graph:

```txt
nodes = wall endpoints / wall junctions
edges = orthogonal wall segments
```

Phase 4 currently predicts wall graph only. Doors and windows can be attached later using the existing 7-class semantic evidence.

---

## Segmentation Model

**Active run:** `segformer_b0_run3`

Classes:

```txt
0 background
1 floor
2 wall
3 window
4 door_arc
5 door_leaf
6 door_origin
```

Architecture:

```txt
SegFormer-B0 frozen backbone
custom trainable FloorplanDecoder
7-class output
```

Earlier `run1` and `run2` checkpoints are historical and are kept only for comparison.

---

## Dataset

Source:

```txt
CubiCasa5K high_quality_architectural
```

Key per-sample files:

```txt
F1_scaled.png
model_clean.png
model.svg
masks/
```

Phase 4 adds:

```txt
optional masks/wall_graph.json
optional masks/wall_graph_debug.svg
optional masks/wall_graph_debug.png
```

Phase 4 model outputs should be stored under:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/<sample>/
```

Each sample folder keeps its input PNG, graph JSON/SVG, overlays, metrics, and component diagnostics together. The old `outputs/raster2graph/` folder was testing-only and is retired.

---

## Important Specs

Core data/training specs:

```txt
specs/spec_v002_svg_to_raster.md
specs/spec_v003_semantic_mask_generation.md
specs/spec_v004_sketch_augmentation.md
specs/spec_v005_segformer_train.md
specs/spec_v006_evaluation.md
```

Vectorization history and current/future specs:

```txt
specs/vectorization_phase_history.md
specs/vectorization_must_rules.md
specs/spec_v003-1_graph_generation.md
specs/spec_v005_raster2graph.md
specs/spec_v010_raster2graph_modifications.md
```

Historical vectorization specs:

```txt
specs/spec_v007_component_primitives.md
specs/spec_v008_mask_to_vector.md
specs/spec_v009_cad_json.md
```

---

## External Attribution

Phase 4 uses adapted code and pretrained checkpoint experiments from the official Raster-to-Graph implementation:

```txt
Hu, S., Wu, W., Su, R., Hou, W., Zheng, L., and Xu, B.
Raster-to-Graph: Floorplan Recognition via Autoregressive Graph Prediction with an Attention Transformer.
Computer Graphics Forum, 43(2), e15007, 2024.
```

Upstream project:

```txt
https://github.com/SizheHu/Raster-to-Graph
```

Paper DOI:

```txt
https://doi.org/10.1111/cgf.15007
```

BibTeX:

```bibtex
@article{hu2024rastertograph,
  author = {Hu, Sizhe and Wu, Wenming and Su, Ruolin and Hou, Wanni and Zheng, Liping and Xu, Benzhu},
  title = {Raster-to-Graph: Floorplan Recognition via Autoregressive Graph Prediction with an Attention Transformer},
  journal = {Computer Graphics Forum},
  year = {2024},
  volume = {43},
  number = {2},
  pages = {e15007},
  doi = {10.1111/cgf.15007},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.15007}
}
```

The copied/adapted implementation lives under:

```txt
external/raster_to_graph/
```

The upstream Raster-to-Graph repository is GPL-3.0 licensed. This project should keep external code attribution visible, preserve upstream license notices when modifying copied files, and cite the paper in portfolio/research writeups that use Raster-to-Graph results.

---

## Setup

```bash
conda create -n floorplan-cad python=3.11
conda activate floorplan-cad
pip install -e .
```

Run tests:

```bash
pytest
```

Train active segmentation model:

```bash
python -m src.train_segmentation --config configs/train_segformer_b0_run3.yaml
```

---

## Development Workflow

Development is spec/task driven.

Rules:

```txt
1. Write or update specs/tasks before major implementation work.
2. Keep readme.md and workflow.md current when project direction changes.
3. Record vectorization attempt history in specs/attempt_history.md.
4. Do not remove outdated specs yet; they document the phase history.
5. Keep generated outputs out of Git unless explicitly approved.
```

See also:

```txt
CODEX.md
workflow.md
```
