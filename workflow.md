# Neural Floorplan Workflow

## 0. Purpose

This document explains how the project stages connect.

Detailed implementation rules belong in specs under `specs/`. This file is the top-level workflow map.

---

## 1. Project Goal

The long-term goal is:

```txt
raster floorplan
-> semantic understanding
-> vector wall graph
-> doors/windows attached to graph
-> clean CAD-like SVG / JSON
```

The project has learned that semantic segmentation alone is not enough. The difficult part is spatial reconstruction: wall graph accuracy, orthogonal alignment, junctions, and opening placement.

---

## 2. Current Active Direction

The current active direction is **Phase 4 pretrained Raster-to-Graph inference**.

Instead of continuing to add more OpenCV/rule-based vectorization logic, Phase 4 now uses the official Raster-to-Graph checkpoint with local preprocessing, generous inference thresholds, multistart recovery, and graph validity filtering.

```txt
input:
  model_clean.png, cropped to content, padded with true 20% white margin,
  scaled to 512 px, and normalized with the original R2G mean/std

model:
  external/raster_to_graph/ checkpoint0299.pth

output:
  graph_pred.json / graph_pred.svg / graph overlays / metrics
```

The graph target remains intentionally minimal:

```txt
nodes = wall endpoints / wall junctions
edges = orthogonal wall segments
```

Doors, windows, wall thickness, and rooms are not part of the current Phase 4 graph output.

---

## 3. Main Data Flow

Current intended data flow:

```txt
CubiCasa model.svg
-> model_clean.png
-> 7-class semantic masks
-> optional wall_graph.json for QA/reference
-> Phase 4 Raster-to-Graph inference from model_clean.png
-> predicted wall graph JSON/SVG
-> later attach doors/windows from semantic masks
-> CAD-like SVG / JSON
```

The semantic segmentation model remains useful for later door/window attachment and CAD classification, but the Raster-to-Graph input is `model_clean.png`, not the seven-class debug overlay.

---

## 4. Four Vectorization Phases

### Phase 1 - 5-Class Segmentation To Line Segments

Pipeline:

```txt
5-class segmented raster
-> pixel/line extraction
-> vector output
```

Problem:

```txt
doors and windows were not separated
opening evidence was ambiguous
vector accuracy was extremely low
```

Outputs:

```txt
outputs/vectorization/v008/iteration1_run1_failed
outputs/vectorization/v008/iteration2_run1_failed
outputs/vectorization/v008/iteration2_run2_failed
```

### Phase 2 - 7-Class Segmentation With Stronger Door/Window Hints

Pipeline:

```txt
7-class segmented raster
-> richer semantic pixel evidence
-> rule/CV vectorization
```

Improvement:

```txt
window
door_arc
door_leaf
door_origin
```

were separated to expose circulation and opening intent.

Problem:

```txt
direct pixel/line conversion still could not reliably recover clean wall topology
```

Outputs:

```txt
outputs/vectorization/v008/iteration3_run2_failed
outputs/vectorization/v008/iteration4_run3_failed
```

### Phase 3 - 7-Class Segmentation To Point-Based Vectorization

Pipeline:

```txt
7-class segmented raster
-> component cleanup
-> point recognition
-> axis alignment
-> graph construction
-> SVG/debug/metrics
```

Improvement:

```txt
explicit points
door bboxes
debug metrics
orthogonal graph attempts
```

Problem:

```txt
point recognition and spatial logic are too brittle for a simple CV/rule system
```

Output:

```txt
outputs/vectorization/v008/iteration5_run3
```

### Phase 4 - Pretrained Raster-To-Graph Wall Extraction

Pipeline:

```txt
model_clean.png
-> content bbox crop
-> true 20% white padding
-> 512 px Raster-to-Graph input
-> pretrained Raster-to-Graph checkpoint
-> generous autoregressive graph generation
-> validity scoring and hard filters
-> mask-and-rerun multistart recovery
-> merge-on-intersection
-> predicted orthogonal wall graph
```

Goal:

```txt
replace rule-only wall graph extraction with an adapted pretrained graph predictor
```

Main specs:

```txt
specs/spec_v003-1_graph_generation.md
specs/spec_v005_raster2graph.md
specs/spec_v010_raster2graph_modifications.md
```

---

## 5. Segmentation Stage

The active segmentation model is:

```txt
segformer_b0_run3
```

Classes:

```txt
background
floor
wall
window
door_arc
door_leaf
door_origin
```

Segmentation responsibility:

```txt
raster image -> useful semantic evidence
```

It should not directly output CAD geometry.

---

## 6. Graph Generation Stage

Optional Phase 4 graph reference labels can be generated from original SVGs:

```txt
docs/high_quality_architectural/<sample>/model.svg
```

Sample artifacts:

```txt
masks/wall_graph.json
masks/wall_graph_debug.svg
masks/wall_graph_debug.png
```

The source SVGs may contain complex layered wall drawings. Therefore graph generation should not assume SVG wall geometry is already clean CAD.

Preferred graph-label process:

```txt
model.svg
-> isolate/render wall evidence
-> wall mask
-> centerline/skeleton
-> orthogonal simplification
-> simple graph
-> debug visualization
```

Bad or uncertain graph labels should be flagged instead of silently used for QA, evaluation, or any future training fallback.

---

## 7. Raster-To-Graph Inference Stage

The raster-to-graph stage currently predicts only:

```txt
wall graph nodes
wall graph edges
```

Current implementation:

```txt
external/raster_to_graph/run_inference_generous_phase4.py
checkpoint: checkpoints_Raster2Graph/checkpoint0299.pth
output: outputs/vectorization/phase4_raster2graph_generous_inference/<sample>/
```

Current preprocessing:

```txt
model_clean.png
-> detect dark content bbox
-> crop exactly to content
-> add true 20% white padding
-> scale long edge to 512 px
-> center on white 512x512 canvas
-> normalize with original R2G mean/std
```

Current inference logic:

```txt
first_step_threshold = 0.02
later_step_threshold = 0.02
first_step_force_best = true
edge_search_threshold = 50 px
monte_times = 4
max_candidates_per_step = 40
max_new_starts = 2
angle hard filter = keep edges within +/-10 degrees of horizontal/vertical
soft reranking = wall evidence, rectangle cycles, dangling penalty, unsupported-edge penalty
```

Evaluation should include:

```txt
node accuracy
edge accuracy
structure accuracy
orthogonality
visual graph overlays
component overlays
stage-count metrics
```

---

## 8. Future CAD Output

After the wall graph is reliable, later stages can attach:

```txt
windows
doors
wall thickness
rooms
classified JSON
final SVG
```

Doors/windows should use the existing semantic evidence:

```txt
window mask
door_arc mask
door_leaf mask
door_origin mask
```

The wall graph stage does not need to solve all of those in the first version.

---

## 9. Documentation Rules

When major project direction changes happen:

```txt
update readme.md
update workflow.md
update relevant specs/tasks
update specs/attempt_history.md for vectorization attempts
```

When a new spec or task is requested:

```txt
ask clarification questions first if requirements are ambiguous
then generate the md file only after the ambiguity is resolved
```

---

## 10. Current Priority

The current priority is:

```txt
1. Treat Phase 4 pretrained Raster-to-Graph inference as the current wall graph method.
2. Keep Phase 4 outputs under outputs/vectorization/phase4_raster2graph_generous_inference/<sample>/.
3. Keep preprocessing standardized as crop512_margin20_truepad.
4. Preserve graph_pred.json, graph_pred.svg, overlays, metrics, and component diagnostics per sample.
5. Use the 7-class semantic model later for door/window attachment and classified CAD output.
6. Keep fine-tuning only as a future fallback if the settled inference method stops being sufficient.
```

The project should avoid expanding semantic classes or restarting Raster-to-Graph training while the current Phase 4 inference method is producing satisfactory graphs.
