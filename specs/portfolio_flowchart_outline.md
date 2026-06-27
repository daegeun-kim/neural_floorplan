# Portfolio Flowchart Outline

## Purpose

This file outlines the portfolio diagrams for the neural floorplan project.

The diagrams should explain the project as a staged research workflow:

```txt
CubiCasa SVG / raster data
-> semantic mask generation
-> SegFormer training
-> vectorization attempts
-> pretrained Raster-to-Graph inference
-> future CAD-like SVG / JSON
```

The goal for the portfolio is not to show every implementation detail. The goal is to show the evolution of the idea: each phase solved one limitation and exposed the next bottleneck.

## Diagram Set

Recommended Illustrator boards:

```txt
Board 1: Overall pipeline
Board 2: Phase 1 - 5-class segmentation to line vectors
Board 3: Phase 2 - 7-class semantic vectorization
Board 4: Phase 3 - point-based graph vectorization
Board 5: Phase 4 - Raster-to-Graph inference
Board 6: Phase comparison / lessons learned
```

## Visual Legend

Suggested node types:

```txt
Input / dataset          rounded rectangle
Generated training label rectangle
Model                    bold rectangle
Vectorization logic      hexagon or process block
Output artifact          document/file icon
Failure / limitation     red or warning callout
Direction change         arrow label or highlighted bridge
```

Suggested color roles:

```txt
raw/source data          light gray
semantic masks           blue
trained model            purple
rule-based vectorization orange
graph representation     green
failure/limitation       red
final/future CAD output  black or dark neutral
```

## Board 1: Overall Pipeline

Main left-to-right flow:

```txt
CubiCasa5K dataset
  -> model.svg
  -> model_clean.png
  -> semantic mask generation
  -> 7-class masks
  -> SegFormer-B0 training
  -> predicted semantic masks
  -> vectorization attempts
  -> wall graph / CAD-like output
```

Add side branch for original raster:

```txt
F1_scaled.png
  -> SegFormer input variant
```

Add side branch for Phase 4:

```txt
model.svg
model_clean.png
  -> Raster-to-Graph inference
  -> predicted wall graph
```

Short caption:

```txt
The project began as semantic segmentation, then shifted toward graph prediction when rule-based vectorization became the bottleneck.
```

## Data Preparation Block

Use this before the phase boards or as a top band in Board 1.

```txt
Input assets
  model.svg
  model_clean.png
  F1_scaled.png

SVG/raster preprocessing
  align SVG and raster coordinate system
  render clean raster from SVG

Semantic mask generation
  background
  floor
  wall
  window
  door_arc
  door_leaf
  door_origin

Training dataset
  input image
  semantic_class_map.png
  train / val / test splits
```

Suggested compact Illustrator text:

```txt
SVG annotations become pixel labels.
Clean SVG renders and original rasters share the same semantic target.
```

## SegFormer Training Block

Flow:

```txt
model_clean.png / F1_scaled.png
  -> SegFormer-B0 frozen backbone
  -> custom FloorplanDecoder
  -> 7-class semantic prediction
  -> evaluation
```

Evaluation nodes:

```txt
pixel accuracy
foreground mIoU
wall IoU
opening boundary F1
preview overlays
```

Output node:

```txt
Predicted 7-class floorplan mask
```

Short caption:

```txt
The segmentation model learned useful architectural evidence, but pixel classes alone did not guarantee clean vector geometry.
```

## Board 2: Phase 1 - 5-Class Segmentation To Line Vectors

Phase label:

```txt
Phase 1
5-class mask -> line vectorization
```

Flow:

```txt
Raster / SVG-derived input
  -> 5-class semantic mask
     background
     wall
     opening
     room
     icon
  -> line / contour extraction
  -> wall and opening vectors
  -> SVG output
```

Failure callouts:

```txt
Doors and windows were merged into one generic opening class.
Door swing and hinge evidence were missing.
Wall topology was inferred from pixels, not architecture.
```

Lesson callout:

```txt
The segmentation classes needed to be designed for vectorization, not only visual recognition.
```

Output folder note:

```txt
outputs/vectorization/phase1_5class_line_vectorization/
```

## Board 3: Phase 2 - 7-Class Semantic Vectorization

Phase label:

```txt
Phase 2
7-class mask -> semantic vectorization
```

Flow:

```txt
model.svg
  -> 7-class semantic masks
  -> SegFormer-B0 run3
  -> predicted wall / window / door evidence
  -> component extraction
  -> wall, window, door SVG primitives
```

Class node:

```txt
background
floor
wall
window
door_arc
door_leaf
door_origin
```

Improvement callouts:

```txt
Windows became explicit.
Door swing evidence became visible.
Door origin and door leaf gave stronger geometric hints.
```

Failure callouts:

```txt
Direct pixel-to-vector conversion was still unstable.
Wall topology remained implicit.
Openings could float or fail to host on walls.
Clean orthogonal SVG output was unreliable.
```

Lesson callout:

```txt
Better semantic evidence helped, but topology still needed an explicit graph.
```

Output folder note:

```txt
outputs/vectorization/phase2_7class_semantic_vectorization/
```

## Board 4: Phase 3 - Point-Based Graph Vectorization

Phase label:

```txt
Phase 3
7-class mask -> architectural points -> graph
```

Flow:

```txt
Predicted 7-class mask
  -> connected components
  -> architectural point recognition
  -> orthogonal alignment
  -> point connection graph
  -> wall / window / door primitives
  -> SVG + debug overlay + metrics
```

Point recognition node:

```txt
wall endpoints
wall corners
T-junctions
cross-junctions
window endpoints
door hinge points
door end points
```

Later simplification node:

```txt
wall subtypes -> generic wall point
red door_arc cluster -> trusted door anchor
```

Improvement callouts:

```txt
Topology became explicit.
Alignment and graph connection became separate steps.
Debug overlays and metrics made failures easier to inspect.
```

Failure callouts:

```txt
Point recognition accuracy was too demanding.
Missing or shifted keypoints broke downstream graph construction.
Door/window pairing remained fragile.
Many thresholds and special cases accumulated.
```

Lesson callout:

```txt
The graph was the right representation, but hand-written point detection was too brittle.
```

Output folder note:

```txt
outputs/vectorization/phase3_7class_point_vectorization/
```

## Board 5: Phase 4 - Raster-To-Graph Inference

Phase label:

```txt
Phase 4
model_clean.png -> pretrained wall graph inference
```

Input-preprocessing flow:

```txt
model_clean.png
  -> crop dark content bbox
  -> add true 20% white margin
  -> long edge scaled to 512 px
  -> centered on white 512x512 canvas
  -> Raster-to-Graph model input
```

Model flow:

```txt
checkpoint0299.pth
  -> generous autoregressive inference
  -> hard angle filter
  -> soft validity scoring and reranking
  -> mask-and-rerun multistart recovery
  -> merge-on-intersection
  -> light post-merge filtering
  -> predicted wall graph
```

Output flow:

```txt
predicted graph nodes
  -> predicted wall edges
  -> graph_pred.json
  -> graph_pred.svg
  -> graph_overlay.png
  -> future final_vector.svg / final_vector.json
```

Why clean input:

```txt
model_clean.png isolates wall graph prediction from raster noise.
F1_scaled.png / messy original rasters are not used in the current Phase 4 method.
```

Improvement callouts:

```txt
Wall topology is predicted directly as a graph.
The method avoids brittle hand-written point detection.
Generous inference improves graph production rate.
Validity scoring removes many bad generated edges.
```

Risk callouts:

```txt
Original pretrained model was trained on a different floorplan style.
Autoregressive graph prediction can fail early if confidence is low.
Preprocessing and margin placement strongly affect graph generation.
```

Output folder note:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/<sample>/
```

## Board 6: Phase Comparison

Use a four-column comparison table or stacked timeline.

```txt
Phase 1
5-class mask
pixel/line extraction
main failure: openings too ambiguous

Phase 2
7-class mask
semantic component vectorization
main failure: topology still implicit

Phase 3
7-class mask
point recognition and graph connection
main failure: hand-written point detection too brittle

Phase 4
model_clean.png
pretrained Raster-to-Graph inference
current: predict topology directly with preprocessing, scoring, multistart, and merge cleanup
```

Suggested final caption:

```txt
The project evolved from pixel classification to graph learning: each phase made architectural structure more explicit.
```

## Optional One-Page Portfolio Story

If only one diagram fits the website, use this compressed flow:

```txt
CubiCasa SVG / raster data
  -> SVG-derived masks
  -> SegFormer 7-class segmentation
  -> Phase 1: 5-class line vectors
       failed: doors/windows ambiguous
  -> Phase 2: 7-class semantic vectors
       failed: topology implicit
  -> Phase 3: point-based graph rules
       failed: keypoint detection brittle
  -> Phase 4: Raster-to-Graph inference
       current: infer wall graph directly with pretrained R2G
  -> future CAD-like SVG / JSON
```

## Suggested Visual Hierarchy

Use thick arrows for the successful main research progression:

```txt
5-class masks
-> 7-class masks
-> point graph
-> pretrained graph model
```

Use thinner downward arrows for artifacts:

```txt
semantic_class_map.png
prediction.png
debug_overlay.png
wall_graph.json
graph_pred.svg
metrics.json
```

Use red side callouts for failures instead of making the diagram feel like each phase simply "failed." The story should read as iterative discovery.
