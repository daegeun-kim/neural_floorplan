# Vectorization Phase History

## Purpose

This document organizes the project's vectorization attempts into four distinct phases.

The goal is to make it clear which outputs belong to which conceptual approach, and to separate:

```txt
CNN model generation
vectorization method generation
output result folder
```

In output folder names:

```txt
iterationN = distinct vectorization method
runN       = distinct CNN model generation
failed     = output did not meet required architectural/vectorization quality
```

## Output Folder Organization

Existing outputs are organized by phase under:

```txt
outputs/vectorization/
```

(task20: previously flat under `outputs/vectorization/v008/<name>`; see `outputs/vectorization/README.md` for the old-to-new path mapping and the rationale for retiring the `v008` folder name)

| Output folder | Phase | CNN run | Vectorization iteration | Status |
|---|---:|---:|---:|---|
| `phase1_5class_line_vectorization/iteration1_run1_failed` | 1 | run1 | iteration1 | failed |
| `phase1_5class_line_vectorization/iteration2_run1_failed` | 1 | run1 | iteration2 | failed |
| `phase1_5class_line_vectorization/iteration2_run2_failed` | 1 | run2 | iteration2 | failed |
| `phase2_7class_semantic_vectorization/iteration3_run2_failed` | 2 | run2 | iteration3 | failed |
| `phase2_7class_semantic_vectorization/iteration4_run3_failed` | 2 | run3 | iteration4 | failed |
| `phase3_7class_point_vectorization/iteration5_run3` | 3 | run3 | iteration5 | active/latest phase-3 output |
| `phase4_raster2graph_generous_inference/<sample>` | 4 | external pretrained checkpoint | Raster-to-Graph inference | current / settled |

## Phase 1 - Five-Class Segmentation To Line Segments

### Input Assumption

Phase 1 used a five-class segmentation approach.

The vectorizer attempted to convert segmented pixels into line segments and architectural primitives.

### Vectorization Strategy

The vectorization logic was primarily line/pixel driven:

```txt
segmented pixels
-> extract regions/edges/line-like evidence
-> convert pixels into wall/opening line segments
-> export SVG-like vector geometry
```

### Main Limitation

The five-class segmentation did not distinguish doors and windows clearly enough.

In particular:

```txt
door evidence and window evidence were too generic
opening type was ambiguous
circulation intent was not explicitly represented
door swing geometry was not available as strong evidence
```

### Result

Accuracy was extremely low because the vectorizer had to infer too much from broad classes.

Phase 1 output folders:

```txt
outputs/vectorization/phase1_5class_line_vectorization/iteration1_run1_failed
outputs/vectorization/phase1_5class_line_vectorization/iteration2_run1_failed
outputs/vectorization/phase1_5class_line_vectorization/iteration2_run2_failed
```

## Phase 2 - Seven-Class Segmentation With Door And Window Evidence

### Input Assumption

Phase 2 moved to seven semantic classes.

The key change was removing less useful furniture-style targets and adding stronger architectural evidence for openings:

```txt
wall
window
door_arc
door_leaf
door_origin
floor
background
```

### Vectorization Strategy

The vectorizer still largely tried to convert raster evidence into vector geometry, but the raster evidence became more specific.

The purpose of the new classes was not pixel-perfect conversion. The purpose was to provide stronger spatial hints:

```txt
window pixels identify hosted windows
door_origin pixels identify threshold/origin direction
door_leaf pixels identify open leaf direction
door_arc pixels identify swing/circulation intent
```

### Main Improvement

Doors and windows became more identifiable to a human viewer and to the vectorizer.

The seven-class raster gave much clearer hints for:

```txt
door count
door location
door swing/circulation intention
window location
wall-hosted opening regions
```

### Main Limitation

Even with better semantic classes, direct conversion from pixels/lines to clean CAD remained unstable.

The vectorizer still struggled with:

```txt
wall topology
opening hosting
orthogonal cleanup
door geometry consistency
clean SVG output
```

### Result

Phase 2 improved semantic evidence, but vectorization quality was still not sufficient.

Phase 2 output folders:

```txt
outputs/vectorization/phase2_7class_semantic_vectorization/iteration3_run2_failed
outputs/vectorization/phase2_7class_semantic_vectorization/iteration4_run3_failed
```

## Phase 3 - Seven-Class Segmentation To Point-Based Vectorization

### Input Assumption

Phase 3 kept the seven-class segmented raster from Phase 2.

The change was in vectorization, not in the segmentation target.

### Vectorization Strategy

Phase 3 shifted from line recognition to point recognition.

Instead of mainly extracting line segments from pixels, the vectorizer attempted to detect architectural points and then build a graph:

```txt
segmented raster
-> connected components
-> architectural point recognition
-> axis alignment
-> point connection graph
-> wall/window/door primitives
-> SVG/debug/metrics
```

The point-based approach attempted to identify points such as:

```txt
wall endpoints
wall corners
T-junctions
cross-junctions
window endpoints
door hinge points
door end points
```

Later Phase 3 debugging simplified the wall side by merging wall endpoint/corner/T/cross subtypes into a generic wall point and using red door bboxes as stronger door anchors.

### Main Improvement

Phase 3 made the vectorization process more architectural and graph-based.

It introduced stronger concepts such as:

```txt
explicit point types
orthogonal point alignment
door bbox based scale
door hinge/end point records
wall/window/door graph edges
debug overlays and metrics for rejected evidence
```

### Main Limitation

Point-based vectorization has a high demand for recognition accuracy.

If key points are misplaced or missing, downstream graph construction becomes unstable.

Observed limitations include:

```txt
low accuracy in wall point recognition
fragile wall-door hinge/end inference
floating or poorly hosted openings
strict point validation causing rejected door/window evidence
many heuristic thresholds and special cases
```

### Current Status

Phase 3 is historical. It remains useful for lessons learned, but it has been superseded by Phase 4 Raster-to-Graph inference for wall graph extraction.

Phase 3 output folder:

```txt
outputs/vectorization/phase3_7class_point_vectorization/iteration5_run3
```

## Phase 4 - Pretrained Raster-To-Graph Inference

### Input Assumption

Phase 4 shifts from rule/CV vectorization to pretrained graph prediction. The input is the clean SVG-rendered raster:

```txt
docs/high_quality_architectural/.../<sample>/model_clean.png
```

The current preprocessing variant is:

```txt
crop512_margin20_truepad
crop content bbox
add true 20% white margin around content
scale long edge -> 512 px
center on white 512x512 canvas
normalize with original repo mean/std
```

Original messy rasters such as `F1_scaled.png` are not used for current Phase 4 wall graph inference.

### Method

The current method uses the official Raster-to-Graph checkpoint and adapts inference rather than training:

```txt
model_clean.png
-> crop512_margin20_truepad preprocessing
-> checkpoint0299.pth
-> generous autoregressive inference
-> validity scoring and hard filters
-> mask-and-rerun multistart recovery
-> merge-on-intersection
-> light post-merge filter
-> graph_pred.json / graph_pred.svg / overlays / metrics
```

Optional `masks/wall_graph.json` and `wall_graph_debug.svg` remain useful for QA/reference and future training fallback, but they are not required for the settled inference path.

### Output Organization

Phase 4 outputs should live under:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/<sample>/
```

Each sample folder should contain all related artifacts for that sample, from input PNGs through predicted graph JSON/SVG, overlays, metrics, and component diagnostics. The testing-only `outputs/raster2graph/` folder is retired.

## Phase Comparison

| Phase | Segmentation classes | Vectorization idea | Strength | Main failure mode |
|---:|---|---|---|---|
| 1 | 5 classes | pixels to line segments | simple initial pipeline | doors/windows too ambiguous |
| 2 | 7 classes | richer semantic pixels to vectors | better opening/door hints | direct pixel/line conversion still unstable |
| 3 | 7 classes | point recognition to graph | more architectural and explicit | point recognition accuracy too demanding |
| 4 | `model_clean.png` RGB input | pretrained Raster-to-Graph inference + validation | direct wall graph prediction with useful topology | current limits are preprocessing sensitivity and incomplete generation on hard plans |

## Naming Rule Going Forward

Use the following interpretation for output folders:

```txt
iteration = vectorization method/version
run       = CNN segmentation model generation
phase     = conceptual family of vectorization approach
```

Recommended future folder names should continue to make both method and model clear:

```txt
iteration6_run3
iteration6_run4
iteration7_run3
```

If a new vectorization concept is introduced, document which phase it belongs to or create a new phase section before adding more output folders.

## Task20 Reorganization Notes

task20 moved the output folders from a flat `outputs/vectorization/v008/<name>` layout into the phase folders this document already described conceptually, via `git mv`/content-identical copy (one folder needed a copy+remove instead of `git mv` due to a transient OS file-lock; verified byte-identical and zero-diff before deleting the original). The `v008` folder name was retired since it described a spec version, not a vectorization phase, and was kept implicit in `configs/vectorization_v008.yaml`'s naming instead.

Two non-source-code references were updated so they keep resolving: `configs/vectorization_v008.yaml`'s `output.output_dir` (now points at `phase3_7class_point_vectorization`, the active phase) and the literal example path in `notebooks/run_single_image_run3_vectorization.ipynb`. `scripts/run_vectorization_v008.py` and `notebooks/run_vectorization_v008_run1.ipynb` still default to the old `outputs/vectorization/v008/...` base - left unchanged per task20's "do not change vectorization source code" constraint; see `outputs/vectorization/README.md`'s "Known drift" note. Historical narrative mentioning the old `iteration5_run3` path in `specs/spec_v008_mask_to_vector.md`'s Task14-19 Debugging Notes was also left as-is (accurate at the time it was written) rather than rewritten - the old-to-new mapping above resolves it if needed.
