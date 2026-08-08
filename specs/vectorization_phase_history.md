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

(task20 moved these out of a flat `outputs/vectorization/v008/<name>` layout. The `v008` name described a spec version rather than a vectorization phase, so it was retired and kept only in `configs/vectorization_v008.yaml`'s filename. `outputs/` is generated and is not tracked in Git.)

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

## Intention Changes Across Phases

The durable record of *why* the project changed direction at each step.

| Stage | Segmentation intention | Vectorization intention | Main failure | Resulting change |
|---|---|---|---|---|
| 5-class | predict broad architectural classes | infer geometry from broad masks | openings too generic | split openings into richer classes |
| 7-class | predict vectorization-useful wall/window/door evidence | convert richer masks into vectors | geometry still failed; topology was implicit | redesign vectorization rules |
| early semantic vectorization | keep class-to-object mapping simple | convert masks to walls/floor/openings | output looked contour-like and disconnected | add architectural primitive rules |
| wall/opening refinement | keep 7-class CNN | fix wall connectivity and hosted openings | component heuristics unstable | restart vectorization from graph points |
| point-graph restart | keep 7-class CNN as evidence | detect points, align, connect | point/scale detection brittle | move to Raster-to-Graph wall graph prediction |
| Raster-to-Graph inference | clean SVG-rendered raster as graph-model input | adapt pretrained checkpoint via preprocessing, thresholds, multistart, scoring, merge cleanup | direct checkpoint inference too often empty | settled on generous inference from `model_clean.png`; no fine-tuning needed |

The single most important conclusion:

```txt
The CNN predicts semantic evidence.
Raster-to-Graph produces the wall topology.
The vector/CAD stage attaches classification and openings afterwards.
```

The CNN was never going to solve wall-graph vectorization on its own. Every
phase before Phase 4 failed in the same underlying way — topology was left
implicit and had to be recovered from pixels by heuristics.

## Phase 4 Implementation Lessons

Durable refinements from the Phase 4 graph-to-vector work. The enforced
invariants live in `specs/vectorization_must_rules.md`; this section records the
reasoning behind them.

**Door primitives must stay a three-part contract.** The first graph-to-vector
attempt collapsed the hosted door edge into a single stroke. The correct
decomposition is `door_origin` (purple line along the wall gap), `door_leaf`
(orange perpendicular from the hinge), and `door_arc` (red 90° arc centred on
the hinge, from the origin far point to the leaf endpoint). Phase 3 already had
this right; Phase 4 regressed it by drawing doors locally instead of reusing the
existing primitive behavior.

**Overlapping openings are a placement problem, not an evidence problem.** When
a door and a window trim overlapping intervals on one wall, the first instinct —
reject the lower-confidence opening — was wrong: the 7-class raster usually
detects both correctly. The rule is to keep the stronger opening fixed and
move or shrink the weaker one (door beats window; otherwise higher confidence
wins). Reject an opening only when no feasible non-overlapping interval exists
on the host wall chain, and record both original and adjusted intervals.

**Partial implementation of a pipeline contract reads as "no change".** After the
de-overlap work, output barely moved because only part of the intended process
was implemented. Pipeline stages need explicit contract enforcement rather than
best-effort application.

**Door swing direction comes from red-side evidence.** Generated arc or leaf
sampling is secondary and must not override strong local red/orange raster
evidence. Hosted opening segments must render with flat endpoints
(`stroke-linecap="butt"` or explicit flat-ended geometry) so buffering and stroke
caps cannot lengthen them past their wall trim nodes.

**Some doors are genuinely double-swing.** When red door-arc evidence appears on
both sides of one origin, forcing a single side is wrong. Classification
distinguishes `single_swing`, `double_swing_shared_origin`,
`separate_single_swing_doors`, and `ignored_duplicate`. A merged double-swing
door draws one shared origin with leaves and arcs on both sides, and trims the
wall only once. Nearby doors stay separate unless they share the origin edge.

## Task20 Reorganization Notes

task20 moved the output folders from a flat `outputs/vectorization/v008/<name>` layout into the phase folders this document already described conceptually, via `git mv`/content-identical copy (one folder needed a copy+remove instead of `git mv` due to a transient OS file-lock; verified byte-identical and zero-diff before deleting the original). The `v008` folder name was retired since it described a spec version, not a vectorization phase, and was kept implicit in `configs/vectorization_v008.yaml`'s naming instead.

Two non-source-code references were updated so they keep resolving: `configs/vectorization_v008.yaml`'s `output.output_dir` (now pointing at `phase3_7class_point_vectorization`) and a literal example path in the Phase 3 notebook. `scripts/run_vectorization_v008.py` still defaults to the old `outputs/vectorization/v008/...` base, left unchanged under task20's "do not change vectorization source code" constraint. Historical narrative in `specs/spec_v008_phase3_mask_to_vector.md` still mentions the old `iteration5_run3` path; it was accurate when written and the mapping above resolves it.
