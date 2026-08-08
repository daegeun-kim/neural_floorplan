# Spec v006: Evaluation Contract

Status: **canonical**. This file is the single quantitative-evaluation contract
for Neural Floorplan. It defines *what* is measured and *how*, before any
implementation. It replaces the earlier four-line segmentation-only sketch.

This specification defines the evaluation. It does not report results. Measured
numbers live in `evaluation/results/<version>/` and are summarized in the
README.

---

## 0. Objective And Claim Hierarchy

```txt
primary   : spatial logic of walls, circulation, doors, and windows
secondary : semantic segmentation quality, including per-class mIoU
not a goal : pixel-perfect tracing of the source floorplan raster
```

A high pixel score **cannot** compensate for broken circulation or incorrectly
hosted openings. A plan whose wall polygons are a few pixels off but whose
rooms, doors, and connectivity are correct is a better result than a plan that
overlaps the source raster closely but leaves a room unreachable or a door
floating off its wall.

Geometric-distance measurements (endpoint distance, width error, angle error,
rasterized overlap) are permitted **only** as labelled secondary diagnostics.
They must never be presented as the project's headline claim.

### 0.1 Three separately reported layers

Each layer is reported on its own. They are never averaged into a single score.

| Layer | Subject | Role |
|---|---|---|
| **L1** | Seven-class SegFormer semantic quality | required secondary evidence |
| **L2** | Raster-to-Graph wall topology quality | primary |
| **L3** | End-to-end Phase 4 circulation and opening logic after hosting, interval trimming, buffering, and SVG/JSON export | primary |

L2 and L3 are the project's claim. L1 is reported because a segmentation model
was trained here and its per-class quality is required evidence, not because
pixel agreement is the goal.

---

## 1. Data And Split Provenance

### 1.1 Source

```txt
dataset : CubiCasa5K
subset  : high_quality_architectural
source  : https://github.com/CubiCasa/CubiCasa5k
local   : docs/high_quality_architectural/<sample_id>/
```

The dataset is obtained separately under its own terms. It is **not**
redistributed in this repository and **not** included in any GitHub Release
asset produced from this repository.

### 1.2 Canonical manifests

```txt
splits/train.json   5964 entries
splits/val.json      745 entries
splits/test.json     747 entries
```

Manifest identity is recorded per run as the SHA-256 of the manifest file bytes.
The values at the time this contract was written are:

```txt
train.json  sha256:b3720724747e1064...
val.json    sha256:54e14be228d21bfb...
test.json   sha256:99ad636ff3f52218...
```

A run whose manifest hashes differ from those recorded in its own
`provenance.json` is a different experiment and must not be compared directly.

`splits/debug_train.json` and `splits/debug_val.json` are development aids and
**must never** contribute to published aggregates. `splits/_overfit_temp.json` is
written at runtime by `src/train_segmentation.py --overfit` and is not tracked.

### 1.3 Entry structure

Each manifest entry is one *rendering* of one CubiCasa floor plan:

```json
{
  "sample_id":  "3046_model_clean_png",
  "image":      "3046/model_clean.png",
  "target":     "3046/masks/semantic_class_map.png",
  "source_svg": "3046/model.svg",
  "input_type": "svg_rendered_clean"
}
```

Two `input_type` values exist:

```txt
svg_rendered_clean   model_clean.png   rendered from model.svg
original_raster      F1_scaled.png     the original CubiCasa scan-style raster
```

The **base plan id** is the leading numeric CubiCasa folder name (`3046`). Both
renderings of a plan share one base plan id and depict the same architecture.

### 1.4 Known leakage in the shipped splits, and the required correction

`src/build_splits.py` shuffles manifest *entries* and slices them into
train/validation/test. It does **not** group by base plan id. Consequently the
two renderings of one floor plan can land in different splits.

Measured on the shipped manifests:

```txt
entry-level overlap between splits          : 0
base-plan-id overlap, train vs test         : 604
base-plan-id overlap, validation vs test    : 69
test entries whose base plan appears in
  train or validation                       : 673 of 747
```

This is group leakage: for 673 of 747 test entries the model saw the same floor
plan's geometry during training under the other rendering. Any metric computed
over the full `splits/test.json` is therefore optimistic and **must not** be
published as the project's headline result.

**Required rule.** The canonical evaluation population is the leakage-free
subset, defined deterministically:

```txt
test_clean = [ e for e in splits/test.json
               if base_plan_id(e) not in base_plan_ids(train)
               and base_plan_id(e) not in base_plan_ids(validation) ]
```

At the recorded manifest hashes this yields:

```txt
test_clean : 74 entries, 38 unique base plans
             37 svg_rendered_clean + 37 original_raster
```

Rules:

1. `test_clean` is the **primary** reported population for every layer.
2. The full `splits/test.json` population may additionally be reported, but only
   when explicitly labelled *leakage-affected* alongside the overlap counts
   above. It is context, never the headline.
3. The Phase 4 pipeline consumes `model_clean.png`, so L2/L3 evaluate the
   `svg_rendered_clean` entries of `test_clean` (37 plans).
4. `test_clean` membership must be recomputed from the manifests at run time and
   its resulting sample list hashed into `provenance.json`. It must not be
   hand-maintained as a separate file that can drift.
5. The existing trained checkpoint is evaluated as-is. Rebuilding splits by
   group would require retraining and is out of scope for this contract; the
   leakage is corrected at *evaluation* time by restricting the population, and
   the limitation is reported rather than hidden.

### 1.5 Reference sources

| Reference | Built from | Used for |
|---|---|---|
| Seven-class semantic label map | `<sample>/masks/semantic_class_map.png` | L1 |
| Wall junction nodes and wall edges | `<sample>/masks/wall_graph.json`, `nodes[type=wall_node]` and `edges` | L2 |
| Door and window openings | `<sample>/masks/wall_graph.json`, `nodes[type=door_center|window_center]` with `host_edge`, `opening_width_px`, `orientation` | L3 |
| Spaces and circulation | derived from the reference wall graph by the partition rule in §4.4 | L3 |

`masks/wall_graph.json` is generated by `src/generate_wall_graphs.py` from
`model.svg` and carries its own `status` field (`ok` / `unusable`) with
`reasons`. This is the authoritative reference-validity signal.

### 1.6 Exclusions

A sample may be excluded only for a declared, machine-recorded reason. Every
exclusion is written to `provenance.json` with its sample id and reason, and
every reported metric states its own denominator.

Permitted exclusion reasons:

```txt
leaked_base_plan            base plan appears in train or validation (§1.4)
missing_input               image file absent locally
missing_semantic_reference  masks/semantic_class_map.png absent
missing_graph_reference     masks/wall_graph.json absent
unusable_graph_reference    wall_graph.json status == "unusable" (carry reasons)
prediction_failed           pipeline raised; carry the exception type
```

Silent dropping is a contract violation. If the number of scorable samples for a
metric differs from the number of scorable samples for another metric, both
denominators are reported separately.

---

## 2. Model And Pipeline Provenance

Every run records the following. Values below are the state at the time this
contract was written and are re-derived, not copied, at run time.

### 2.1 Segmentation

```txt
run              segformer_b0_run3
config           configs/train_segformer_b0_run3.yaml
architecture     frozen SegFormer-B0 encoder (nvidia/mit-b0)
                 + trainable custom FloorplanDecoder
arch_version     v4_seven_class_doors
checkpoint       checkpoints_CNN/segformer_b0_run3/best.pt
                 sha256:f9cd45f84cd186a40d22b0f05b9b3e131c8f29e0dcac79b73dfadb2d1ea729a3
input            512 x 512, dataset normalization per src/dataset.py
seed             42
```

Class map (fixed, seven classes):

```txt
0 background   1 floor   2 wall   3 window
4 door_arc     5 door_leaf        6 door_origin
```

### 2.2 Wall graph

```txt
implementation   external/raster_to_graph/  (vendored upstream, GPL-3.0)
checkpoint       checkpoints_Raster2Graph/checkpoint0299.pth
                 sha256:4fd7a69990da0aaa614ffd7a3c07537f0d4a3f072cd94684e857e23ebb7ccecb
provenance       published by the upstream Raster-to-Graph authors;
                 obtained separately, never redistributed here
citation         Hu et al., Computer Graphics Forum 43(2) e15007, 2024
```

### 2.3 Phase 4 vectorization

```txt
entry point      src.vectorization.phase4.pipeline.run_phase4_pipeline
settled config   specs/spec_v010_phase4_raster2graph_modifications.md §15/§18
preprocessing    crop to content bbox -> true 20% white padding
                 -> long edge 512 px -> centered on white 512x512 canvas
inference        first/later step threshold 0.02, edge search 50 px,
                 monte_times 4, max_candidates_per_step 40,
                 mask-and-rerun multistart (max_new_starts 2)
post-process     merge-on-intersection (snap 6, split 8, collinear 8)
                 + light post-merge filter (angle, dedup, self-loop)
```

### 2.4 Environment

Recorded per run: Python version, `torch`, `torchvision`, `transformers`,
`numpy`, `opencv`, `shapely`, `scikit-image` versions, platform string, device
(`cuda:N` with GPU name, or `cpu`), and the deterministic seed.

Determinism requirement: the seed is set for `random`, `numpy`, and `torch`, and
the same command on the same assets must reproduce identical aggregate values.
Where a component is not bit-reproducible, that component is named explicitly in
`provenance.json` rather than left implicit.

---

## 3. Coordinate Frames

All L2/L3 matching happens in the **512 x 512 preprocessed canvas** frame, which
is the frame the Phase 4 pipeline predicts in.

Reference geometry from `masks/wall_graph.json` is in the original raster frame
(`image_width` x `image_height`). It is mapped into the canvas frame by applying
the same recorded preprocessing transform the prediction used (content-bbox
crop, 20% pad, uniform scale, centering offset). The transform parameters used
for each sample are written into the per-sample record so a reviewer can verify
the mapping rather than trust it.

Tolerances are expressed relative to this canvas so they are scale-aware and
comparable across plans of different physical size.

---

## 4. Required Metrics

Every metric below states its matching rule, tolerance, unit, denominator, and
aggregation. Unless stated otherwise:

- **Matching** is one-to-one, computed by optimal bipartite assignment (Hungarian)
  on the stated cost, restricted to pairs within tolerance.
- **Aggregation** is reported twice: *micro* (pool TP/FP/FN across all samples,
  then compute P/R/F1) and *macro* (compute P/R/F1 per sample, then average over
  scorable samples). Both are reported; neither is presented alone.
- **Denominator** is the count of scorable samples for that specific metric.

Primary tolerance constant:

```txt
TOL_NODE = 8 px on the 512 x 512 canvas   (~1.6% of canvas width)
```

A second, looser value `TOL_NODE_RELAXED = 16 px` is reported alongside it so a
reader can see tolerance sensitivity rather than one hand-picked threshold.

### 4.1 Wall junction / node metrics — primary

**Node precision / recall / F1.**
Cost is Euclidean distance in canvas pixels; a pair is eligible when distance
<= `TOL_NODE`. Predicted wall-graph nodes are matched against reference
`wall_node` entries. Unmatched predictions are false positives; unmatched
references are false negatives. Unit: dimensionless. Reported at both tolerances.

**Degree-aware junction F1.**
As above, but a matched pair counts as a true positive only when the node
*degree* also agrees, where degree is the number of incident wall edges. Two
variants are reported:

```txt
degree_exact     predicted degree == reference degree
degree_within_1  |predicted degree - reference degree| <= 1
```

Degree-aware F1 is the metric that distinguishes a genuinely recovered T-junction
or cross-junction from a coincidentally nearby endpoint. It is reported
prominently, not buried.

### 4.2 Wall edge metrics — primary

**Edge precision / recall / F1 by topological matching.**
A predicted edge matches a reference edge when *both* of its endpoints match the
reference edge's endpoints under the §4.1 node assignment (in either
orientation). This is topological correspondence, not pixel overlap.

**Edge metrics by segment overlap** (reported alongside, not instead).
For collinear-within-tolerance edge pairs, overlap ratio is

```txt
overlap_ratio = length(projection of prediction onto reference)
                / length(reference)
```

A pair counts as matched when the perpendicular offset <= `TOL_NODE` and
`overlap_ratio >= 0.5`. This second view exists because a correct wall that the
node matcher splits differently should not be scored as a total miss. Both
views are published; neither is dropped when the other looks better.

**Wall connected-component error.**

```txt
component_count_error = predicted_component_count - reference_component_count
```

Reported as the signed mean, the mean absolute value, and the fraction of
samples with zero error. Unit: components.

**Invalid / dangling topology rate.** Per sample, over the predicted graph:

```txt
dangling_node_ratio     degree-1 nodes / total nodes
isolated_node_ratio     degree-0 nodes / total nodes
non_orthogonal_ratio    edges beyond +/-10 degrees of an axis / total edges
duplicate_edge_count    exact duplicate endpoint pairs
zero_length_edge_count  edges shorter than 1 px
```

The same five quantities are computed on the reference graph and reported side
by side, so the reader can see whether a high rate reflects a bad prediction or
a genuinely fragmented plan.

### 4.3 Opening metrics — primary

Openings are typed: `door` or `window`. Reference openings come from
`wall_graph.json` `door_center` / `window_center` nodes.

**Door detection F1 and window detection F1** (reported separately, never
pooled). A predicted opening matches a reference opening when:

```txt
type matches exactly
AND center distance <= TOL_OPENING = 12 px on canvas
```

Cost for the assignment is center distance. Type confusion is reported
separately as `mistyped_openings` (matched by position, wrong type) and is
counted as both a false positive for the predicted type and a false negative for
the reference type.

**Opening host correctness.** For a matched opening pair, the host is correct
when the wall edge the prediction is attached to corresponds — under the §4.2
edge assignment — to the wall edge named by the reference's `host_edge`.

```txt
host_accuracy = correctly hosted matched openings / matched openings
```

An opening with no host wall in the prediction is `unhosted` and is counted in
the failure table, never silently ignored.

**Opening gap correctness.** A hosted opening must interrupt its wall exactly
once. For each matched, correctly hosted opening:

```txt
gap_present     the host wall interval is actually interrupted
gap_count       number of interruptions attributable to this opening
gap_width_ratio predicted gap width / reference opening_width_px
```

`gap_correct` requires `gap_present` **and** `gap_count == 1` **and**
`0.7 <= gap_width_ratio <= 1.4`. Duplicate or overlapping gaps
(`gap_count > 1`) are reported as their own failure category, because a doubled
gap is a distinct defect from a missing one.

### 4.4 Space and circulation metrics — primary

**Space derivation.** Spaces are derived identically from the reference graph
and from the predicted graph, by the same code, so the comparison is fair:

```txt
1. rasterize all wall edges onto the 512 x 512 canvas with a fixed
   WALL_STAMP_WIDTH = 3 px, with every opening interval CLOSED
2. label connected components of non-wall pixels (4-connectivity)
3. discard components smaller than MIN_SPACE_AREA = 400 px^2
4. the component touching the canvas border is labelled exterior
```

Closing openings in step 1 is deliberate: it makes rooms separate regions so
that connectivity is decided by openings in step 5, not by wall gaps.

```txt
5. an opening connects the two space labels adjacent to it,
   sampled perpendicular to its host wall at +/- 5 px from its center
```

**Space correspondence.** Reference and predicted spaces are matched one-to-one
by maximum pixel IoU, requiring `IoU >= 0.3`. This overlap is used **only** to
establish identity. The scored outcome is adjacency and reachability logic, not
room boundary precision. Unmatched spaces on either side are reported as
`space_count_error` and excluded from adjacency scoring with their count stated.

**Space-adjacency precision / recall / F1.** Over matched space pairs, an
adjacency edge exists when an opening connects the two spaces. Predicted
adjacency edges are compared with reference adjacency edges under the space
correspondence. Unit: dimensionless. Micro and macro both reported.

**Reachable-space-pair agreement.** Compute the transitive closure of each
adjacency graph. For all unordered pairs of matched spaces:

```txt
reachability_agreement = pairs with identical reachable/unreachable status
                         / total matched space pairs
```

Also reported separately, because they fail differently:

```txt
false_connections    unreachable in reference, reachable in prediction
false_separations    reachable in reference, unreachable in prediction
```

A single spurious opening can merge a whole plan into one reachable blob, so
`false_connections` is reported explicitly rather than averaged away.

### 4.5 Per-sample validity and failure counts — primary

Counted per sample and aggregated as totals plus per-sample means:

```txt
unhosted_openings          predicted opening with no host wall
missing_openings           reference opening with no prediction (per type)
spurious_openings          predicted opening with no reference (per type)
mistyped_openings          matched by position, wrong type
broken_junctions           matched node with degree mismatch
duplicate_gaps             hosted opening producing more than one gap
disconnected_circulation   reference-reachable space pairs that are
                           unreachable in the prediction
```

### 4.6 Segmentation metrics — required secondary

Computed with `src/metrics.py`, extended rather than reimplemented.

```txt
IoU per class          all seven classes, individually named
mean IoU               background INCLUDED; the inclusion rule is stated
                       next to the number wherever it is published
foreground mean IoU    classes 1-6, background excluded
pixel accuracy         all pixels
boundary F1            wall, window, door_arc, door_leaf, door_origin,
                       tolerance 2 px (matches the run3 training config)
```

Empty-class handling: a class absent from both prediction and reference for a
sample yields `NaN` for that sample and is **excluded** from that sample's mean,
never counted as 0.0. Per-class aggregates state, for each class, how many
samples contributed. A class present in the reference but entirely missed by the
prediction yields IoU 0.0 and *is* counted.

Both micro (pooled confusion matrix over all pixels of all samples) and macro
(per-sample IoU averaged) aggregation are reported, because they diverge sharply
for rare classes such as `door_origin`.

### 4.7 Secondary geometric diagnostics

Permitted, and clearly labelled *diagnostic* wherever published:

```txt
node_endpoint_distance_px      mean / median over matched nodes
opening_center_distance_px     mean / median over matched openings
opening_width_error_ratio      predicted / reference opening width
wall_angle_error_deg           deviation from nearest axis
wall_mask_iou                  rasterized predicted walls vs reference walls
```

These must never be presented as the primary success criterion, and no summary
sentence may lead with them.

---

## 5. Baseline Comparison

At least one fair, runnable baseline is required.

### 5.1 Required baseline: raw Raster-to-Graph

```txt
baseline : raw pretrained Raster-to-Graph output, taken BEFORE this project's
           Phase 4 alignment, recovery, merge, and vector reconstruction
current  : the full Phase 4 pipeline
```

Both paths must use:

```txt
the same test_clean population
the same preprocessing
the same reference data
the same metric implementation and denominators
```

Any metric the baseline cannot support (it produces no openings, no spaces, and
no circulation, so §4.3 to §4.5 do not apply to it) is reported as
`unavailable` with a one-line explanation. Coercing an unsupported baseline
metric to `0.0` is prohibited, because it would manufacture an artificial
improvement.

Applicable baseline metrics are therefore the §4.1, §4.2, and §4.7 wall-graph
families.

### 5.2 Optional historical context

An earlier segmentation checkpoint or an earlier vectorization phase may be used
as a quantitative baseline **only** when its code, configuration, checkpoint, and
data provenance are all reproducible today. Otherwise it is described as
qualitative historical context and carries no numbers.

Reconstructing baseline numbers from old output folders is prohibited.

---

## 6. Reporting And Failure Analysis

### 6.1 Output files

One versioned directory, `evaluation/results/<version>/`, containing only
compact, reviewer-useful artifacts:

```txt
provenance.json           §1, §2 provenance; split hashes; checkpoint hashes;
                          software/hardware; seed; exclusions with reasons
per_sample.csv            one row per evaluated sample, all metrics
segmentation.csv          per-class IoU, boundary F1, contributing counts
topology.csv              node/edge/opening/circulation aggregates
baseline_comparison.csv   raw R2G vs Phase 4, aligned metric names
failures.json             failure categories with counts and sample ids
selected_cases.json       deterministic representative cases (§6.3)
summary.json              the aggregate numbers the README consumes
```

Raw predictions, dataset copies, full checkpoints, and large debug dumps are
written outside the committed result set.

### 6.2 Required content

Every published table carries its sample count and denominator. Macro and micro
aggregation are reported wherever both are meaningful. The baseline-versus-current
table is required.

### 6.3 Deterministic case selection

Representative cases are chosen by rule, never by eye:

```txt
success : highest end-to-end score, ties broken by ascending sample_id
mixed   : score closest to the median, ties broken by ascending sample_id
failure : lowest end-to-end score, ties broken by ascending sample_id
```

where the end-to-end score is the unweighted mean of degree-aware junction F1,
topological edge F1, door F1, window F1, host accuracy, and reachability
agreement. The selection rule and the resulting sample ids are both written to
`selected_cases.json`, so the choice is auditable and stable across runs.

### 6.4 Failure categories and limitations

The report must name its failure categories and state its limitations, including
at minimum: the §1.4 leakage restriction and the resulting small `test_clean`
population; plans whose wall topology is out of the pretrained checkpoint's
distribution; ambiguous or missing opening evidence; and any samples excluded as
unscorable.

### 6.5 Prohibited claims

The report must not claim pixel-perfect recovery, production readiness, or
universal accuracy. Measured weaknesses are results to publish, not numbers to
conceal or soften.

---

## 7. Reproduction And Acceptance

### 7.1 Commands

Two modes are required and must be documented with exact flags.

```txt
smoke : fixture-only or tiny sample limit; no dataset, no checkpoints,
        no GPU, no network; runs in ordinary CI
full  : explicit local dataset root, explicit checkpoint paths, explicit
        output directory, explicit device and seed
```

The CLI must fail early and actionably on missing data, missing checkpoints,
malformed manifests, incompatible configs, an unavailable GPU that was
requested, or an unwritable output path. It must never download data or
checkpoints implicitly.

### 7.2 Acceptance

No target score is set before the first standardized run. Setting one would
invite tuning against the test set, which §0 and Task 42 both prohibit.

Acceptance for the first release is that the protocol is:

```txt
reproducible  same command + same local assets -> same numbers
complete      every metric in §4 is produced or explicitly marked unavailable
honest        denominators, exclusions, leakage, and baselines are stated
```

Measured weaknesses remain results to report. The evaluation measures the
current system; it must not modify the model or the vectorizer to improve a
score.
