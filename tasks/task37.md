# Task 37 - Portfolio Website Neural Floorplan Page Update

## Target Repository

Update the portfolio website repository:

```txt
C:\Users\kdgki\Desktop\PI\DaegeunKim_website
```

Primary page to update:

```txt
C:\Users\kdgki\Desktop\PI\DaegeunKim_website\neural_floorplan.html
```

The website is pushed from:

```txt
https://github.com/daegeun-kim/DaegeunKim_website
```

## Required Reading Before Editing

Before modifying the website, read and obey:

```txt
C:\Users\kdgki\Desktop\PI\DaegeunKim_website\CLAUDE.md
```

Then read the current neural floorplan project docs from:

```txt
C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan
```

Use these files as the main project source of truth:

```txt
readme.md
workflow.md
specs/vectorization_phase_history.md
specs/spec_v008_phase4_vectorization.md
specs/spec_v010_phase4_raster2graph_modifications.md
specs/portfolio_flowchart_outline.md
```

Do not use the individual task files as the main website source. They are too
implementation-specific for the portfolio page.

## Page Goal

Rewrite the existing `neural_floorplan.html` page from an early Phase 1
segmentation page into a current Phase 4 research/technical portfolio page.

The page should communicate the project as:

```txt
raster floorplan
-> semantic understanding
-> wall graph prediction
-> opening/component attachment
-> editable CAD-like SVG / JSON vector output
```

The page should be understandable to:

```txt
ML / computer vision readers
AEC technology readers
computational designers
geometry processing engineers
AI/BIM modelers
software engineering reviewers
general portfolio visitors
```

The emphasis should be strongest on the research and technical pipeline, while
still being legible to non-specialist readers.

## Overall Tone

Use a research-documentation / technical case-study tone.

Do not present the project as a polished commercial product. It is a working
research pipeline whose vector outputs are useful but still imperfect.

It is acceptable and important to explain why earlier phases failed and how
those failures motivated Phase 4.

Keep limitations brief and professional:

```txt
current vector outputs are workable but not perfect
some samples still contain wrong spatial logic or component placement errors
future work includes fine-tuning and improving the pipeline
```

Do not over-emphasize future work or make the page feel unfinished.

## Main Claim

The central project claim should be:

```txt
The project converts raster floorplans into editable architectural vector
components by combining semantic segmentation with graph-based wall extraction.
```

Also make clear that the current Phase 4 pipeline can generate:

```txt
final_vector.svg
final_vector.json
walls
windows
doors
door arcs/leaves/origins
component metadata
```

## Existing Page Content

The current page was written during the early Phase 1 stage and focuses on
up-to-5-class image segmentation.

Keep existing content only when it is still valid, especially:

```txt
CubiCasa5K input data
segmentation data generation
data augmentation
SegFormer training
custom decoder/head explanation if still accurate
valid existing images and captions
```

Remove or rewrite content that implies the project stops at 5-class segmentation.

Make clear that the active segmentation output is now 7 classes:

```txt
background
floor
wall
window
door_arc
door_leaf
door_origin
```

If old image files are no longer referenced after the rewrite, delete the unused
old image files according to the website repository's own `CLAUDE.md` rules.

## Suggested Page Structure

Use the website's existing style and section conventions. The exact section
division can be determined while editing, but the content should roughly cover:

1. Overview
2. Current Phase 4 Pipeline
3. Two-Model Strategy: SegFormer + Raster-to-Graph
4. Vectorization: graph alignment, opening hosting, wall trimming, wall buffering
5. Phase 4 sample process grids
6. Development phases and lessons learned
7. Current limitations
8. References / citations

Do not add glossary-style explanations unless needed for clarity.

## Phase 4 Pipeline Content

Phase 4 should receive the most space on the page.

Explain the full current pipeline clearly:

```txt
CubiCasa5K input floorplan
-> shared preprocessing
-> SegFormer 7-class segmentation
-> Raster-to-Graph wall graph prediction
-> orthogonal graph alignment
-> scale inference from door evidence
-> door/window localization from segmentation
-> snap openings to wall graph
-> trim wall graph at openings
-> buffer connected wall chains into wall polygons
-> export final SVG / JSON
```

Important conceptual explanation:

```txt
SegFormer provides semantic evidence:
  where doors/windows are, what type of opening they are, and door swing hints.

Raster-to-Graph provides topology:
  wall endpoints, junctions, and wall edges that are more stable than direct
  pixel-to-line conversion.

Phase 4 combines them:
  wall graph from Raster-to-Graph + opening evidence from 7-class segmentation.
```

For the pipeline diagram section, include a large placeholder for a manually
created Illustrator/SVG diagram. Do not create the actual diagram.

Placeholder label:

```txt
phase4_full_pipeline_flowchart.svg
```

Do not include internal spec names such as `spec_v008` inside visible pipeline
boxes.

## Technical Notes

Include only the most important numeric or technical details, and style them as
small secondary notes rather than main body text.

Use only details confirmed from the project docs/source. Avoid uncertain values.

Appropriate technical notes include:

```txt
Raster-to-Graph preprocessing:
  crop to content bbox
  add true 20% white margin
  scale long edge to 512 px
  center on white 512x512 canvas

Raster-to-Graph inference modifications:
  first/later step thresholds lowered to 0.02
  edge search threshold 50 px
  monte_times 4
  max_candidates_per_step 40
  max_new_starts 2
  keep edges within +/-10 degrees of horizontal/vertical

Vector wall generation:
  snap/alignment to orthogonal graph
  trim walls at hosted doors/windows
  connect wall graph before buffering
  target wall thickness 200mm when scale inference is available
```

Keep these notes concise and visually secondary.

## Development Phases Section

Include one larger section titled something like "Development Phases" or
"Research Iteration".

Inside it, include short subsections for Phase 1, Phase 2, Phase 3, and Phase 4.

Phase 4 should still dominate the page overall; Phases 1-3 should be short and
used to explain why the method evolved.

### Phase 1 - 5-Class Segmentation To Line/Component Vectorization

Explain:

```txt
The first attempt tried to convert 5-class segmentation edges directly into
line segments and architectural components.
```

Main failure:

```txt
doors/windows were too ambiguous
opening and circulation evidence was not explicit enough
vector accuracy was extremely low
```

Include placeholder(s) for failed Phase 1 output images.

### Phase 2 - 7-Class Segmentation Refinement

Explain:

```txt
The semantic classes were redesigned to separate wall, window, door_arc,
door_leaf, and door_origin evidence.
```

Main improvement:

```txt
the raster prediction became much more informative for doors, windows, and
circulation intent
```

Main failure:

```txt
direct semantic-pixel-to-vector conversion still did not reliably recover clean
wall topology
```

Include placeholder(s) or retained valid images.

### Phase 3 - Point-Based Primitive Vectorization

Explain:

```txt
The pipeline tried to recognize architectural points and construct walls,
windows, and door primitives from the 7-class raster.
```

Main improvement:

```txt
more architectural logic: wall points, opening points, axis alignment, door
primitive generation, debug metrics
```

Main failure:

```txt
hand-written point detection was too brittle; missing or wrong points caused
spatial logic to collapse
```

Include placeholder(s) for Phase 3 output/debug images.

### Phase 4 - Hybrid Graph + Semantic Vectorization

Explain:

```txt
Phase 4 uses Raster-to-Graph to predict wall topology directly, then uses the
7-class segmentation output to locate and classify doors/windows before final
vector export.
```

Make clear that this is the current main project status.

## Phase 4 Sample Grids

Create three separate sample grids:

```txt
Sample 1316  - simple plan, best/cleanest current result
Sample 10026 - more complex plan, useful for showing remaining errors
Sample 10029 - more complex plan, useful for showing remaining errors
```

Each sample should have an 8-item process grid laid out as 2 rows x 4 columns on
desktop and responsive on smaller screens.

Use grey placeholder boxes with clear labels and suggested filenames. The user
will manually add the actual images later.

Do not copy images from the neural_floorplan repo into the website repo in this
task. Only create placeholders.

Use this placeholder naming convention:

```txt
sample_1316_01_original_or_image.png
sample_1316_02_preprocessed_input.png
sample_1316_03_7class_segmentation.png
sample_1316_04_graph_pred.svg
sample_1316_05_graph_overlay.png
sample_1316_06_graph_overlay_orthogonal.png
sample_1316_07_debug_overlay.png
sample_1316_08_final_vector.svg

sample_10026_01_original_or_image.png
sample_10026_02_preprocessed_input.png
sample_10026_03_7class_segmentation.png
sample_10026_04_graph_pred.svg
sample_10026_05_graph_overlay.png
sample_10026_06_graph_overlay_orthogonal.png
sample_10026_07_debug_overlay.png
sample_10026_08_final_vector.svg

sample_10029_01_original_or_image.png
sample_10029_02_preprocessed_input.png
sample_10029_03_7class_segmentation.png
sample_10029_04_graph_pred.svg
sample_10029_05_graph_overlay.png
sample_10029_06_graph_overlay_orthogonal.png
sample_10029_07_debug_overlay.png
sample_10029_08_final_vector.svg
```

Suggested captions should explain what each image can show, but do not imply
the exact final image is already present.

For example:

```txt
Original / source floorplan
Preprocessed 512px input
7-class semantic prediction
Raster-to-Graph wall graph
Graph overlay on input
Orthogonally aligned graph
Door/window hosting debug overlay
Final vector SVG
```

## Image Placeholder Behavior

Use grey labelled placeholders where final images are not yet present.

Each placeholder should make the intended filename obvious so the user can place
the asset manually later.

If a currently used old image is still valid for the segmentation/data/training
sections, keep it.

If an old image is no longer used by the updated page, remove its reference and
delete the image file from the website repo if permitted by `CLAUDE.md`.

## Citations And License Awareness

Add proper citations with inline links / footnotes for only the major external
research/software sources:

```txt
CubiCasa5K
SegFormer
Raster-to-Graph
```

Before writing citations, verify the official project/paper/repository URLs and
license requirements. Cite them in a way that satisfies their stated license or
attribution guidance.

Do not over-cite general tools such as PyTorch, OpenCV, Shapely, or browser/SVG
libraries unless the existing page already does so.

## Content To Avoid

Do not:

```txt
make Phase 4 sound fully solved or production-ready
hide the failed earlier attempts
turn the page into a generic glossary
overload the page with implementation minutiae
put spec_vxxx labels in visible pipeline boxes
copy Phase 4 output images yet
use task md files as the main content source
```

## Verification

After editing, verify:

```txt
neural_floorplan.html renders without broken layout
old unused images are removed if permitted
placeholders are visually clear and labelled
the page remains responsive
citations have working links
the page content matches the current Phase 4 project status
```

Follow the website repository's `CLAUDE.md` for any exact preview, validation,
asset, or styling rules.

## Acceptance Criteria

- `neural_floorplan.html` is updated from early Phase 1 segmentation content to
  the current Phase 4 hybrid vectorization project.
- The page emphasizes the research/technical pipeline while staying readable for
  architecture/AEC and general portfolio readers.
- Phase 4 receives the most page space.
- Phases 1, 2, and 3 are briefly explained as research iterations with their
  limitations and lessons.
- The two-model strategy, SegFormer + Raster-to-Graph, is clearly explained.
- The Phase 4 flowchart placeholder is included.
- Three 8-image placeholder grids are included for samples 1316, 10026, and
  10029.
- Valid old segmentation/data/training content is preserved where appropriate.
- Outdated 5-class-only framing is removed.
- Citations are included for CubiCasa5K, SegFormer, and Raster-to-Graph.
- Current limitations are mentioned briefly and honestly.
