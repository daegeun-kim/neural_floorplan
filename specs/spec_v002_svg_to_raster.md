Goal:
Convert CubiCasa SVG annotations into raster masks.

Vector data: Located in "C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\docs\high_quality_architectural" as svg format.
Job:
- for all svg files in the folder, convert to a png file with white background
- name of the converted png files are model_clean.png
- Ignore other png files in the folders
- place the png file in the same location as each of the svg file

---

Task 01 — Export SVG Floorplans to Raster With Hidden Geometry Made Visible

Problem:
Some CubiCasa SVG files contain multi-floor plans where one or more <g class="Floor">
wrapper groups are marked with display:none.  cairosvg respects display:none, so the
exported model_clean.png was missing entire floor's worth of geometry.  Meanwhile the
semantic mask generator (generate_semantic_masks.py) already collects all Floorplan
containers regardless of their parent display state, causing raster/mask misalignment.

Solution implemented in src/svg_to_raster.py:
- normalize_svg_visibility(svg_bytes) parses the SVG with lxml and strips display:none
  from direct children of the Model group that carry class="Floor".
- Only the Floor wrapper groups are targeted; all other hidden elements (dimension marks,
  selection controls, FloorsCompose) are left untouched.
- FloorsCompose is intentionally kept hidden: its <use> transforms apply coordinate
  offsets that would misalign the raster from the masks, and cairosvg cannot render
  display:none elements referenced by <use> anyway.
- convert_svg_to_png() normalizes in-memory before passing bytes to cairosvg.
- process_dataset() logs each visibility fix and warns when a PNG has fewer than 100
  non-white pixels (suspicious blank output).

Dataset impact:
- 3732 total SVG files in high_quality_architectural subset
- ~753 SVGs had at least one hidden Floor wrapper group (Floor-2 missing from raster)
- All 753 now export both floors at their natural SVG coordinates

Acceptance criteria satisfied:
- Hidden Floor-2 geometry is visible in exported PNG
- Original SVG files are not modified (normalization is in-memory)
- UI/control artifacts remain hidden (targeted only Floor wrappers)
- Output validated by non-white pixel count; blank outputs trigger a warning
- Batch-processing via process_dataset() unchanged in interface
- 22 tests pass (11 new, 11 existing) covering normalize, strip, pixel count, and dataset processing