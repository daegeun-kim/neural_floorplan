# Task 01 — Export SVG Floorplans to Raster With Hidden Geometry Made Visible

## Context

This project trains a neural floorplan segmentation model using raster images exported from CubiCasa5K-style SVG floorplan files. Some SVG files contain valid floorplan geometry that is not visible in browser-based SVG rendering or direct SVG-to-raster conversion because parts of the SVG are marked with `display: none` or equivalent hidden styling.

original data sources are located at "C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\docs\original_vector\cubicasa5k\cubicasa5k\high_quality_architectural"


A sample SVG shows that some complete floor groups can be hidden, for example:

<g style="display: none;" class="Floor">
  <g id="Floor-2" class="Floorplan Floor-2">

The same SVG can also contain a composed multi-floor view that references multiple floors but is hidden:

<g class="FloorsCompose" style="display: none;">
  <g class="ComposeElement">
    <use xlink:href="#Floor-1"/>
  </g>
  <g transform="matrix(1,0,0,1,125.535,12.9822)" class="ComposeElement">
    <use xlink:href="#Floor-2"/>
  </g>
</g>

Browsers and rasterizers correctly respect `display: none`, so the exported PNG can miss an entire floor or significant plan geometry. Illustrator may still show the hidden geometry because it imports SVG objects as editable layers, but this is not reliable for automated training-data generation.

## Objective

Create a preprocessing/export pipeline that converts SVG floorplan files into raster images while making all floorplan-relevant geometry visible. The source SVG files should remain unchanged. The script should generate temporary normalized SVG files or in-memory SVG strings, then export those normalized SVGs to PNG.

The generated raster images must include hidden floorplan geometry that would otherwise be omitted by browser-style SVG rendering.

## Requirements

1. Read project configuration from `spec.md`.
2. Preserve original SVG files.
3. Make hidden floorplan geometry visible.
4. Use architectural class filtering.
5. Prefer `FloorsCompose` when it exists.
6. Handle `<use xlink:href="#...">` references correctly.
7. Maintain original visual style unless visibility must be fixed.
8. Export PNGs using the project’s raster settings.
9. Validate output by checking non-white pixels.
10. Batch-process the full SVG directory.

## Acceptance Criteria

This task is complete when:

- SVG files with hidden `Floor-2` or hidden `FloorsCompose` geometry export to PNG with those floorplan lines visible.
- Original SVG files are not modified.
- Raster output is suitable for training and no longer misses hidden floorplan regions.
- UI/control artifacts such as resize handles, selection controls, and hidden dimension marks are not unintentionally rendered.
- The script can batch-process the dataset.
- The script logs visibility fixes and warns on suspicious blank outputs.