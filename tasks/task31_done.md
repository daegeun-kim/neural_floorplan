# Task 31 - Phase 4 Graph-To-Vector Implementation And Notebook Generation

## Objective

Implement the Phase 4 graph-to-vector pipeline described in:

```txt
specs/spec_v008_phase4_vectorization.md
```

This task starts from the settled Phase 4 Raster-to-Graph wall graph output and adds the final vectorization stage:

```txt
preprocessed input image
-> Raster-to-Graph wall centerline graph
-> 7-class segmentation on the same preprocessed image
-> orthogonal graph alignment
-> scale inference
-> door/window graph hosting
-> wall interval trimming
-> connected wall-chain buffering
-> final_vector.svg / final_vector.json
```

## Important Scope Rule

Do not generate `notebooks/phase4_vectorization.ipynb` first.

The notebook must be generated only after the Phase 4 vectorization source modules/classes/functions are implemented under `src/` and can be imported.

The notebook is an integration/demo artifact, not the place where the main algorithm should live.

## Required Source Implementation

Add the Phase 4 graph-to-vector implementation under `src/`.

Recommended module organization:

```txt
src/vectorization/phase4/
  __init__.py
  preprocessing.py
  graph_alignment.py
  graph_geometry.py
  segmentation_inference.py
  scale_inference.py
  opening_detection.py
  opening_hosting.py
  wall_interval_editing.py
  wall_buffering.py
  export_json.py
  export_svg.py
  debug_overlay.py
  pipeline.py
```

If a different module layout is clearly better, keep it similarly explicit and documented.

Do not bury core logic inside a notebook.

## Required Pipeline Behavior

The implementation must follow `spec_v008_phase4_vectorization.md`.

Required behavior:

```txt
1. use one shared preprocessing result for both Raster-to-Graph and segmentation
2. load/use Raster-to-Graph wall graph as the wall topology source of truth
3. run 7-class segmentation on the same preprocessed 512x512 image
4. orthogonally align the graph by shared axis/graph logic
5. infer scale primarily from red door_arc component bbox long edges
6. infer door/window endpoint candidates from semantic components
7. host each door/window endpoint pair onto the same wall edge or same wall chain interval
8. reject or debug-flag openings whose two endpoints cannot host on the same edge/chain
9. insert opening endpoints as graph nodes
10. trim the wall interval between each accepted opening pair
11. connect remaining wall centerlines into graph chains before buffering
12. buffer the connected line system into final 200mm walls when scale is resolved
13. export final SVG/JSON and debug artifacts
```

## Non-Negotiable Geometry Rules

### R2G Graph Is The Wall Source Of Truth

The final wall topology must come from the Raster-to-Graph graph.

Black wall pixels from the segmentation output may be used as evidence/confidence only. They must not replace the R2G graph as the main wall topology source.

### Door Point Hosting

Door points should be inferred from red `door_arc` bbox candidates and proximity to the aligned R2G graph.

Do not repeat the Phase 3 method where door points are primarily inferred from black wall pixels.

The two points of one door must host onto:

```txt
the same wall edge
or
the same wall chain interval
```

They must not independently snap to two unrelated edges.

### Window Point Hosting

Window endpoints should be inferred from blue `window` components and proximity to the aligned R2G graph.

Blue/black contact can be evidence, but it must not be the only source of final endpoint placement.

The two endpoints of one window must host onto:

```txt
the same wall edge
or
the same wall chain interval
```

They must not independently snap to two unrelated edges.

### Opening Trimming Before Buffering

All accepted door/window openings must be inserted and trimmed in centerline graph space before wall polygon generation.

Do not buffer the walls first and then try to cut holes out of buffered wall polygons.

### Connected Wall Chains Before Buffering

The remaining wall edges must be connected as graph chains before buffering.

This is required so wall corners and junctions render cleanly.

Final wall thickness:

```txt
200 mm total
100 mm buffer on each side of the centerline
```

## Required Output Files

For each single-image run, write:

```txt
input.png
image_segmentation.png
image_debug_overlay.png
graph_pred.svg
graph_pred.json
graph_overlay.png
graph_overlay_aligned.png
final_vector.svg
final_vector.json
```

Use exactly `graph_overlay_aligned.png` for the orthogonally aligned graph overlay. Do not reuse `graph_overlay.png` for two different states.

## Final JSON Requirements

`final_vector.json` must include at minimum:

```json
{
  "coordinate_space": "preprocessed_512",
  "preprocessing": {},
  "scale": {
    "status": "resolved|estimated|unknown",
    "px_to_mm": null,
    "evidence": []
  },
  "wall_graph": {
    "raw": {},
    "aligned": {},
    "trimmed": {}
  },
  "openings": {
    "doors": [],
    "windows": [],
    "rejected": []
  },
  "geometry": {
    "walls": [],
    "doors": [],
    "windows": []
  },
  "metrics": {}
}
```

Each accepted door/window must record:

```txt
source_component_id
host_edge_id or host_chain_id
raw_points
snapped_points
width_px
width_mm
confidence
```

Each rejected opening must record:

```txt
source_component_id
opening_type
raw_points if available
rejection_reason
debug_confidence
```

## Notebook Generation Requirement

After the source implementation is complete and importable, generate:

```txt
notebooks/phase4_vectorization.ipynb
```

The notebook must replace `notebooks/phase4_raster2graph.ipynb` as the main Phase 4 demo notebook.

The notebook must import the implemented source functions/classes from `src/`.

The notebook must not duplicate the main algorithm inline.

The notebook must:

```txt
1. accept a single image path
2. run the implemented Phase 4 pipeline
3. write the required output files
4. print the output directory and basic metrics
5. be clean enough for interactive use
```

## Validation

Add focused tests for:

```txt
orthogonal graph alignment
same-edge door hosting
same-edge window hosting
rejection when opening endpoints would host to different disconnected edges
wall interval trimming before buffering
connected wall-chain buffering
scale inference from red door_arc bbox long edges
final JSON schema shape
```

Run the relevant test suite after implementation.

If GPU-dependent R2G or SegFormer inference cannot run in the test environment, unit-test the pure geometry stages with synthetic graphs and masks.

## Completion Criteria

This task is complete when:

```txt
1. Phase 4 graph-to-vector source modules are implemented under src/
2. opening endpoints are hosted only on the same edge/chain
3. wall intervals are trimmed before buffering
4. connected wall chains are buffered into final wall geometry
5. final_vector.svg and final_vector.json are generated
6. notebooks/phase4_vectorization.ipynb is generated after source implementation
7. the notebook imports source modules instead of holding the algorithm inline
8. focused tests pass or any environment blocker is clearly documented
```

After completion, rename this file from:

```txt
tasks/task31.md
```

to:

```txt
tasks/task31_done.md
```
