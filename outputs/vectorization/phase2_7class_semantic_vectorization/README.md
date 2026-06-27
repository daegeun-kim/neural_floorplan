# Phase 2 - Seven-Class Segmentation With Door And Window Evidence

See `specs/vectorization_phase_history.md` for the full writeup. Summary:

- **Phase goal**: keep converting raster evidence into vector geometry, but with much stronger semantic hints for openings.
- **Input segmentation type**: 7-class (`wall`, `window`, `door_arc`, `door_leaf`, `door_origin`, `floor`, `background`).
- **Vectorization idea**: window pixels identify hosted windows; `door_origin` identifies threshold/origin direction; `door_leaf` identifies open-leaf direction; `door_arc` identifies swing/circulation intent - direct pixel/line conversion, just with richer evidence.
- **Known limitation**: door/window identifiability improved a lot, but direct pixel-to-vector conversion remained unstable for wall topology, opening hosting, orthogonal cleanup, and clean SVG output.

## Iteration folders here

- `iteration3_run2_failed/`
- `iteration4_run3_failed/`
