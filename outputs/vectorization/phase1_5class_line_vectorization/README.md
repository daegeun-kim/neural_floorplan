# Phase 1 - Five-Class Segmentation To Line Segments

See `specs/vectorization_phase_history.md` for the full writeup. Summary:

- **Phase goal**: convert a segmented raster directly into line-segment/architectural-primitive vector geometry.
- **Input segmentation type**: 5-class (broad classes; doors/windows not separately distinguished).
- **Vectorization idea**: segmented pixels -> extract region/edge/line-like evidence -> convert into wall/opening line segments -> export SVG-like vector geometry.
- **Known limitation**: door and window evidence were too generic and ambiguous; opening type and door swing geometry weren't available as strong evidence, so accuracy was extremely low.

## Iteration folders here

- `iteration1_run1_failed/`
- `iteration2_run1_failed/`
- `iteration2_run2_failed/`
