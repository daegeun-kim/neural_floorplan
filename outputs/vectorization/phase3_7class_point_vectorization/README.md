# Phase 3 - Seven-Class Segmentation To Point-Based Vectorization

See `specs/vectorization_phase_history.md` and `specs/spec_v008_mask_to_vector.md` for the full writeup. Summary:

- **Phase goal**: shift from line recognition to architectural point recognition + graph construction.
- **Input segmentation type**: 7-class (same as phase 2 - the change is in vectorization, not the segmentation target).
- **Vectorization idea**: segmented raster -> connected components -> architectural point recognition -> axis alignment -> point connection graph -> wall/window/door primitives -> SVG/debug/metrics. Later debugging (tasks 15-19) merged wall endpoint/corner/T/cross subtypes into one generic wall point and made red door bboxes the trusted door anchor.
- **Known limitation**: point-based vectorization demands high recognition accuracy - misplaced/missing points destabilize downstream graph construction (fragile wall-door hinge/end inference, floating or poorly hosted openings, many heuristic thresholds).

## Iteration folders here

- `iteration5_run3/` - current/active output, no `failed` suffix. This is the required test case for `specs/vectorization_must_rules.md` validation (see `spec_v008_mask_to_vector.md`'s Task14-19 Debugging Notes).
