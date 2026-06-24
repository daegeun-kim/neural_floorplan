# Task 11 - Replace Vectorization From Scratch

## Objective

Replace the current vectorization implementation from scratch.

Do not patch, preserve, or incrementally repair the existing vectorization flow. The current vectorization source is considered failed and should be rebuilt around the new v008 point-graph process.

## Source Of Truth

Use this spec as the implementation source of truth:

```txt
specs/spec_v008_mask_to_vector.md
```

Use `specs/spec_v007_component_primitives.md` only for shared primitive definitions that do not conflict with v008.

## Required Work

Rebuild the vectorization code under:

```txt
src/vectorization
src/vectorization/primitives
configs/vectorization_v008.yaml
tests/test_vectorization_v008.py
```

The new implementation must follow the v008 orthogonal point-graph pipeline:

```txt
decode masks
extract components
resolve scale
search the seven allowed point types
align points orthogonally
connect wall/window/door-origin graph edges
generate door leaf and arc
export SVG
write debug overlay and metrics
```

## Hard Rules

- no retired 5-class assumptions
- no contour-tracing final output
- no floor generation for this restart
- no 45-degree final wall output
- only orthogonal wall/window/door-origin graph edges
- doors exist only when red `door_arc` evidence exists
- door widths snap to `700 mm` or `900 mm`
- window minimum width is `300 mm`
- wall thickness is `100 mm` or `200 mm`
- door origin, door leaf, and door arc are thin symbolic SVG elements
- final `vector.svg` contains only `wall`, `window`, and `door` visible groups

## Acceptance Criteria

This task is complete when the old vectorization path has been replaced and the new v008 tests verify the point-graph behavior described in `spec_v008_mask_to_vector.md`.

