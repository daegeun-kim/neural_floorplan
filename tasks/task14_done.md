# Task 14 - Debug Vectorization Against Must Rules

## Objective

The current vectorization source still has bugs relative to the existing vectorization specs and task requirements.

Do not invent additional vectorization rules for this task. The source of truth is:

```txt
specs/vectorization_must_rules.md
```

Read that file first, then continuously debug the current vectorization source until the generated output satisfies every observable must requirement in that file.

The goal is not to redesign the pipeline. The goal is to make the current vectorization process meet the existing must rules.

## Required Test Case

Use this test image/output folder as the primary debugging target:

```txt
C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\outputs\vectorization\v008\iteration5_run3
```

The image in that folder can be converted to vector output using the current source through:

```txt
C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\notebooks\run_single_image_run3_vectorization.ipynb
```

Use that notebook or the equivalent source calls it invokes to regenerate:

```txt
vector.svg
debug_overlay.png
metrics.json
```

## Required Workflow

1. Read `specs/vectorization_must_rules.md`.
2. Run vectorization on the required test image.
3. Inspect the generated `vector.svg`, `debug_overlay.png`, and `metrics.json`.
4. Compare the observed output against every must rule in `specs/vectorization_must_rules.md`.
5. Identify concrete bugs in the current source.
6. Fix the source under `src/vectorization`.
7. Re-run the same test image.
8. Repeat until all observable must requirements are satisfied.

Do not stop after the first fix if later output still violates the must rules.

## Debugging Scope

Allowed changes:

```txt
src/vectorization
configs/vectorization_v008.yaml
tests
notebooks/run_single_image_run3_vectorization.ipynb
```

Only edit the notebook if it is necessary to run the existing vectorization process correctly.

Do not modify CNN training code.

Do not change the must-rules file unless the user explicitly asks for a rules change.

## Required Output Checks

The debugging must verify at least:

1. Scale is inferred from red `door_arc` bbox long edges.
2. Door widths are `700 mm` or `900 mm`.
3. Wall thicknesses are `100 mm` or `200 mm`.
4. Every accepted red `door_arc` cluster becomes a door object.
5. Door hinge/end points are inferred and shown in the debug overlay.
6. Door hinge/end points are within `200 mm` of the associated red bbox.
7. Final points are only the seven allowed point types.
8. Final wall/window/door-origin edges are orthogonal.
9. No 45-degree or arbitrary-angle wall geometry appears in final output.
10. Windows are hosted on wall topology and do not float.
11. Doors are hosted on wall topology and do not float.
12. Walls are black closed filled polygons.
13. Windows are blue closed filled polygons.
14. Door origin is a thin purple symbolic line.
15. Door leaf is a thin orange symbolic line.
16. Door arc is a thin red 90-degree symbolic arc centered on the hinge.
17. Final SVG contains only the allowed visible groups: `wall`, `window`, and `door`.
18. Final SVG contains no debug, unidentified, or generic `opening` group.
19. Debug overlay shows rejected evidence and low-confidence door inference when applicable.
20. Metrics include red scale diagnostics and one door-candidate record per red cluster.

## Required Tests

Add or update tests so the fixed behavior is protected.

Tests should cover the bugs found while debugging the required test image, especially any failures related to:

```txt
red door_arc clusters not becoming doors
scale not using red bbox long edges
door widths not snapping to 700/900 mm
wall thickness not normalizing to 100/200 mm
floating windows or doors
non-orthogonal final edges
incorrect SVG groups or colors
missing debug/metrics records
```

## Acceptance Criteria

This task is complete when:

1. `specs/vectorization_must_rules.md` has been read and used as the only rule checklist.
2. The required test image has been vectorized with the current source.
3. The generated `vector.svg`, `debug_overlay.png`, and `metrics.json` satisfy all observable must rules.
4. Source bugs discovered during the process are fixed.
5. Tests cover the fixed behavior.
6. The final task report names any must rule that could not be automatically verified, with a clear reason.
