# Task 19 - Debug Overlay Cleanup: Drop Rejected Evidence, Fix Hinge/End Colors

## Objective

From debug overlay images, remove rejected/unresolved evidence - they do not need to be shown. Make sure `wall_door_hinge_point`/`wall_door_end_point` are shown in orange and purple (their intended `POINT_COLORS`), instead of red circles.

## Implementation

- `debug.py` `build_debug_overlay`: removed the gray-rectangle rendering loop over `rejected_evidence`; removed the matching legend row and the now-unused `REJECTED_COLOR` constant. `rejected_evidence` stays in `metrics.json` (`build_metrics`) unchanged - this is purely about what the overlay image draws.
- `debug.py` door-candidate loop: stopped re-drawing a second, confidence-colored (red/yellow) circle around each hinge/end point. The bbox rectangle and hinge-to-end connector line still use the confidence color; the points themselves are only drawn once, by the existing per-point-type loop, in their own `POINT_COLORS` (orange hinge, purple end).
- `specs/vectorization_must_rules.md`: rule 107 updated (rejected evidence is metrics-only now), old rule 111 ("must show rejected evidence") removed, rule 113 (now 112) clarified to require hinge/end in their own point-type colors; rules renumbered 107-123 to stay sequential.
- `specs/spec_v008_mask_to_vector.md`: ## 15 updated; new "Task19 Debugging Notes" section.
- `tests/test_vectorization_v008.py`: `test_debug_overlay_includes_rejected_evidence` replaced with `test_debug_overlay_does_not_render_rejected_evidence`, which asserts rejected evidence has zero effect on the rendered pixels.

## Verification

- Full test suite: 93 passed.
- Regenerated all three required samples (`outputs/vectorization/v008/iteration5_run3`) and visually inspected `debug_overlay.png`: no gray rejected-evidence boxes remain, hinge/end markers read as small orange/purple circles across all 17 doors. `metrics.json` content/validation issues unchanged.
