# Task 18 - Door Bbox Aspect-Ratio Acceptance Floor

## Objective

Add one more condition for door bbox acceptance: only bboxes with aspect ratio smaller than 2:1 are accepted (all the way from 2:1 to a square are accepted; a rectangle with a larger difference between width and height, e.g. 1:3, is rejected).

## Implementation

- `point_detection.py` `_detect_door_points`: before vertex selection runs at all, compute `long_side / short_side` for the red `door_arc` bbox; reject (`RejectedEvidence(kind="unresolved_door_arc_aspect_ratio")`) when it exceeds `max_door_bbox_aspect_ratio` (default `2.0`, boundary inclusive). This is a shape floor on the red cluster itself - the same class of check as the existing minimum-component-area floor, not an evidence-based rejection.
- `configs/vectorization_v008.yaml` / `run_mask_to_vector.py`: new `doors.max_bbox_aspect_ratio` config key (default `2.0`), wired into `detect_cfg["max_door_bbox_aspect_ratio"]`.
- `specs/vectorization_must_rules.md`: rule 51 extended, new rule 53.
- `specs/spec_v008_mask_to_vector.md`: ## 9.3 updated, new validation item 53, new "Task18 Debugging Notes" section.
- `tests/test_vectorization_v008.py`: widened several synthetic door-arc fixtures that were deliberately elongated (~3.5:1, a stand-in shape never meant to model the new floor) to a realistic, roughly-square shape; one test using extreme elongation to represent a "weak" cluster was changed to represent weakness via smaller area instead.

## Verification

- Full test suite: 93 passed.
- Real samples (`outputs/vectorization/v008/iteration5_run3`): all 17 accepted red `door_arc` bboxes across sample_003/004/005 already fall between `1.05:1` and `2.00:1` - this floor changed no observable output for the required samples (a guard against future/other input, not a fix for a current failure). `metrics.json` validation issues unchanged: sample_003/004 zero issues, sample_005 retains only the pre-existing, unrelated `floating_window_point` limitation already documented in task15/16/17.
