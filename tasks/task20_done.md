# Task 20 - Organize Vectorization Outputs By Phase

## Objective

Organize the existing vectorization output files and folders according to the three vectorization phases documented in:

```txt
specs/vectorization_phase_history.md
```

This is an organization task only.

Do not change vectorization source code.

Do not change model training code.

Do not alter the contents of generated output files unless a filename or folder path must be updated for organization.

## Source Of Truth

Use this spec as the source of truth for phase definitions and folder mapping:

```txt
specs/vectorization_phase_history.md
```

The three phases are:

```txt
phase1 = 5-class segmentation -> pixel/line-segment vectorization
phase2 = 7-class segmentation -> richer semantic pixel/vector conversion
phase3 = 7-class segmentation -> point-based vectorization
```

Folder naming meaning:

```txt
iterationN = distinct vectorization method
runN       = distinct CNN model generation
failed     = output did not meet required architectural/vectorization quality
```

## Current Output Folders

The current folders under:

```txt
outputs/vectorization/v008
```

must be organized according to this mapping:

| Current folder | Phase |
|---|---:|
| `iteration1_run1_failed` | phase 1 |
| `iteration2_run1_failed` | phase 1 |
| `iteration2_run2_failed` | phase 1 |
| `iteration3_run2_failed` | phase 2 |
| `iteration4_run3_failed` | phase 2 |
| `iteration5_run3` | phase 3 |

## Required Folder Structure

Create or update the output organization so it clearly separates phases.

Recommended structure:

```txt
outputs/vectorization/
  phase1_5class_line_vectorization/
    iteration1_run1_failed/
    iteration2_run1_failed/
    iteration2_run2_failed/

  phase2_7class_semantic_vectorization/
    iteration3_run2_failed/
    iteration4_run3_failed/

  phase3_7class_point_vectorization/
    iteration5_run3/
```

If preserving the existing `v008` folder is preferred, use:

```txt
outputs/vectorization/v008/
  phase1_5class_line_vectorization/
  phase2_7class_semantic_vectorization/
  phase3_7class_point_vectorization/
```

Choose one layout and keep it consistent.

## Required Documentation

Add a short README file in the organized output root explaining:

```txt
what phase1 means
what phase2 means
what phase3 means
what iteration means
what run means
why failed appears in some folder names
```

The README should point back to:

```txt
specs/vectorization_phase_history.md
```

Each phase folder should also include a small `README.md` or similar note describing:

```txt
phase goal
input segmentation type
vectorization idea
known limitation
which iteration folders belong there
```

## File And Folder Naming Rules

Keep folder names explicit and stable.

Use lowercase with underscores where possible:

```txt
phase1_5class_line_vectorization
phase2_7class_semantic_vectorization
phase3_7class_point_vectorization
```

Do not rename individual sample files inside each iteration folder unless there is a clear duplicate or ambiguity.

Preserve generated artifacts such as:

```txt
input.png
prediction.png
vector.svg
metrics.json
debug_overlay.png
```

If any iteration folder contains nonstandard artifact names, document them rather than silently deleting or overwriting them.

## Safety Rules

Before moving any folders, inspect:

```txt
git status
```

Do not delete output folders.

Prefer moving folders into the phase structure over copying, unless copying is safer because external references still rely on the old paths.

If moving folders would break notebooks, specs, or documented paths, either:

```txt
update those references
```

or:

```txt
leave a README note explaining the old-to-new mapping
```

Do not run vectorization as part of this task unless needed only to verify the folder contents.

## Required Verification

After organization, verify:

1. Every listed iteration folder exists in exactly one phase folder.
2. No listed iteration folder was deleted.
3. Phase folder names clearly communicate the vectorization approach.
4. README documentation exists at the organized output root.
5. Each phase has a short explanatory note.
6. `specs/vectorization_phase_history.md` remains consistent with the final folder organization.
7. `git status` clearly shows only organization/documentation changes.

## Acceptance Criteria

This task is complete when:

1. Existing vectorization outputs are organized into phase-specific folders.
2. Folder names reflect the three vectorization phases.
3. Iteration/run naming is preserved inside each phase.
4. Documentation explains the organization.
5. No source-code behavior has been changed.
6. No generated output artifacts have been deleted.
