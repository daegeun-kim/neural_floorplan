# Agent Instructions — Neural Floorplan

This is the single source of agent instructions for this repository. There is no
other agent-instruction file.

## Project

Convert raster floor plans and SVG-derived rasters into classified, CAD-like
vector geometry.

```txt
primary    : spatial logic - wall topology, circulation, doors and windows
secondary  : semantic segmentation quality, including per-class mIoU
not a goal : pixel-perfect tracing of the source raster
```

A result that is a few pixels off but architecturally correct beats one that
matches the raster closely while leaving a room unreachable or a door unhosted.

## Pipeline

```txt
model_clean.png
  -> preprocessing (crop to content, 20% white pad, 512x512 canvas)
  -> SegFormer-B0 seven-class segmentation      (semantic evidence)
  -> pretrained Raster-to-Graph inference       (wall topology)
  -> Phase 4: graph alignment, opening hosting,
     wall interval trimming, wall chain buffering
  -> final_vector.svg / final_vector.json
```

## Safety and scope

- Work only inside this repository. Do not read, modify, or access parent
  directories or unrelated folders.
- **Never** open, read, print, or modify `.env` or any credential-bearing file.
  This project needs no credentials for its offline workflow.
- **Never** run Git commands or perform any GitHub-side action — no branches,
  commits, pushes, history rewrites, releases, or API calls. The repository
  owner handles all Git and GitHub operations manually, including uploading
  release assets.
- Before running a shell command: show it, explain why it is needed, and wait
  for approval.
- Do not add a framework, tool, or dataset that the current work does not
  require.

## Environment

```bash
conda activate floorplan-cad          # Python 3.11
python -m pip install -e ".[dev]"
```

All project configuration lives in one `pyproject.toml`. Do not add a second,
overlapping configuration system.

## Quality gates

Run all three after any code change. They are exactly what CI runs:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

The default test suite must stay asset-independent — no dataset, no checkpoint,
no GPU, no network. Tests that genuinely need local assets must skip with an
explicit reason or use the registered `integration` marker. Never convert a real
failure into a skip, and never weaken a test to get a green result.

## Working method

- Plan before implementing. Read the relevant spec first.
- Validate directly. Never declare success from file inspection alone; run the
  thing and report the actual output.
- Report honestly. If tests fail, say so with the output. If something is
  blocked, say what and why rather than quietly narrowing scope.
- Evaluate feasibility and alignment with project intent. Push back with
  reasoning when a request looks wrong, then proceed if the owner confirms.
- Ask when a decision is genuinely ambiguous and the answer changes the work.

## Data, checkpoints, and outputs

Never commit, package, or redistribute:

```txt
CubiCasa5K data and anything derived from it   docs/high_quality_architectural/
generated datasets, caches, feature caches     features/, data/
training checkpoints                           checkpoints_CNN/, checkpoints_Raster2Graph/
experiment outputs and scratch runs            outputs/, runs/, tmp/
the local model release bundle                 dist/
```

Deliberately versioned: `docs/images/` (curated README evidence),
`evaluation/results/` (compact metrics), `splits/` (canonical manifests),
`tests/` fixtures.

Preserve whole checkpoint folders locally so previous models remain usable.

## Documentation

Keep durable information in its canonical location and nowhere else:

```txt
README.md                                public overview, real evidence, results,
                                         setup, usage, license
CLAUDE.md                                this file - agent instructions
specs/spec_v006_evaluation.md            the evaluation contract
specs/vectorization_must_rules.md        current architectural invariants
specs/vectorization_phase_history.md     durable lessons from superseded phases
specs/spec_v0NN_*.md                     current implementation specs
```

Rules:

- One source of truth per fact. If two documents would state it, pick the
  canonical one and link from the other.
- README numbers are generated from `evaluation/results/` by
  `scripts/sync_readme_metrics.py`. Never hand-edit them; a test enforces this.
- Update the relevant spec when behavior changes, in the same change.
- New specs follow `specs/spec_vNNN_short_name.md`.
- No placeholders, invented metrics, guessed URLs, or claims that have not been
  directly validated.
- Never describe output as pixel-perfect or production-ready.
