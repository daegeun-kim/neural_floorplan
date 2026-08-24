# Neural Floorplan — static demo

A self-contained static page that replays the vectorization pipeline one stage
at a time for three floor plans. Styled to match
<https://daegeunkim.com/neural-floorplan/> and meant to be linked or iframed
from it.

Nothing in here modifies the project. The demo reads artifacts that the
pipeline already wrote and copies them into `assets/`.

## Why it replays instead of running

Live in-browser inference is not practical for this pipeline:

- the Raster-to-Graph checkpoint is ~397 MB, and its decoder is an iterative
  Deformable-DETR-style graph decoder rather than a single-shot forward pass
- the Phase 4 geometry stage is several thousand lines of Python

So the page shows real artifacts — stage images and the exported SVG — as
static files. It says so on the page, in the title block stamp and in the
footer.

## What it shows

Six stages, driven by `assets/data.js`:

| # | Stage | Artifact |
|---|---|---|
| 1 | Input | `01-input.png` |
| 2 | Segmentation | `02-segmentation.png` |
| 3 | Wall graph | `03-wall-graph.png` |
| 4 | Alignment | `04-aligned.png` |
| 5 | Opening hosting | `05-hosting.png` |
| 6 | Classified vector | `final_vector.svg`, inlined for layer toggles |

Every number in the title block is read from that sample's own
`final_vector.json` — node and edge counts, hosted openings, wall chains,
`px_to_mm`, scale status and confidence, stage runtime. None of it is
hard-coded per sample.

## Rebuild the assets

**The generated assets are not committed.** `demo/assets/samples/` and
`demo/assets/data.js` are renderings of, and pipeline output derived from,
CubiCasa5K, so they are gitignored under the same rule that covers `outputs/`
and `docs/high_quality_architectural/`. A fresh clone has the page, its
stylesheet, its script and this builder, but no artifacts &mdash; the page will
say so until you run:

```bash
python demo/build_demo_assets.py
```

Reads from `outputs/vectorization/phase4_vectorization/` and writes
`demo/assets/samples/` plus `demo/assets/data.js`. Options:

```bash
python demo/build_demo_assets.py --samples 1316 10026 10029 --source <run-dir>
```

The builder strips `preprocessing.source_image` from every copied
`final_vector.json`, because that field records an absolute path on the
machine that produced the run.

## Preview locally

The page loads its data through `<script src>` rather than `fetch()`, so
opening `demo/index.html` directly from disk works. To serve it instead:

```bash
python -m http.server 8777
```

Then open <http://localhost:8777/demo/index.html>.

## Styling

The page reuses the design tokens from `daegeunkim.com/shared/site.css`
(`--bg`, `--ink`, `--line`, `--blue`, the `--space-*` and `--fs-*` scales,
`--radius`, `--ease`) so it reads as part of that site. It loads **no
webfonts** — the portfolio uses a system Helvetica stack, and matching it
means the demo makes zero external requests.

The only colors not taken from the portfolio are the classified-geometry
ones (`--wall`, `--window`, `--door-origin`, `--door-leaf`, `--door-arc`).
Those are copied from `final_vector.svg` and must keep matching what the
drawing actually draws.

If the portfolio tokens change, update the `:root` block at the top of
`assets/styles.css` to match.

## Embed mode

`index.html?embed=1` hides the page masthead so the instrument can sit in an
iframe on the project page without repeating its title:

```html
<iframe src="/neural-floorplan/demo/?embed=1"
        title="Neural Floorplan pipeline demo"
        loading="lazy"
        style="width:100%;height:1180px;border:1px solid var(--line);border-radius:var(--radius)"></iframe>
```

The colophon stays visible in embed mode on purpose — the CubiCasa5K
attribution is a license condition and has to travel with the page.

## Deploy

All paths are relative, so the folder works at any subpath. Two options:

**A — serve it from the portfolio site (in use).** The published copy lives in
the `DaegeunKim_website` repository at `neural-floorplan/demo/` and is served
from:

    https://daegeunkim.com/neural-floorplan/demo/

That copy is the documented redistribution point for the CubiCasa5K-derived
artifacts, the same carve-out `docs/images/` gets here; it carries the
attribution in its page footer. It has deliberately diverged from this one:
it loads the site's `shared/site.css` for design tokens and uses the shared
header and footer, so only `assets/samples/` and `assets/data.js` should be
copied across after a rebuild &mdash; not `index.html` or `assets/styles.css`.

**B — serve it from GitHub Pages.** In this repo, Settings → Pages → Source
"Deploy from a branch" → `main` / `(root)`. After it builds:

    https://daegeun-kim.github.io/neural_floorplan/demo/

Pushing alone does not publish; Pages has to be enabled once. If Pages ever
mangles the asset paths, add an empty `.nojekyll` file at the repository root
to skip Jekyll processing.

Either way, keep the CubiCasa5K attribution and the source-repository link in
the footer — both are license obligations, see `NOTICE-demo.txt`.

Total bundle is roughly 750 KB, static, no build step, no dependencies.

## Caveats worth keeping visible

- **These are training-split samples.** `1316`, `10026` and `10029` all appear
  in `splits/train.json` and are not part of the evaluation population in
  `evaluation/results/v1/`, so they carry no held-out metrics. The page states
  this in the footer. For held-out results, `evaluation/results/v1/
  selected_cases.json` names the deterministically chosen success, mixed and
  failure cases.
- **The artifacts predate the current pipeline.** The files under
  `outputs/vectorization/phase4_vectorization/` were written 2026-06-28;
  most Phase 4 modules were last changed 2026-08-08. Re-running the pipeline
  for these three samples and rebuilding would bring the demo in line with the
  current code.

## Licensing

Code here is GPL-3.0-only with the rest of the repository. The sample
artifacts are derived from CubiCasa5K and are CC BY-NC 4.0, not GPL. See
[`NOTICE-demo.txt`](NOTICE-demo.txt).
