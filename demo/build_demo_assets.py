"""Build the static demo asset bundle from existing pipeline output.

Read-only with respect to the rest of the repository: this script copies
artifacts out of ``outputs/vectorization/phase4_vectorization/`` into
``demo/assets/`` and writes a single ``data.js`` payload the static page reads.
It never runs the pipeline and never modifies project code.

    python demo/build_demo_assets.py

Sanitising: ``final_vector.json`` records an absolute ``preprocessing.
source_image`` path from the machine that produced it. That path is stripped
from every copied artifact so no local filesystem layout is published.

The source floor plans are CubiCasa5K renderings, licensed CC BY-NC 4.0.
See demo/NOTICE-demo.txt for the attribution that must travel with them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent

DEFAULT_SOURCE = PROJECT_ROOT / "outputs" / "vectorization" / "phase4_vectorization"
DEFAULT_OUT = DEMO_DIR / "assets"

SAMPLE_IDS = ["1316", "10026", "10029"]

# (artifact filename in the run directory, published filename)
STAGE_IMAGES = [
    ("input.png", "01-input.png"),
    ("image_segmentation.png", "02-segmentation.png"),
    ("graph_overlay.png", "03-wall-graph.png"),
    ("graph_overlay_aligned.png", "04-aligned.png"),
    ("image_debug_overlay.png", "05-hosting.png"),
]

# Keys copied out of final_vector.json["metrics"]. Everything shown on the page
# comes from here or from ["scale"] / ["preprocessing"] - nothing is invented.
METRIC_KEYS = [
    "elapsed_s",
    "r2g_nodes",
    "r2g_edges",
    "aligned_edges",
    "trimmed_wall_edges",
    "opening_gaps",
    "hosted_doors",
    "hosted_windows",
    "rejected_total",
    "wall_chains",
    "wall_thickness_mm",
    "px_to_mm",
    "scale_status",
    "pre_buffer_node_count",
    "post_snap_node_count",
    "disconnected_endpoint_count",
]

PREPROCESSING_KEYS = [
    "content_bbox_original",
    "padding_fraction",
    "coordinate_space",
    "source_variant",
    "content_touches_edge",
    "content_bbox_after_preprocess",
]

SCALE_KEYS = ["status", "px_to_mm", "source", "confidence"]


def sanitise_final_vector(raw: dict) -> dict:
    """Drop the absolute source path recorded by the run machine."""
    cleaned = dict(raw)
    pre = dict(cleaned.get("preprocessing", {}))
    pre.pop("source_image", None)
    cleaned["preprocessing"] = pre
    return cleaned


def build_sample(sample_id: str, source_dir: Path, out_dir: Path) -> dict:
    run_dir = source_dir / sample_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No run directory for sample {sample_id!r}: {run_dir}")

    sample_out = out_dir / "samples" / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)

    for src_name, published_name in STAGE_IMAGES:
        src = run_dir / src_name
        if not src.is_file():
            raise FileNotFoundError(f"Missing artifact: {src}")
        shutil.copy2(src, sample_out / published_name)

    svg_text = (run_dir / "final_vector.svg").read_text(encoding="utf-8")
    (sample_out / "final_vector.svg").write_text(svg_text, encoding="utf-8")

    raw = json.loads((run_dir / "final_vector.json").read_text(encoding="utf-8"))
    cleaned = sanitise_final_vector(raw)
    (sample_out / "final_vector.json").write_text(json.dumps(cleaned, indent=2), encoding="utf-8")

    metrics = cleaned.get("metrics", {})
    scale = cleaned.get("scale", {})
    pre = cleaned.get("preprocessing", {})
    openings = cleaned.get("openings", {})
    geometry = cleaned.get("geometry", {})

    return {
        "id": sample_id,
        # Counts taken from the geometry actually exported, so what the page
        # states matches what the SVG draws.
        "counts": {
            "walls": len(geometry.get("walls", [])),
            "doors": len(geometry.get("doors", [])),
            "windows": len(geometry.get("windows", [])),
            "rejected": len(openings.get("rejected", [])),
        },
        "metrics": {k: metrics[k] for k in METRIC_KEYS if k in metrics},
        "scale": {k: scale[k] for k in SCALE_KEYS if k in scale},
        "preprocessing": {k: pre[k] for k in PREPROCESSING_KEYS if k in pre},
        "images": {
            "input": f"assets/samples/{sample_id}/01-input.png",
            "segmentation": f"assets/samples/{sample_id}/02-segmentation.png",
            "graph": f"assets/samples/{sample_id}/03-wall-graph.png",
            "aligned": f"assets/samples/{sample_id}/04-aligned.png",
            "hosting": f"assets/samples/{sample_id}/05-hosting.png",
        },
        "downloads": {
            "svg": f"assets/samples/{sample_id}/final_vector.svg",
            "json": f"assets/samples/{sample_id}/final_vector.json",
        },
        # Inlined so the page works from file:// with no server and no fetch().
        "svg": svg_text,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", nargs="+", default=SAMPLE_IDS)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    samples = [build_sample(s, args.source, args.out) for s in args.samples]

    payload = {
        "generator": "demo/build_demo_assets.py",
        "source_run": str(args.source.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "samples": samples,
    }

    data_js = args.out / "data.js"
    data_js.write_text(
        "// Generated by demo/build_demo_assets.py - do not edit by hand.\n"
        "window.DEMO_DATA = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )

    print(f"Wrote {data_js.relative_to(PROJECT_ROOT)}")
    for s in samples:
        c = s["counts"]
        print(
            f"  {s['id']:>6}  walls={c['walls']:<3} doors={c['doors']:<3} "
            f"windows={c['windows']:<3} rejected={c['rejected']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
