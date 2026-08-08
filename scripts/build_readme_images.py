"""Build the curated README image set from real pipeline artifacts.

Reproducible: reads the deterministically selected cases from an evaluation run
(``selected_cases.json``) and copies/renders only the small set of images the
README actually shows. Nothing is invented or hand-picked.

    python scripts/build_readme_images.py \
        --work-dir outputs/evaluation_work \
        --results evaluation/results/v1 \
        --out docs/images

The source floor plans are CubiCasa5K renderings, licensed CC BY-NC 4.0. See
`NOTICE` section 5 for the attribution that must travel with these images.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Web display width. The pipeline canvas is 512 px, so this neither upscales
# nor loses meaningful detail.
TARGET_WIDTH = 512

# (artifact filename, output stem) copied for the primary success case only.
STAGE_ARTIFACTS = [
    ("input.png", "stage-1-input"),
    ("image_segmentation.png", "stage-2-segmentation"),
    ("graph_overlay_aligned.png", "stage-3-wall-graph"),
    ("image_debug_overlay.png", "stage-4-opening-hosting"),
]


def render_svg(svg_path: Path, out_path: Path, width: int = TARGET_WIDTH) -> None:
    from src.cairo_runtime import load_cairosvg

    cairosvg = load_cairosvg()
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(out_path),
        output_width=width,
        output_height=width,
        background_color="white",
    )


def optimize(path: Path, width: int = TARGET_WIDTH) -> None:
    """Downscale to web width and re-save with PNG optimization."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.width != width:
            height = round(img.height * width / img.width)
            img = img.resize((width, height), Image.LANCZOS)
        img.save(path, format="PNG", optimize=True)


def side_by_side(left: Path, right: Path, out_path: Path, gap: int = 16) -> None:
    """Compose the before/after hero image."""
    with Image.open(left) as a, Image.open(right) as b:
        a = a.convert("RGB")
        b = b.convert("RGB")
        height = max(a.height, b.height)
        canvas = Image.new("RGB", (a.width + gap + b.width, height), (255, 255, 255))
        canvas.paste(a, (0, (height - a.height) // 2))
        canvas.paste(b, (a.width + gap, (height - b.height) // 2))
        canvas.save(out_path, format="PNG", optimize=True)


def build_case(work_dir: Path, sample_id: str, out_dir: Path, prefix: str) -> None:
    """Emit the before/after pair for one case."""
    source = work_dir / sample_id
    if not source.exists():
        raise FileNotFoundError(f"missing pipeline artifacts: {source}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{prefix}-", dir=out_dir) as temp_dir:
        before = Path(temp_dir) / "input.png"
        after = Path(temp_dir) / "vector.png"

        shutil.copy2(source / "input.png", before)
        optimize(before)
        render_svg(source / "final_vector.svg", after)
        optimize(after)
        side_by_side(before, after, out_dir / f"{prefix}-before-after.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/evaluation_work"))
    parser.add_argument("--results", type=Path, default=Path("evaluation/results/v1"))
    parser.add_argument("--out", type=Path, default=Path("docs/images"))
    args = parser.parse_args()

    selected_path = args.results / "selected_cases.json"
    if not selected_path.exists():
        print(f"error: {selected_path} not found; run the evaluation first", file=sys.stderr)
        return 2

    cases = json.loads(selected_path.read_text(encoding="utf-8"))["cases"]
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for label in ("success", "mixed", "failure"):
        if label not in cases:
            continue
        build_case(args.work_dir, cases[label]["sample_id"], out_dir, label)
        print(f"  {label:8s} {cases[label]['sample_id']}")

    # Pipeline stage strip, taken from the success case so each stage is legible.
    success_id = cases["success"]["sample_id"]
    source = args.work_dir / success_id
    for filename, stem in STAGE_ARTIFACTS:
        src = source / filename
        if not src.exists():
            print(f"  warning: missing {src}", file=sys.stderr)
            continue
        dest = out_dir / f"{stem}.png"
        shutil.copy2(src, dest)
        optimize(dest)
        print(f"  stage    {stem}.png")

    dest = out_dir / "stage-5-final-vector.png"
    render_svg(source / "final_vector.svg", dest)
    optimize(dest)
    print("  stage    stage-5-final-vector.png")

    total = sum(p.stat().st_size for p in out_dir.glob("*.png"))
    print(f"\n{len(list(out_dir.glob('*.png')))} images, {total / 1024:.0f} KB total")
    print(f"cases: {json.dumps({k: v['sample_id'] for k, v in cases.items()})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
