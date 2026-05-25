"""Convert CubiCasa SVG annotations to raster PNG files (spec_v003)."""

import argparse
import logging
from pathlib import Path

import cairosvg

logger = logging.getLogger(__name__)

OUTPUT_NAME = "model_clean.png"
SVG_NAME = "model.svg"


def convert_svg_to_png(svg_path: Path, output_path: Path) -> None:
    """Render an SVG file to a white-background PNG at native SVG dimensions."""
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(output_path),
        background_color="white",
    )


def process_dataset(root_dir: Path, overwrite: bool = False) -> tuple[int, int]:
    """Walk root_dir subdirectories, convert each model.svg to model_clean.png.

    Returns (converted, skipped) counts.
    """
    converted = 0
    skipped = 0

    svg_files = sorted(root_dir.rglob(SVG_NAME))
    total = len(svg_files)
    logger.info("Found %d SVG files under %s", total, root_dir)

    for i, svg_path in enumerate(svg_files, 1):
        output_path = svg_path.parent / OUTPUT_NAME
        if output_path.exists() and not overwrite:
            logger.debug("[%d/%d] Skipping (already exists): %s", i, total, output_path)
            skipped += 1
            continue

        try:
            convert_svg_to_png(svg_path, output_path)
            logger.info("[%d/%d] Converted: %s", i, total, output_path)
            converted += 1
        except Exception:
            logger.exception("[%d/%d] Failed: %s", i, total, svg_path)

    return converted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CubiCasa SVGs to PNG rasters.")
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory containing per-sample subdirectories with model.svg files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert even if model_clean.png already exists.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    converted, skipped = process_dataset(args.root_dir, overwrite=args.overwrite)
    logger.info("Done. converted=%d  skipped=%d", converted, skipped)


if __name__ == "__main__":
    main()
