"""Convert CubiCasa SVG annotations to raster PNG files (spec_v003, task01 enhanced).

Task 01 adds normalize_svg_visibility() which strips display:none from <g class="Floor">
wrapper elements so hidden floors (Floor-2, etc.) appear in the exported raster.

Background: cairosvg respects display:none, so multi-floor SVGs that hide one floor's
wrapper group export a raster missing that floor's geometry.  The semantic mask
generator (generate_semantic_masks.py) already collects all Floorplan containers
regardless of their parent display state, so masks and rasters were misaligned.
Stripping display:none from the Floor wrappers realigns them.

FloorsCompose is intentionally left hidden: its <use> transforms would shift floor
coordinates away from the natural SVG positions used by the mask generator, breaking
raster/mask pixel alignment.
"""

import argparse
import logging
import re
from io import BytesIO
from pathlib import Path

import cairosvg
import numpy as np
from lxml import etree
from PIL import Image

logger = logging.getLogger(__name__)

OUTPUT_NAME = "model_clean.png"
SVG_NAME = "model.svg"

_MIN_NON_WHITE_PIXELS = 100


def _css_classes(elem) -> set[str]:
    """Return the set of CSS class tokens on an element."""
    return set((elem.get("class") or "").split())


def _has_display_none(elem) -> bool:
    return bool(re.search(r"(?i)display\s*:\s*none", elem.get("style") or ""))


def _strip_display_none(style: str) -> str:
    """Remove 'display: none' (case-insensitive) from a CSS style string."""
    style = re.sub(r"(?i)display\s*:\s*none\s*;?\s*", "", style)
    return style.strip().strip(";").strip()


def _find_model_element(root):
    """Return the <g class='Model ...'> element, falling back to root."""
    for child in root:
        if "Model" in _css_classes(child):
            return child
    for child in root:
        for grandchild in child:
            if "Model" in _css_classes(grandchild):
                return grandchild
    return root


def normalize_svg_visibility(svg_bytes: bytes) -> tuple[bytes, list[str]]:
    """Strip display:none from <g class='Floor'> wrapper elements.

    Only targets direct children of the top-level Model group whose class
    attribute contains 'Floor' but not 'FloorsCompose'.  All other hidden
    elements (dimension marks, UI controls, FloorsCompose) are left untouched.

    Returns (modified_svg_bytes, list_of_change_descriptions).
    """
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(BytesIO(svg_bytes), parser)
    root = tree.getroot()
    changes: list[str] = []

    model = _find_model_element(root)

    for child in model:
        classes = _css_classes(child)
        if "Floor" not in classes or "FloorsCompose" in classes:
            continue
        if _has_display_none(child):
            style = child.get("style", "")
            child.set("style", _strip_display_none(style))
            floor_id = child.get("id") or child.get("class") or "?"
            changes.append(
                f"Unhid Floor group '{floor_id}' (class='{child.get('class')}')"
            )

    out = BytesIO()
    tree.write(out, xml_declaration=True, encoding="UTF-8")
    return out.getvalue(), changes


def _count_non_white_pixels(png_path: Path) -> int:
    """Count pixels that are not nearly-white (any channel below 250)."""
    img = np.array(Image.open(png_path).convert("RGB"))
    return int(np.any(img < 250, axis=-1).sum())


def convert_svg_to_png(svg_path: Path, output_path: Path) -> list[str]:
    """Normalize SVG floor visibility and render to a white-background PNG.

    Returns a list of visibility-fix descriptions (empty when no fixes were needed).
    """
    svg_bytes = svg_path.read_bytes()
    normalized, changes = normalize_svg_visibility(svg_bytes)
    cairosvg.svg2png(
        bytestring=normalized,
        write_to=str(output_path),
        background_color="white",
    )
    return changes


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
            changes = convert_svg_to_png(svg_path, output_path)
            for change in changes:
                logger.info("[%d/%d] Visibility fix: %s", i, total, change)

            non_white = _count_non_white_pixels(output_path)
            if non_white < _MIN_NON_WHITE_PIXELS:
                logger.warning(
                    "[%d/%d] Suspicious blank output: %d non-white pixels in %s",
                    i, total, non_white, output_path,
                )
            else:
                logger.info(
                    "[%d/%d] Converted: %s (%d non-white px)",
                    i, total, output_path, non_white,
                )
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
