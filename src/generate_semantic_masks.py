"""Generate semantic masks from CubiCasa5K SVG annotations (spec_v003).

CubiCasa5K SVG structure (actual):
  svg
    g[id=Model]
      g[class=Floor]
        g[class="Floorplan Floor-1"]   ← floor container
          g[id=uuid, class="Space ..."]     ← room
          g[id=Wall, class="Wall ..."]      ← wall (may contain Door/Window children)
            g[id=Door]                      ← opening nested in wall
            g[id=Window]                    ← opening nested in wall
          g[id=FixedFurnitureSet]           ← icon/furniture

Room and opening elements have white/light fills in the SVG.  We force them
to black before rendering so the binary mask captures the correct area.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from copy import deepcopy
from pathlib import Path

import cairosvg
import numpy as np
from lxml import etree
from PIL import Image

logger = logging.getLogger(__name__)

SVG_NAME = "model.svg"
MASKS_DIR = "masks"

CLASS_IDS: dict[str, int] = {
    "background": 0,
    "wall": 1,
    "opening": 2,
    "room": 3,
    "icon": 4,
}

# Priority order: last applied = highest priority
APPLY_ORDER = ["room", "icon", "opening", "wall"]

DEBUG_COLORS: dict[str, tuple[int, int, int]] = {
    "wall": (0, 0, 0),
    "opening": (220, 50, 50),
    "room": (100, 180, 220),
    "icon": (50, 180, 80),
}

# Binary mask threshold: pixel value < this → class present
# 200 avoids misclassifying window glass (#f0f0ff ≈ 242) as wall
_MASK_THRESHOLD = 200


# ---------------------------------------------------------------------------
# SVG parsing
# ---------------------------------------------------------------------------


def _parse_svg(svg_path: Path) -> etree._Element:
    parser = etree.XMLParser(remove_comments=True)
    return etree.parse(str(svg_path), parser).getroot()


def _get_dimensions(svg_root: etree._Element, reference: Path | None) -> tuple[int, int]:
    """Return (width, height) for mask output, preferring reference image."""
    if reference and reference.exists():
        with Image.open(reference) as img:
            return img.size  # (width, height)

    vb = svg_root.get("viewBox", "")
    if vb:
        parts = vb.split()
        if len(parts) == 4:
            try:
                return max(1, int(float(parts[2]))), max(1, int(float(parts[3])))
            except ValueError:
                pass

    try:
        w = int(float(svg_root.get("width", "0").rstrip("px")))
        h = int(float(svg_root.get("height", "0").rstrip("px")))
        if w > 0 and h > 0:
            return w, h
    except ValueError:
        pass

    return 800, 600  # safe fallback


def _is_hidden(el: etree._Element) -> bool:
    style = el.get("style", "")
    return "display: none" in style or "display:none" in style


def _local_name(el: etree._Element) -> str:
    try:
        return etree.QName(el.tag).localname
    except Exception:
        return ""


def _find_floor_containers(svg_root: etree._Element) -> list[etree._Element]:
    """Find all <g class="Floorplan ..."> containers (one per floor)."""
    containers = []
    for el in svg_root.iter():
        if el.tag is etree.Comment:
            continue
        if _local_name(el) != "g":
            continue
        cls_parts = (el.get("class") or "").split()
        if "Floorplan" in cls_parts:
            containers.append(el)
    return containers


def _classify_floor_child(el: etree._Element) -> str | None:
    """Classify a direct child of a floor container into a semantic category.

    Returns "wall", "room", "opening", "icon", or None.
    """
    if el.tag is etree.Comment or _is_hidden(el) or _local_name(el) != "g":
        return None

    elem_id = (el.get("id") or "").strip()
    cls_parts = (el.get("class") or "").split()

    # Wall: id="Wall" or class contains "Wall"
    if elem_id == "Wall" or "Wall" in cls_parts:
        return "wall"

    # Room/Space: class contains "Space", or id is "Space"/"Room"
    if "Space" in cls_parts or elem_id.lower() in ("space", "room"):
        return "room"

    # Opening: top-level Door or Window (test SVGs / non-standard)
    if elem_id in ("Door", "Window") or "Door" in cls_parts or "Window" in cls_parts:
        return "opening"

    # Icon/Furniture: FixedFurnitureSet or FixedFurniture
    if (
        elem_id in ("FixedFurnitureSet", "FixedFurniture")
        or "FixedFurnitureSet" in cls_parts
        or "FixedFurniture" in cls_parts
        or "FixedFurniture" in elem_id
    ):
        return "icon"

    return None


def _get_door_window_children(wall_el: etree._Element) -> list[etree._Element]:
    """Return direct Door/Window child elements of a wall element."""
    result = []
    for child in wall_el:
        if child.tag is etree.Comment or _local_name(child) != "g":
            continue
        child_id = (child.get("id") or "").strip()
        if child_id in ("Door", "Window"):
            result.append(child)
    return result


def _collect_elements_by_category(svg_root: etree._Element) -> dict[str, list[etree._Element]]:
    """Return all semantic elements grouped by category."""
    walls: list[etree._Element] = []
    rooms: list[etree._Element] = []
    icons: list[etree._Element] = []
    openings: list[etree._Element] = []

    containers = _find_floor_containers(svg_root)
    if not containers:
        # Fallback: treat root itself as the container (e.g. minimal test SVGs)
        containers = [svg_root]

    for container in containers:
        for child in container:
            category = _classify_floor_child(child)
            if category == "wall":
                walls.append(child)
                for dw in _get_door_window_children(child):
                    openings.append(dw)
            elif category == "room":
                rooms.append(child)
            elif category == "icon":
                icons.append(child)
            elif category == "opening":
                # Top-level Door/Window (test SVGs or non-standard structures)
                openings.append(child)

    logger.debug(
        "Collected: walls=%d  rooms=%d  openings=%d  icons=%d",
        len(walls), len(rooms), len(openings), len(icons),
    )
    return {"wall": walls, "room": rooms, "opening": openings, "icon": icons}


# ---------------------------------------------------------------------------
# Mask rendering
# ---------------------------------------------------------------------------


def _force_fill_black(el: etree._Element) -> None:
    """Recursively set fill=#000000 on element and all visible descendants."""
    if el.tag is etree.Comment or _is_hidden(el):
        return

    el.set("fill", "#000000")
    if "stroke" in el.attrib:
        el.set("stroke", "none")

    # Strip fill/stroke/opacity from inline style to avoid CSS overrides
    style = el.get("style", "")
    if style:
        kept = []
        for part in style.split(";"):
            part = part.strip()
            if not part:
                continue
            prop = part.split(":")[0].strip().lower()
            if prop in ("fill", "stroke", "fill-opacity", "stroke-opacity", "stroke-width"):
                continue
            kept.append(part)
        el.set("style", "; ".join(kept))

    for child in el:
        _force_fill_black(child)


def _make_renderable_svg(
    svg_root: etree._Element,
    elements: list[etree._Element],
    force_black: bool = False,
) -> bytes:
    """Build a minimal SVG bytes object containing only *elements*."""
    new_root = etree.Element(svg_root.tag, nsmap=svg_root.nsmap)
    for attr in ("viewBox", "width", "height", "version"):
        val = svg_root.get(attr)
        if val:
            new_root.set(attr, val)

    # Preserve <defs> (gradients, patterns referenced by children)
    for child in svg_root:
        if child.tag is etree.Comment:
            continue
        if _local_name(child) == "defs":
            new_root.append(deepcopy(child))
            break

    for el in elements:
        el_copy = deepcopy(el)
        if force_black:
            _force_fill_black(el_copy)
        new_root.append(el_copy)

    return etree.tostring(new_root, encoding="UTF-8", xml_declaration=True)


def _render_mask(svg_bytes: bytes, width: int, height: int) -> np.ndarray:
    """Render SVG to a binary uint8 mask (0 = absent, 255 = present)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cairosvg.svg2png(
            bytestring=svg_bytes,
            write_to=str(tmp_path),
            output_width=width,
            output_height=height,
            background_color="white",
        )
        with Image.open(tmp_path) as img:
            arr = np.array(img.convert("L"))
    finally:
        tmp_path.unlink(missing_ok=True)

    return np.where(arr < _MASK_THRESHOLD, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-sample mask generation
# ---------------------------------------------------------------------------

# Categories that use white/light fills in the SVG — must be forced black
_FORCE_BLACK_CATEGORIES = {"room", "opening", "icon"}


def generate_masks(
    sample_dir: Path,
    output_dir_name: str = MASKS_DIR,
    overwrite: bool = False,
    debug_overlays: bool = False,
    verbose: bool = False,
) -> dict:
    """Process one sample directory. Returns a status dict."""
    svg_path = sample_dir / SVG_NAME
    if not svg_path.exists():
        logger.warning("No model.svg — skipping: %s", sample_dir)
        return {"status": "missing_svg"}

    masks_dir = sample_dir / output_dir_name
    semantic_map_path = masks_dir / "semantic_class_map.png"

    if semantic_map_path.exists() and not overwrite:
        logger.debug("Already exists, skipping: %s", sample_dir)
        return {"status": "skipped"}

    masks_dir.mkdir(exist_ok=True)

    try:
        svg_root = _parse_svg(svg_path)
    except Exception:
        logger.exception("SVG parse error: %s", svg_path)
        return {"status": "failed", "reason": "svg_parse_error"}

    reference = sample_dir / "model_clean.png"
    width, height = _get_dimensions(svg_root, reference)

    elements_by_cat = _collect_elements_by_category(svg_root)

    category_masks: dict[str, np.ndarray] = {}
    missing_classes: list[str] = []

    for category in ["wall", "opening", "room", "icon"]:
        elements = elements_by_cat.get(category, [])

        if not elements:
            logger.debug("No elements for '%s' in %s", category, sample_dir.name)
            missing_classes.append(category)
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            try:
                force = category in _FORCE_BLACK_CATEGORIES
                svg_bytes = _make_renderable_svg(svg_root, elements, force_black=force)
                mask = _render_mask(svg_bytes, width, height)
                if verbose:
                    logger.info("  %s: %d nonzero px", category, int(np.count_nonzero(mask)))
            except Exception:
                logger.exception("Render failed for '%s' in %s", category, sample_dir.name)
                mask = np.zeros((height, width), dtype=np.uint8)
                missing_classes.append(category)

        Image.fromarray(mask, mode="L").save(masks_dir / f"{category}_mask.png")
        category_masks[category] = mask

    # Build semantic class map — later assignment = higher priority
    semantic_map = np.zeros((height, width), dtype=np.uint8)
    for cat in APPLY_ORDER:
        if cat in category_masks:
            semantic_map[category_masks[cat] > 0] = CLASS_IDS[cat]

    Image.fromarray(semantic_map, mode="L").save(semantic_map_path)

    counts = {
        name: int(np.sum(semantic_map == cid)) for name, cid in CLASS_IDS.items()
    }
    suspicious = counts["wall"] == 0

    metadata = {
        "source_svg": SVG_NAME,
        "width": width,
        "height": height,
        "classes": {str(v): k for k, v in CLASS_IDS.items()},
        "class_pixel_counts": counts,
        "missing_classes": missing_classes,
        "status": "suspicious" if suspicious else "ok",
    }
    with open(masks_dir / "mask_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if debug_overlays:
        _debug_overlay(sample_dir, masks_dir, category_masks, width, height)

    status = "suspicious" if suspicious else "ok"
    logger.debug("Sample %s → %s", sample_dir.name, status)
    return {"status": status, "missing_classes": missing_classes}


def _debug_overlay(
    sample_dir: Path,
    masks_dir: Path,
    masks: dict[str, np.ndarray],
    width: int,
    height: int,
) -> None:
    ref = sample_dir / "model_clean.png"
    if ref.exists():
        with Image.open(ref) as img:
            base = np.array(img.convert("RGB").resize((width, height)))
    else:
        base = np.full((height, width, 3), 255, dtype=np.uint8)

    for cat, color in DEBUG_COLORS.items():
        if cat in masks:
            base[masks[cat] > 0] = color

    Image.fromarray(base).save(masks_dir / "debug_overlay.png")


# ---------------------------------------------------------------------------
# Dataset-level processing
# ---------------------------------------------------------------------------


def process_dataset(
    root_dir: Path,
    output_dir_name: str = MASKS_DIR,
    overwrite: bool = False,
    debug_overlays: bool = False,
    verbose: bool = False,
) -> dict:
    """Process all immediate subdirectories of root_dir."""
    sample_dirs = sorted(d for d in root_dir.iterdir() if d.is_dir())
    total = len(sample_dirs)
    counts = {
        "processed": 0,
        "skipped_existing": 0,
        "missing_svg": 0,
        "failed": 0,
        "suspicious": 0,
    }

    for i, sample_dir in enumerate(sample_dirs, 1):
        logger.info("[%d/%d] %s", i, total, sample_dir.name)
        result = generate_masks(sample_dir, output_dir_name, overwrite, debug_overlays, verbose)
        status = result.get("status", "failed")

        if status == "skipped":
            counts["skipped_existing"] += 1
        elif status == "missing_svg":
            counts["missing_svg"] += 1
        elif status == "failed":
            counts["failed"] += 1
        elif status == "suspicious":
            counts["processed"] += 1
            counts["suspicious"] += 1
        else:
            counts["processed"] += 1

    summary_path = root_dir / "semantic_mask_generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(counts, f, indent=2)

    logger.info(
        "Done. processed=%d  skipped=%d  missing_svg=%d  failed=%d  suspicious=%d",
        counts["processed"],
        counts["skipped_existing"],
        counts["missing_svg"],
        counts["failed"],
        counts["suspicious"],
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate semantic masks from CubiCasa SVGs.")
    parser.add_argument("root_dir", type=Path, help="Dataset root with per-sample subdirectories.")
    parser.add_argument("--overwrite", action="store_true", help="Re-generate existing masks.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir-name", default=MASKS_DIR, metavar="NAME")
    parser.add_argument("--debug-overlays", action="store_true", help="Save debug_overlay.png.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    process_dataset(
        args.root_dir,
        output_dir_name=args.output_dir_name,
        overwrite=args.overwrite,
        debug_overlays=args.debug_overlays,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
