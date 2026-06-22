"""Generate semantic masks from CubiCasa5K SVG annotations (spec_v005 run3).

CubiCasa5K SVG structure (actual):
  svg
    g[id=Model]
      g[class=Floor]
        g[class="Floorplan Floor-1"]   ← floor container
          g[id=uuid, class="Space ..."]     ← floor
          g[id=Wall, class="Wall ..."]      ← wall (may contain Door/Window children)
            g[id=Window]                      ← window opening nested in wall
            g[id=Door]                        ← door opening nested in wall
              g[id=Threshold]                   ← polygon; door_origin = its bbox centerline
              g[id=Panel, class="Panel ..."]    ← door swing evidence
                g[id=PanelArea]                    ← not used directly
                path d="M... q... l...Z"           ← q=door_arc wedge, l=door_leaf line

Floor and window elements have white/light fills in the SVG.  We force them to black
before rendering so the binary mask captures the correct area.  door_origin/door_arc/
door_leaf are synthesized as fixed-width stroked lines (or a filled wedge for door_arc),
not deep copies of the original elements — see _collect_elements_by_category.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
    "floor": 1,
    "wall": 2,
    "window": 3,
    "door_arc": 4,
    "door_leaf": 5,
    "door_origin": 6,
}

# Priority order: last applied = highest priority (spec_v005 run3 §11)
APPLY_ORDER = ["floor", "wall", "window", "door_origin", "door_arc", "door_leaf"]

DEBUG_COLORS: dict[str, tuple[int, int, int]] = {
    "floor": (245, 240, 232),
    "wall": (30, 30, 30),
    "window": (60, 120, 220),
    "door_arc": (220, 90, 90),
    "door_leaf": (235, 140, 80),
    "door_origin": (160, 70, 180),
}

# Binary mask threshold: pixel value < this → class present
# 200 avoids misclassifying window glass (#f0f0ff ≈ 242) as wall
_MASK_THRESHOLD = 200

# Fixed stroke width (native px) shared by door_leaf and door_origin lines. Constant across
# every sample regardless of the SVG's native resolution or the door's wall thickness, so
# every door_leaf/door_origin line is rendered at the same width.
_DOOR_STROKE_WIDTH_PX = 6


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


def _class_tokens(el: etree._Element) -> list[str]:
    return (el.get("class") or "").split()


def _find_floor_containers(svg_root: etree._Element) -> list[etree._Element]:
    """Find all <g class="Floorplan ..."> containers (one per floor)."""
    containers = []
    for el in svg_root.iter():
        if el.tag is etree.Comment:
            continue
        if _local_name(el) != "g":
            continue
        if "Floorplan" in _class_tokens(el):
            containers.append(el)
    return containers


def _classify_floor_child(el: etree._Element) -> str | None:
    """Classify a direct child of a floor container into a semantic category.

    Returns "wall", "floor", "window", "door", or None.
    """
    if el.tag is etree.Comment or _is_hidden(el) or _local_name(el) != "g":
        return None

    elem_id = (el.get("id") or "").strip()
    cls_parts = _class_tokens(el)

    # Wall: id="Wall" or class contains "Wall"
    if elem_id == "Wall" or "Wall" in cls_parts:
        return "wall"

    # Floor/Space: class contains "Space", or id is "Space"
    if "Space" in cls_parts or elem_id.lower() == "space":
        return "floor"

    # Window: top-level Window (test SVGs / non-standard structures)
    if elem_id == "Window" or "Window" in cls_parts:
        return "window"

    # Door: top-level Door (test SVGs / non-standard structures)
    if elem_id == "Door" or "Door" in cls_parts:
        return "door"

    return None


def _get_window_children(wall_el: etree._Element) -> list[etree._Element]:
    """Return direct Window child elements of a wall element."""
    return [
        child for child in wall_el
        if child.tag is not etree.Comment
        and _local_name(child) == "g"
        and (child.get("id") or "").strip() == "Window"
    ]


def _get_door_children(wall_el: etree._Element) -> list[etree._Element]:
    """Return direct Door child elements of a wall element."""
    return [
        child for child in wall_el
        if child.tag is not etree.Comment
        and _local_name(child) == "g"
        and (child.get("id") or "").strip() == "Door"
    ]


def _find_threshold_polygons(door_el: etree._Element) -> list[etree._Element]:
    """Return the <polygon> elements nested under the Door's Threshold group(s)."""
    polygons: list[etree._Element] = []
    for child in door_el:
        if child.tag is etree.Comment or _local_name(child) != "g":
            continue
        elem_id = (child.get("id") or "").strip()
        if elem_id != "Threshold" and "Threshold" not in _class_tokens(child):
            continue
        for desc in child.iter():
            if desc.tag is etree.Comment:
                continue
            if _local_name(desc) == "polygon":
                polygons.append(desc)
    return polygons


def _parse_points(points_str: str) -> list[tuple[float, float]]:
    """Parse an SVG polygon "points" attribute into a list of (x, y) pairs."""
    tokens = points_str.replace(",", " ").split()
    coords = [float(t) for t in tokens]
    return list(zip(coords[0::2], coords[1::2]))


def _bbox_centerline(
    points: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Centerline segment along the long axis of *points*' axis-aligned bounding box.

    The threshold polygon is a rectangle spanning (door width) x (wall thickness); its
    long axis runs along the wall, which is the "wall-aligned" origin segment (spec_v005
    run3 §6). Short axis (wall thickness) collapses to the rectangle's midline.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    if (maxx - minx) >= (maxy - miny):
        y_mid = (miny + maxy) / 2
        return (minx, y_mid), (maxx, y_mid)
    x_mid = (minx + maxx) / 2
    return (x_mid, miny), (x_mid, maxy)


def _find_panel_paths(door_el: etree._Element) -> list[etree._Element]:
    """Return <path> elements nested under the Door's Panel group(s).

    Excludes "PanelArea" — a distinct class token used for a polygon, not a path.
    """
    paths: list[etree._Element] = []
    for child in door_el:
        if child.tag is etree.Comment or _local_name(child) != "g":
            continue
        elem_id = (child.get("id") or "").strip()
        if elem_id != "Panel" and "Panel" not in _class_tokens(child):
            continue
        for desc in child.iter():
            if desc.tag is etree.Comment:
                continue
            if _local_name(desc) == "path":
                paths.append(desc)
    return paths


# ---------------------------------------------------------------------------
# Door panel path splitting: M x,y q cx,cy ex,ey l dx,dy Z
#   q-curve  -> door_arc wedge (whole closed path, filled)
#   l-segment -> door_leaf line (stroked)
# ---------------------------------------------------------------------------

_PANEL_PATH_RE = re.compile(
    r"M\s*([-\d.]+)[,\s]+([-\d.]+)\s*"
    r"q\s*([-\d.]+)[,\s]+([-\d.]+)\s+([-\d.]+)[,\s]+([-\d.]+)\s*"
    r"l\s*([-\d.]+)[,\s]+([-\d.]+)\s*[zZ]",
)


def _split_panel_path(
    d: str,
) -> tuple[str | None, tuple[tuple[float, float], tuple[float, float]] | None]:
    """Parse a Door/Panel path "M x,y q cx,cy ex,ey l dx,dy Z".

    Returns:
        (wedge_d, leaf_endpoints) where:
          - wedge_d is the original *d* string, used directly as the door_arc fill geometry
            (it is already the closed quarter-circle wedge).
          - leaf_endpoints is ((arc_end_x, arc_end_y), (leaf_end_x, leaf_end_y)) in absolute
            coordinates, used to build the door_leaf stroke line.
        Returns (None, None) if *d* does not match the expected swing-door pattern.
    """
    match = _PANEL_PATH_RE.search(d)
    if not match:
        return None, None

    mx, my, _qcx, _qcy, qex, qey, ldx, ldy = (float(g) for g in match.groups())
    arc_end = (mx + qex, my + qey)
    leaf_end = (arc_end[0] + ldx, arc_end[1] + ldy)
    return d, (arc_end, leaf_end)


def _door_leaf_stroke_width(native_size: int) -> float:
    """Fixed stroke width (native px), identical for every door_leaf in every sample."""
    del native_size  # width is intentionally constant, not scaled by image resolution
    return _DOOR_STROKE_WIDTH_PX


def _door_origin_stroke_width(native_size: int) -> float:
    """Fixed stroke width (native px) — always equal to the door_leaf stroke width."""
    del native_size  # width is intentionally constant, not scaled by image resolution
    return _DOOR_STROKE_WIDTH_PX


# ---------------------------------------------------------------------------
# Element collection
# ---------------------------------------------------------------------------


def _collect_elements_by_category(
    svg_root: etree._Element, native_size: int,
) -> dict[str, list[etree._Element]]:
    """Return all semantic elements grouped by category.

    "door_origin", "door_arc", and "door_leaf" entries are synthetic <path> elements built
    from parsed Door/Threshold/Panel geometry (not deep copies of original elements).
    """
    walls: list[etree._Element] = []
    floors: list[etree._Element] = []
    windows: list[etree._Element] = []
    door_origins: list[etree._Element] = []
    door_arcs: list[etree._Element] = []
    door_leaves: list[etree._Element] = []

    leaf_stroke_width = _door_leaf_stroke_width(native_size)
    origin_stroke_width = _door_origin_stroke_width(native_size)

    def _process_door(door_el: etree._Element) -> None:
        for threshold_poly in _find_threshold_polygons(door_el):
            points = _parse_points(threshold_poly.get("points", ""))
            if len(points) < 2:
                continue
            (ox1, oy1), (ox2, oy2) = _bbox_centerline(points)
            origin = etree.Element("path")
            origin.set("d", f"M{ox1},{oy1} L{ox2},{oy2}")
            origin.set("fill", "none")
            origin.set("stroke", "#000000")
            origin.set("stroke-width", str(origin_stroke_width))
            door_origins.append(origin)

        for path_el in _find_panel_paths(door_el):
            d = path_el.get("d", "")
            wedge_d, leaf_endpoints = _split_panel_path(d)
            if wedge_d is None:
                logger.debug("Panel path did not match swing pattern, skipping arc/leaf: %s", d)
                continue

            wedge = etree.Element("path")
            wedge.set("d", wedge_d)
            wedge.set("fill", "#000000")
            wedge.set("stroke", "none")
            door_arcs.append(wedge)

            (ax, ay), (lx, ly) = leaf_endpoints
            leaf = etree.Element("path")
            leaf.set("d", f"M{ax},{ay} L{lx},{ly}")
            leaf.set("fill", "none")
            leaf.set("stroke", "#000000")
            leaf.set("stroke-width", str(leaf_stroke_width))
            door_leaves.append(leaf)

    containers = _find_floor_containers(svg_root)
    if not containers:
        # Fallback: treat root itself as the container (e.g. minimal test SVGs)
        containers = [svg_root]

    for container in containers:
        for child in container:
            category = _classify_floor_child(child)
            if category == "wall":
                walls.append(child)
                for win in _get_window_children(child):
                    windows.append(win)
                for door in _get_door_children(child):
                    _process_door(door)
            elif category == "floor":
                floors.append(child)
            elif category == "window":
                # Top-level Window (test SVGs or non-standard structures)
                windows.append(child)
            elif category == "door":
                # Top-level Door (test SVGs or non-standard structures)
                _process_door(child)

    logger.debug(
        "Collected: walls=%d  floors=%d  windows=%d  door_origin=%d  door_arc=%d  door_leaf=%d",
        len(walls), len(floors), len(windows), len(door_origins), len(door_arcs), len(door_leaves),
    )
    return {
        "wall": walls,
        "floor": floors,
        "window": windows,
        "door_origin": door_origins,
        "door_arc": door_arcs,
        "door_leaf": door_leaves,
    }


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

# Categories whose source elements use white/light fills in the SVG — must be forced black.
# "door_origin"/"door_arc"/"door_leaf" are synthetic elements already styled correctly;
# not included here.
_FORCE_BLACK_CATEGORIES = {"floor", "window"}

_ALL_CATEGORIES = ["floor", "wall", "window", "door_origin", "door_arc", "door_leaf"]


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

    elements_by_cat = _collect_elements_by_category(svg_root, native_size=max(width, height))

    category_masks: dict[str, np.ndarray] = {}
    missing_classes: list[str] = []

    for category in _ALL_CATEGORIES:
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

    for cat in APPLY_ORDER:
        if cat in masks and cat in DEBUG_COLORS:
            base[masks[cat] > 0] = DEBUG_COLORS[cat]

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
