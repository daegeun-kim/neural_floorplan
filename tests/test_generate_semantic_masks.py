"""Tests for src/generate_semantic_masks.py (spec_v003)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.generate_semantic_masks import (
    CLASS_IDS,
    MASKS_DIR,
    SVG_NAME,
    _classify_floor_child,
    generate_masks,
    process_dataset,
)
from lxml import etree


def _make_g(elem_id: str = "", cls: str = "") -> etree._Element:
    el = etree.Element("g")
    if elem_id:
        el.set("id", elem_id)
    if cls:
        el.set("class", cls)
    return el

# ---------------------------------------------------------------------------
# Synthetic SVGs
# ---------------------------------------------------------------------------

# Simple 100x100 SVG with wall, opening, room, and icon groups.
FULL_SVG = """\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
  <g id="Wall">
    <rect x="0" y="0" width="100" height="10" fill="black"/>
  </g>
  <g id="Door">
    <rect x="40" y="0" width="20" height="10" fill="gray"/>
  </g>
  <g id="Space">
    <rect x="5" y="15" width="90" height="80" fill="lightblue"/>
  </g>
  <g id="FixedFurniture">
    <rect x="10" y="20" width="15" height="15" fill="green"/>
  </g>
</svg>
"""

# SVG with only a wall — no room, so suspicious
WALL_ONLY_SVG = """\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" viewBox="0 0 50 50">
  <g id="Wall">
    <rect x="0" y="0" width="50" height="5" fill="black"/>
  </g>
</svg>
"""

# SVG with no semantic groups — no wall → suspicious
EMPTY_SVG = """\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" viewBox="0 0 50 50">
  <rect x="0" y="0" width="50" height="50" fill="white"/>
</svg>
"""


def _write_svg(directory: Path, content: str) -> Path:
    path = directory / SVG_NAME
    path.write_text(content, encoding="utf-8")
    return path


def _masks_dir(sample_dir: Path) -> Path:
    return sample_dir / MASKS_DIR


# ---------------------------------------------------------------------------
# _classify_floor_child unit tests
# ---------------------------------------------------------------------------


def test_classify_wall_by_id():
    assert _classify_floor_child(_make_g(elem_id="Wall")) == "wall"


def test_classify_wall_by_class():
    assert _classify_floor_child(_make_g(cls="Wall External")) == "wall"


def test_classify_room_by_class():
    assert _classify_floor_child(_make_g(cls="Space Kitchen")) == "room"


def test_classify_room_by_id():
    assert _classify_floor_child(_make_g(elem_id="Space")) == "room"


def test_classify_opening_door():
    assert _classify_floor_child(_make_g(elem_id="Door")) == "opening"


def test_classify_opening_window():
    assert _classify_floor_child(_make_g(elem_id="Window")) == "opening"


def test_classify_icon_fixed_furniture_set():
    assert _classify_floor_child(_make_g(elem_id="FixedFurnitureSet")) == "icon"


def test_classify_icon_by_class():
    assert _classify_floor_child(_make_g(cls="FixedFurnitureSet")) == "icon"


def test_classify_unknown_returns_none():
    assert _classify_floor_child(_make_g(elem_id="SomethingElse")) is None


def test_classify_hidden_element_returns_none():
    el = _make_g(elem_id="Wall")
    el.set("style", "display: none;")
    assert _classify_floor_child(el) is None


# ---------------------------------------------------------------------------
# generate_masks — happy path
# ---------------------------------------------------------------------------


def test_creates_all_mask_files(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path, debug_overlays=False)

    md = _masks_dir(tmp_path)
    for fname in ("wall_mask.png", "opening_mask.png", "room_mask.png", "icon_mask.png",
                  "semantic_class_map.png", "mask_metadata.json"):
        assert (md / fname).exists(), f"Missing: {fname}"


def test_semantic_class_map_valid_ids(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    class_map = np.array(Image.open(_masks_dir(tmp_path) / "semantic_class_map.png"))
    valid_ids = set(CLASS_IDS.values())
    assert set(np.unique(class_map)).issubset(valid_ids), "Unexpected class IDs in semantic map"


def test_masks_are_binary(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    for fname in ("wall_mask.png", "opening_mask.png", "room_mask.png", "icon_mask.png"):
        arr = np.array(Image.open(_masks_dir(tmp_path) / fname))
        unique = set(np.unique(arr))
        assert unique.issubset({0, 255}), f"{fname} is not binary: found values {unique}"


def test_all_masks_same_dimensions(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    md = _masks_dir(tmp_path)
    sizes = {}
    for fname in ("wall_mask.png", "opening_mask.png", "room_mask.png",
                  "icon_mask.png", "semantic_class_map.png"):
        with Image.open(md / fname) as img:
            sizes[fname] = img.size

    values = list(sizes.values())
    assert all(v == values[0] for v in values), f"Dimension mismatch: {sizes}"


def test_wall_pixels_present(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    wall_mask = np.array(Image.open(_masks_dir(tmp_path) / "wall_mask.png"))
    assert np.any(wall_mask > 0), "Wall mask is empty"


def test_class_map_contains_room_class(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    class_map = np.array(Image.open(_masks_dir(tmp_path) / "semantic_class_map.png"))
    assert CLASS_IDS["room"] in np.unique(class_map)


def test_metadata_json_created(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    meta_path = _masks_dir(tmp_path) / "mask_metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert "width" in meta
    assert "height" in meta
    assert "class_pixel_counts" in meta
    assert "missing_classes" in meta
    assert "status" in meta


def test_metadata_correct_dimensions(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    meta = json.loads((_masks_dir(tmp_path) / "mask_metadata.json").read_text())
    assert meta["width"] == 100
    assert meta["height"] == 100


# ---------------------------------------------------------------------------
# Priority / overlap
# ---------------------------------------------------------------------------


def test_wall_wins_over_opening_in_overlap_region(tmp_path):
    """The wall group and door group share the same row (y=0..10).
    After priority merge, those pixels should be class 'wall' (id=1), not 'opening' (id=2).
    """
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    class_map = np.array(Image.open(_masks_dir(tmp_path) / "semantic_class_map.png"))
    # Wall region is the top 10 rows; opening pixels are a subset of that row.
    top_row = class_map[0, :]
    # All non-background pixels in the top row should be wall (id=1).
    non_bg = top_row[top_row != CLASS_IDS["background"]]
    assert np.all(non_bg == CLASS_IDS["wall"]), (
        f"Expected wall class in top row but found: {np.unique(top_row)}"
    )


# ---------------------------------------------------------------------------
# Skip / overwrite
# ---------------------------------------------------------------------------


def test_skip_when_outputs_exist(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    result_first = generate_masks(tmp_path, overwrite=False)
    assert result_first["status"] != "skipped"

    result_second = generate_masks(tmp_path, overwrite=False)
    assert result_second["status"] == "skipped"


def test_overwrite_regenerates(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    # Corrupt the semantic map
    corrupt_path = _masks_dir(tmp_path) / "semantic_class_map.png"
    corrupt_path.write_bytes(b"not a real png")

    generate_masks(tmp_path, overwrite=True)

    # Should be a valid PNG again
    with Image.open(corrupt_path) as img:
        assert img.size == (100, 100)


# ---------------------------------------------------------------------------
# Missing SVG
# ---------------------------------------------------------------------------


def test_missing_svg_returns_missing_svg_status(tmp_path):
    result = generate_masks(tmp_path)
    assert result["status"] == "missing_svg"


# ---------------------------------------------------------------------------
# Suspicious sample
# ---------------------------------------------------------------------------


def test_suspicious_when_no_wall(tmp_path):
    _write_svg(tmp_path, EMPTY_SVG)
    result = generate_masks(tmp_path)
    assert result["status"] == "suspicious"


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------


def test_debug_overlay_created(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path, debug_overlays=True)

    overlay = _masks_dir(tmp_path) / "debug_overlay.png"
    assert overlay.exists()
    with Image.open(overlay) as img:
        assert img.mode == "RGB"


def test_no_debug_overlay_without_flag(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path, debug_overlays=False)

    assert not (_masks_dir(tmp_path) / "debug_overlay.png").exists()


# ---------------------------------------------------------------------------
# process_dataset integration
# ---------------------------------------------------------------------------


def test_process_dataset_multiple_samples(tmp_path):
    for name in ("a", "b", "c"):
        d = tmp_path / name
        d.mkdir()
        _write_svg(d, FULL_SVG)

    counts = process_dataset(tmp_path)

    assert counts["processed"] == 3
    assert counts["skipped_existing"] == 0
    assert counts["missing_svg"] == 0
    assert counts["failed"] == 0


def test_process_dataset_summary_json(tmp_path):
    (tmp_path / "s1").mkdir()
    _write_svg(tmp_path / "s1", FULL_SVG)

    process_dataset(tmp_path)

    summary = json.loads((tmp_path / "semantic_mask_generation_summary.json").read_text())
    assert "processed" in summary
    assert "failed" in summary


def test_process_dataset_skip_missing_svg(tmp_path):
    (tmp_path / "no_svg").mkdir()  # no model.svg here

    counts = process_dataset(tmp_path)
    assert counts["missing_svg"] == 1
    assert counts["processed"] == 0


def test_process_dataset_skips_existing(tmp_path):
    d = tmp_path / "s1"
    d.mkdir()
    _write_svg(d, FULL_SVG)
    process_dataset(tmp_path)  # first run

    counts = process_dataset(tmp_path, overwrite=False)  # second run
    assert counts["skipped_existing"] == 1
    assert counts["processed"] == 0


def test_process_dataset_overwrite(tmp_path):
    d = tmp_path / "s1"
    d.mkdir()
    _write_svg(d, FULL_SVG)
    process_dataset(tmp_path)  # first run

    counts = process_dataset(tmp_path, overwrite=True)
    assert counts["processed"] == 1
    assert counts["skipped_existing"] == 0
