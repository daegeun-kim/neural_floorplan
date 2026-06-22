"""Tests for src/generate_semantic_masks.py (spec_v005 run3 — seven-class door subclasses)."""

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
    _split_panel_path,
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

# 100x100 canvas: a Wall strip (y=0..10) containing a Window and a swing Door
# (with Threshold + Panel>path evidence), plus a floor Space below the wall.
FULL_SVG = """\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
  <g id="Wall" fill="#000000" stroke="#000000">
    <polygon points="0,0 100,0 100,10 0,10"/>
    <g id="Window" fill="#f0f0ff" stroke="#000000" class="Window Regular">
      <polygon points="10,0 10,10 20,10 20,0"/>
      <g id="Glass" class="Glass"><polygon points="10,0 10,10 20,10 20,0"/></g>
      <g id="Panel" class="Panel"><line x1="15" x2="15" y1="0" y2="10"/></g>
    </g>
    <g id="Door" fill="#ffffff" stroke="#000000" class="Door Swing Beside">
      <polygon points="40,0 40,10 60,10 60,0"/>
      <g id="Threshold" class="Threshold">
        <polygon points="40,0 40,10 60,10 60,0"/>
      </g>
      <g id="Panel" fill="none" class="Panel Left Positive">
        <g id="PanelArea" fill="none" stroke="none" class="PanelArea">
          <polygon points="40,0 40,10 60,10 60,0"/>
        </g>
        <path d="M60,10 q10,0 10,-10 l-10,0 Z"/>
      </g>
    </g>
  </g>
  <g id="space-uuid" fill="#ffffff" stroke="#ffffff" class="Space Kitchen">
    <polygon points="5,15 95,15 95,95 5,95"/>
  </g>
</svg>
"""

# SVG with only a wall — no floor, so suspicious
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


_MASK_FILENAMES = (
    "floor_mask.png", "wall_mask.png", "window_mask.png",
    "door_origin_mask.png", "door_arc_mask.png", "door_leaf_mask.png",
)


# ---------------------------------------------------------------------------
# _classify_floor_child unit tests
# ---------------------------------------------------------------------------


def test_classify_wall_by_id():
    assert _classify_floor_child(_make_g(elem_id="Wall")) == "wall"


def test_classify_wall_by_class():
    assert _classify_floor_child(_make_g(cls="Wall External")) == "wall"


def test_classify_floor_by_class():
    assert _classify_floor_child(_make_g(cls="Space Kitchen")) == "floor"


def test_classify_floor_by_id():
    assert _classify_floor_child(_make_g(elem_id="Space")) == "floor"


def test_classify_window_top_level():
    assert _classify_floor_child(_make_g(elem_id="Window")) == "window"


def test_classify_door_top_level():
    assert _classify_floor_child(_make_g(elem_id="Door")) == "door"


def test_classify_unknown_returns_none():
    assert _classify_floor_child(_make_g(elem_id="SomethingElse")) is None


def test_classify_hidden_element_returns_none():
    el = _make_g(elem_id="Wall")
    el.set("style", "display: none;")
    assert _classify_floor_child(el) is None


# ---------------------------------------------------------------------------
# _split_panel_path unit tests
# ---------------------------------------------------------------------------


def test_split_panel_path_arc_and_leaf_endpoints():
    wedge_d, endpoints = _split_panel_path("M60,10 q10,0 10,-10 l-10,0 Z")
    assert wedge_d == "M60,10 q10,0 10,-10 l-10,0 Z"
    arc_end, leaf_end = endpoints
    assert arc_end == pytest.approx((70.0, 0.0))
    assert leaf_end == pytest.approx((60.0, 0.0))


def test_split_panel_path_no_match_returns_none():
    wedge_d, endpoints = _split_panel_path("M0,0 L10,10")
    assert wedge_d is None
    assert endpoints is None


# ---------------------------------------------------------------------------
# generate_masks — happy path
# ---------------------------------------------------------------------------


def test_creates_all_mask_files(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path, debug_overlays=False)

    md = _masks_dir(tmp_path)
    for fname in (*_MASK_FILENAMES, "semantic_class_map.png", "mask_metadata.json"):
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

    for fname in _MASK_FILENAMES:
        arr = np.array(Image.open(_masks_dir(tmp_path) / fname))
        unique = set(np.unique(arr))
        assert unique.issubset({0, 255}), f"{fname} is not binary: found values {unique}"


def test_all_masks_same_dimensions(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    md = _masks_dir(tmp_path)
    sizes = {}
    for fname in (*_MASK_FILENAMES, "semantic_class_map.png"):
        with Image.open(md / fname) as img:
            sizes[fname] = img.size

    values = list(sizes.values())
    assert all(v == values[0] for v in values), f"Dimension mismatch: {sizes}"


def test_wall_pixels_present(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    wall_mask = np.array(Image.open(_masks_dir(tmp_path) / "wall_mask.png"))
    assert np.any(wall_mask > 0), "Wall mask is empty"


def test_class_map_contains_floor_class(tmp_path):
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    class_map = np.array(Image.open(_masks_dir(tmp_path) / "semantic_class_map.png"))
    assert CLASS_IDS["floor"] in np.unique(class_map)


def test_door_subclasses_present(tmp_path):
    """door_origin, door_arc, and door_leaf should all have nonzero evidence."""
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    class_map = np.array(Image.open(_masks_dir(tmp_path) / "semantic_class_map.png"))
    unique = set(np.unique(class_map).tolist())
    assert CLASS_IDS["door_origin"] in unique
    assert CLASS_IDS["door_arc"] in unique
    assert CLASS_IDS["door_leaf"] in unique


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


def test_window_excluded_from_wall_class(tmp_path):
    """The Window footprint must not be labeled wall in the final semantic map."""
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    class_map = np.array(Image.open(_masks_dir(tmp_path) / "semantic_class_map.png"))
    window_col = class_map[5, 15]  # inside the 10..20 window span, mid-wall row
    assert window_col != CLASS_IDS["wall"]


def test_door_leaf_wins_over_door_arc_in_overlap_region(tmp_path):
    """door_leaf is rasterized after door_arc — at their shared boundary pixels,
    the semantic map must show door_leaf, not door_arc (spec_v005 run3 §6/§11)."""
    _write_svg(tmp_path, FULL_SVG)
    generate_masks(tmp_path)

    door_arc_mask = np.array(Image.open(_masks_dir(tmp_path) / "door_arc_mask.png"))
    door_leaf_mask = np.array(Image.open(_masks_dir(tmp_path) / "door_leaf_mask.png"))
    class_map = np.array(Image.open(_masks_dir(tmp_path) / "semantic_class_map.png"))

    overlap = (door_arc_mask > 0) & (door_leaf_mask > 0)
    assert np.any(overlap), "Expected door_arc and door_leaf to overlap in this fixture"
    assert np.all(class_map[overlap] == CLASS_IDS["door_leaf"]), (
        "Overlapping door_arc/door_leaf pixels must resolve to door_leaf"
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
