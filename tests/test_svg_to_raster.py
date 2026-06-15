"""Tests for src/svg_to_raster.py (spec_v003, task01 enhanced)."""

import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.svg_to_raster import (
    OUTPUT_NAME,
    SVG_NAME,
    _count_non_white_pixels,
    _strip_display_none,
    convert_svg_to_png,
    normalize_svg_visibility,
    process_dataset,
)

# ---------------------------------------------------------------------------
# SVG fixtures
# ---------------------------------------------------------------------------

MINIMAL_SVG = """\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <rect width="10" height="10" fill="black"/>
</svg>
"""

# Two-floor plan: Floor-1 (left) always visible; Floor-2 (right) hidden.
TWO_FLOOR_SVG = """\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="200" height="100" viewBox="0 0 200 100">
  <defs/>
  <g id="Model" class="Model v1-1">
    <g class="Floor">
      <g id="Floor-1" class="Floorplan Floor-1">
        <rect x="10" y="10" width="80" height="80" fill="black" stroke="none"
              style="fill-opacity:1;stroke-opacity:1;stroke-width:0.2;"/>
      </g>
    </g>
    <g style="display: none;" class="Floor">
      <g id="Floor-2" class="Floorplan Floor-2">
        <rect x="110" y="10" width="80" height="80" fill="black" stroke="none"
              style="fill-opacity:1;stroke-opacity:1;stroke-width:0.2;"/>
      </g>
    </g>
  </g>
</svg>
"""

# SVG with FloorsCompose (should remain hidden after normalization).
FLOORS_COMPOSE_SVG = """\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="200" height="100" viewBox="0 0 200 100">
  <defs/>
  <g id="Model" class="Model v1-1">
    <g class="Floor">
      <g id="Floor-1" class="Floorplan Floor-1">
        <rect x="10" y="10" width="80" height="80" fill="black" stroke="none"/>
      </g>
    </g>
    <g style="display: none;" class="Floor">
      <g id="Floor-2" class="Floorplan Floor-2">
        <rect x="110" y="10" width="80" height="80" fill="black" stroke="none"/>
      </g>
    </g>
    <g class="FloorsCompose" style="display: none;">
      <g class="ComposeElement">
        <use xlink:href="#Floor-1"/>
      </g>
      <g transform="matrix(1,0,0,1,20,0)" class="ComposeElement">
        <use xlink:href="#Floor-2"/>
      </g>
    </g>
  </g>
</svg>
"""

# SVG with nested hidden UI elements that must stay hidden.
NESTED_HIDDEN_SVG = """\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <g id="Model" class="Model v1-1">
    <g class="Floor">
      <g id="Floor-1" class="Floorplan Floor-1">
        <g class="Space Living">
          <rect x="10" y="10" width="80" height="80" fill="white" stroke="black"/>
          <g class="Dimension">
            <g style="display: none;" class="Visual">
              <polygon points="45,0 0,0 5,5" fill="black"/>
            </g>
          </g>
        </g>
      </g>
    </g>
  </g>
</svg>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_png_ihdr(path: Path) -> dict:
    """Parse width, height, bit_depth, color_type from a PNG IHDR chunk."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "Not a valid PNG"
    ihdr_data = data[16:29]
    width, height, bit_depth, color_type = struct.unpack(">IIBB", ihdr_data[:10])
    return {"width": width, "height": height, "bit_depth": bit_depth, "color_type": color_type}


def _png_top_left_pixel_rgb(path: Path) -> tuple[int, int, int]:
    """Return the RGB value of the top-left pixel by decoding the first scanline."""
    import struct as _struct
    import zlib as _zlib

    data = path.read_bytes()
    pos = 8
    idat_payloads = []
    while pos < len(data):
        length = _struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        if chunk_type == b"IDAT":
            idat_payloads.append(payload)
        elif chunk_type == b"IEND":
            break
        pos += 12 + length

    raw = _zlib.decompress(b"".join(idat_payloads))
    filter_byte = raw[0]
    assert filter_byte in (0, 1), f"Unsupported filter byte {filter_byte}"
    r, g, b = raw[1], raw[2], raw[3]
    return r, g, b


# ---------------------------------------------------------------------------
# Unit tests — normalize_svg_visibility
# ---------------------------------------------------------------------------


def test_normalize_hidden_floor_unhidden():
    """Floor group with display:none must become visible after normalization."""
    normalized, changes = normalize_svg_visibility(TWO_FLOOR_SVG.encode())
    assert len(changes) == 1
    assert "Floor-2" in changes[0] or "Floor" in changes[0]
    assert b"display: none" not in normalized or b'class="FloorsCompose"' in normalized


def test_normalize_visible_floor_unchanged():
    """A Floor group that is already visible must not be touched."""
    # Single-floor SVG — nothing to unhide.
    svg = MINIMAL_SVG.encode()
    _, changes = normalize_svg_visibility(svg)
    assert changes == []


def test_normalize_floors_compose_stays_hidden():
    """FloorsCompose must remain hidden (its transforms would break mask alignment)."""
    normalized, _ = normalize_svg_visibility(FLOORS_COMPOSE_SVG.encode())
    # FloorsCompose style must still contain display:none
    import re
    # Find FloorsCompose in output
    assert b"FloorsCompose" in normalized
    # The FloorsCompose group must retain display:none
    match = re.search(
        rb'class="FloorsCompose"[^>]*style="([^"]*)"'
        rb'|style="([^"]*)"[^>]*class="FloorsCompose"',
        normalized,
    )
    assert match, "Could not find FloorsCompose element in normalized output"
    style_bytes = (match.group(1) or match.group(2)).decode()
    assert "display" in style_bytes and "none" in style_bytes, (
        f"FloorsCompose should remain hidden, got style='{style_bytes}'"
    )


def test_normalize_nested_hidden_elements_untouched():
    """Nested UI elements (Visual, DimensionMark) inside a visible Floor must stay hidden."""
    normalized, changes = normalize_svg_visibility(NESTED_HIDDEN_SVG.encode())
    # No Floor groups were hidden, so no changes expected
    assert changes == []
    # The nested Visual element must remain hidden
    assert b'class="Visual"' in normalized
    decoded = normalized.decode("utf-8", errors="replace")
    # Find the Visual element's style
    import re
    match = re.search(r'class="Visual"[^>]*style="([^"]*)"', decoded)
    if not match:
        match = re.search(r'style="([^"]*)"[^>]*class="Visual"', decoded)
    assert match, "Visual element not found in normalized SVG"
    assert "display" in match.group(1) and "none" in match.group(1)


def test_normalize_two_floor_produces_visible_floor2(tmp_path):
    """After normalization, the rendered PNG must include pixels from the Floor-2 region."""
    svg_path = tmp_path / SVG_NAME
    svg_path.write_bytes(TWO_FLOOR_SVG.encode())
    out_path = tmp_path / OUTPUT_NAME

    convert_svg_to_png(svg_path, out_path)

    img = np.array(Image.open(out_path).convert("RGB"))
    h, w = img.shape[:2]
    # Floor-2 rect occupies roughly x:110-190 in a 200px-wide image
    right_region = img[:, w // 2 :, :]
    non_white_right = int(np.any(right_region < 250, axis=-1).sum())
    assert non_white_right > 0, "Floor-2 geometry must be visible in the right half of the PNG"


def test_normalize_returns_changes_list():
    """convert_svg_to_png must return a list; non-empty for hidden-floor SVGs."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        svg_path = p / SVG_NAME
        svg_path.write_bytes(TWO_FLOOR_SVG.encode())
        out_path = p / OUTPUT_NAME
        changes = convert_svg_to_png(svg_path, out_path)

    assert isinstance(changes, list)
    assert len(changes) >= 1


# ---------------------------------------------------------------------------
# Unit tests — _strip_display_none
# ---------------------------------------------------------------------------


def test_strip_display_none_basic():
    assert _strip_display_none("display: none;") == ""


def test_strip_display_none_with_other_props():
    result = _strip_display_none("fill-opacity: 1; display: none; stroke-opacity: 1")
    assert "display" not in result
    assert "fill-opacity" in result


def test_strip_display_none_no_display():
    style = "fill-opacity: 1; stroke: black"
    assert _strip_display_none(style) == style


def test_strip_display_none_no_spaces():
    assert _strip_display_none("display:none") == ""


# ---------------------------------------------------------------------------
# Unit tests — _count_non_white_pixels
# ---------------------------------------------------------------------------


def test_count_non_white_pixels_all_white(tmp_path):
    path = tmp_path / "white.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(path)
    assert _count_non_white_pixels(path) == 0


def test_count_non_white_pixels_all_black(tmp_path):
    path = tmp_path / "black.png"
    Image.new("RGB", (10, 10), (0, 0, 0)).save(path)
    assert _count_non_white_pixels(path) == 100  # 10×10


def test_count_non_white_pixels_mixed(tmp_path):
    path = tmp_path / "mixed.png"
    img = Image.new("RGB", (10, 10), (255, 255, 255))
    # Make top-left 5×5 black
    pixels = img.load()
    for x in range(5):
        for y in range(5):
            pixels[x, y] = (0, 0, 0)
    img.save(path)
    assert _count_non_white_pixels(path) == 25


# ---------------------------------------------------------------------------
# Unit tests — convert_svg_to_png (existing contract preserved)
# ---------------------------------------------------------------------------


def test_convert_svg_to_png_creates_file(tmp_path):
    svg = tmp_path / SVG_NAME
    svg.write_text(MINIMAL_SVG)
    out = tmp_path / OUTPUT_NAME

    convert_svg_to_png(svg, out)

    assert out.exists(), "Output PNG was not created"
    assert out.stat().st_size > 0


def test_convert_svg_to_png_is_valid_png(tmp_path):
    svg = tmp_path / SVG_NAME
    svg.write_text(MINIMAL_SVG)
    out = tmp_path / OUTPUT_NAME

    convert_svg_to_png(svg, out)

    ihdr = _read_png_ihdr(out)
    assert ihdr["width"] == 10
    assert ihdr["height"] == 10


def test_convert_svg_to_png_white_background(tmp_path):
    transparent_svg = """\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
</svg>
"""
    svg = tmp_path / SVG_NAME
    svg.write_text(transparent_svg)
    out = tmp_path / OUTPUT_NAME

    convert_svg_to_png(svg, out)

    r, g, b = _png_top_left_pixel_rgb(out)
    assert (r, g, b) == (255, 255, 255), f"Expected white but got ({r},{g},{b})"


# ---------------------------------------------------------------------------
# Integration tests — process_dataset (existing contract preserved)
# ---------------------------------------------------------------------------


def test_process_dataset_converts_all(tmp_path):
    for name in ("a", "b", "c"):
        d = tmp_path / name
        d.mkdir()
        (d / SVG_NAME).write_text(MINIMAL_SVG)

    converted, skipped = process_dataset(tmp_path)

    assert converted == 3
    assert skipped == 0
    for name in ("a", "b", "c"):
        assert (tmp_path / name / OUTPUT_NAME).exists()


def test_process_dataset_skips_existing(tmp_path):
    d = tmp_path / "sample"
    d.mkdir()
    (d / SVG_NAME).write_text(MINIMAL_SVG)
    existing = d / OUTPUT_NAME
    existing.write_bytes(b"placeholder")

    converted, skipped = process_dataset(tmp_path, overwrite=False)

    assert converted == 0
    assert skipped == 1
    assert existing.read_bytes() == b"placeholder"


def test_process_dataset_overwrite(tmp_path):
    d = tmp_path / "sample"
    d.mkdir()
    (d / SVG_NAME).write_text(MINIMAL_SVG)
    existing = d / OUTPUT_NAME
    existing.write_bytes(b"placeholder")

    converted, skipped = process_dataset(tmp_path, overwrite=True)

    assert converted == 1
    assert skipped == 0
    assert existing.stat().st_size > len(b"placeholder")


def test_process_dataset_ignores_other_pngs(tmp_path):
    d = tmp_path / "sample"
    d.mkdir()
    (d / SVG_NAME).write_text(MINIMAL_SVG)
    other_png = d / "F1_original.png"
    other_png.write_bytes(b"untouched")

    process_dataset(tmp_path)

    assert other_png.read_bytes() == b"untouched"


def test_process_dataset_no_svgs(tmp_path):
    converted, skipped = process_dataset(tmp_path)
    assert converted == 0
    assert skipped == 0


def test_process_dataset_two_floor_converts_with_fix(tmp_path):
    """process_dataset must export a PNG that contains Floor-2 geometry."""
    d = tmp_path / "two_floor"
    d.mkdir()
    (d / SVG_NAME).write_bytes(TWO_FLOOR_SVG.encode())

    converted, skipped = process_dataset(tmp_path)

    assert converted == 1
    out = d / OUTPUT_NAME
    assert out.exists()
    img = np.array(Image.open(out).convert("RGB"))
    # Floor-2 is in the right half of the 200px-wide image
    right = img[:, img.shape[1] // 2 :, :]
    assert np.any(right < 250), "Floor-2 must be visible in the right half of the PNG"
