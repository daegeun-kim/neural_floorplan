"""Tests for src/svg_to_raster.py (spec_v003)."""

import struct
import zlib
from pathlib import Path

import pytest

from src.svg_to_raster import OUTPUT_NAME, SVG_NAME, convert_svg_to_png, process_dataset

# Minimal valid SVG with a 10x10 white rect
MINIMAL_SVG = """\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <rect width="10" height="10" fill="black"/>
</svg>
"""


def _read_png_ihdr(path: Path) -> dict:
    """Parse width, height, bit_depth, color_type from a PNG IHDR chunk."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "Not a valid PNG"
    # IHDR is always the first chunk, starts at byte 8
    ihdr_data = data[16:29]  # 13 bytes of IHDR payload
    width, height, bit_depth, color_type = struct.unpack(">IIBB", ihdr_data[:10])
    return {"width": width, "height": height, "bit_depth": bit_depth, "color_type": color_type}


def _png_top_left_pixel_rgb(path: Path) -> tuple[int, int, int]:
    """Return the RGB value of the top-left pixel by decoding the first scanline.

    Handles PNG filter types 0 (None) and 1 (Sub). For the first pixel, Sub
    filter reduces to None because the left-neighbour is out-of-bounds (treated
    as 0), so the first bpp bytes equal the filtered bytes directly.
    Assumes 8-bit RGB or RGBA encoding.
    """
    import zlib as _zlib

    data = path.read_bytes()
    # Walk chunks to find IDAT
    pos = 8
    idat_payloads = []
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        if chunk_type == b"IDAT":
            idat_payloads.append(payload)
        elif chunk_type == b"IEND":
            break
        pos += 12 + length

    raw = _zlib.decompress(b"".join(idat_payloads))
    # Each scanline starts with a filter byte; pixel 0 of row 0 follows immediately
    # Assumes RGB or RGBA (color_type 2 or 6), 8-bit depth
    filter_byte = raw[0]
    assert filter_byte in (0, 1), f"Unsupported filter byte {filter_byte} in test helper"
    # For filter 0 (None) and filter 1 (Sub), the first pixel bytes are identical:
    # Sub filter for x<bpp: recon(x) = filt(x) + recon(x-bpp) = filt(x) + 0
    r, g, b = raw[1], raw[2], raw[3]
    return r, g, b


# ── unit tests ──────────────────────────────────────────────────────────────


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
    """SVG has a black rect; white background should be composited below it.
    The spec requires white background, so we check with an SVG whose content
    does not cover the full canvas — but since our minimal SVG fills it with black,
    we instead verify the background is applied by using a transparent SVG."""
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
    assert (r, g, b) == (255, 255, 255), f"Expected white (255,255,255) but got ({r},{g},{b})"


# ── integration tests ────────────────────────────────────────────────────────


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
    # Placeholder must not be overwritten
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
    """Other PNG files in the folder must not be touched."""
    d = tmp_path / "sample"
    d.mkdir()
    (d / SVG_NAME).write_text(MINIMAL_SVG)
    other_png = d / "F1_original.png"
    other_png.write_bytes(b"untouched")

    process_dataset(tmp_path)

    assert other_png.read_bytes() == b"untouched"


def test_process_dataset_no_svgs(tmp_path):
    """Directory with no SVG files should return (0, 0) without error."""
    converted, skipped = process_dataset(tmp_path)
    assert converted == 0
    assert skipped == 0
