"""Shared preprocessing for Phase 4: crop + pad + scale to 512x512 canvas.

Both Raster-to-Graph and 7-class segmentation must run on the IDENTICAL
preprocessed image (spec_v008 §2).  This module wraps the R2G preprocessing
function and saves the transform manifest so all downstream coordinates are
unambiguously in `preprocessed_512` space.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PIL import Image

# The R2G preprocessing function lives in external/raster_to_graph.
# Locate the project root and add external dir to sys.path if needed.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EXTERNAL_DIR = _PROJECT_ROOT / "external" / "raster_to_graph"

# Mirrors CANVAS_SIZE in external/raster_to_graph/run_inference_generous_phase4.py.
# Kept here so the manifest transform can be derived without importing the
# vendored module at definition time.
CANVAS_SIZE = 512


def _ensure_r2g_importable() -> None:
    ext = str(_EXTERNAL_DIR)
    if ext not in sys.path:
        sys.path.insert(0, ext)


def preprocess_image(
    image_path: str | Path,
) -> tuple[Image.Image, dict[str, Any]]:
    """Preprocess an input floor plan image to the 512×512 R2G canvas.

    Returns:
        (canvas_pil, manifest)
        - canvas_pil: 512×512 RGB PIL image (the shared preprocessing result)
        - manifest:   dict matching spec_v008 §2 JSON schema
    """
    _ensure_r2g_importable()
    from run_inference_generous_phase4 import (
        preprocess_crop512_margin20_truepad,  # type: ignore[import]
    )

    image_path = Path(image_path).resolve()
    base_pil, preproc_metrics = preprocess_crop512_margin20_truepad(image_path)

    # The upstream metrics dict reports only the content bbox and the margin
    # fraction, so the forward transform is re-derived here from the same
    # arithmetic preprocess_crop512_margin20_truepad() performs. Recording it
    # explicitly is what lets a consumer map original-raster coordinates onto
    # the 512x512 canvas (see specs/spec_v006_evaluation.md §3); without it the
    # manifest would silently claim an identity transform.
    with Image.open(image_path) as _src:
        source_width, source_height = _src.size

    margin = float(preproc_metrics.get("standardized_margin", 0.20))
    x0, y0, x1, y1 = (int(v) for v in preproc_metrics.get("content_bbox_original", [0, 0, 0, 0]))
    content_w = max(x1 - x0, 1)
    content_h = max(y1 - y0, 1)
    pad_x = max(int(content_w * margin), 1)
    pad_y = max(int(content_h * margin), 1)
    padded_width = content_w + 2 * pad_x
    padded_height = content_h + 2 * pad_y
    scale_to_512 = CANVAS_SIZE / max(padded_width, padded_height)
    scaled_w = int(padded_width * scale_to_512)
    scaled_h = int(padded_height * scale_to_512)
    canvas_offset_x = (CANVAS_SIZE - scaled_w) // 2
    canvas_offset_y = (CANVAS_SIZE - scaled_h) // 2

    manifest: dict[str, Any] = {
        "source_image": str(image_path),
        "source_width": source_width,
        "source_height": source_height,
        "content_bbox_original": preproc_metrics.get("content_bbox_original", [0, 0, 0, 0]),
        "padding_fraction": margin,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "padded_width": padded_width,
        "padded_height": padded_height,
        "scale_to_512": scale_to_512,
        "canvas_offset_x": canvas_offset_x,
        "canvas_offset_y": canvas_offset_y,
        "coordinate_space": "preprocessed_512",
        "source_variant": preproc_metrics.get("source_variant", "crop512_margin20_truepad"),
        # pass through extra R2G metrics for debug
        "content_touches_edge": preproc_metrics.get("content_touches_edge", False),
        "final_canvas_margins_px": preproc_metrics.get("final_canvas_margins_px", {}),
        "content_bbox_after_preprocess": preproc_metrics.get("content_bbox_after_preprocess", []),
    }
    return base_pil, manifest
