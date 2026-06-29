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
    from run_inference_generous_phase4 import preprocess_crop512_margin20_truepad  # type: ignore[import]

    image_path = Path(image_path).resolve()
    base_pil, preproc_metrics = preprocess_crop512_margin20_truepad(image_path)

    manifest: dict[str, Any] = {
        "source_image": str(image_path),
        "source_width": preproc_metrics.get("source_width", 0),
        "source_height": preproc_metrics.get("source_height", 0),
        "content_bbox_original": preproc_metrics.get("content_bbox_original", [0, 0, 0, 0]),
        "padding_fraction": preproc_metrics.get("standardized_margin", 0.20),
        "padded_width": preproc_metrics.get("padded_width", 0),
        "padded_height": preproc_metrics.get("padded_height", 0),
        "scale_to_512": preproc_metrics.get("scale_to_512", 1.0),
        "canvas_offset_x": preproc_metrics.get("canvas_offset_x", 0),
        "canvas_offset_y": preproc_metrics.get("canvas_offset_y", 0),
        "coordinate_space": "preprocessed_512",
        "source_variant": preproc_metrics.get("source_variant", "crop512_margin20_truepad"),
        # pass through extra R2G metrics for debug
        "content_touches_edge": preproc_metrics.get("content_touches_edge", False),
        "final_canvas_margins_px": preproc_metrics.get("final_canvas_margins_px", {}),
        "content_bbox_after_preprocess": preproc_metrics.get("content_bbox_after_preprocess", []),
    }
    return base_pil, manifest
