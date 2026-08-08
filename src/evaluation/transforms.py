"""Coordinate mapping between the original raster and the 512x512 canvas.

spec_v006 §3: all L2/L3 matching happens in the preprocessed canvas frame,
because that is the frame the Phase 4 pipeline predicts in. Reference geometry
from ``masks/wall_graph.json`` is in the original raster frame and must be
mapped forward using the *same* transform the prediction used.
"""

from __future__ import annotations

from dataclasses import dataclass

CANVAS_SIZE = 512


@dataclass(frozen=True)
class CanvasTransform:
    """Forward map: original raster pixel -> 512x512 canvas pixel.

    The map is crop-to-content, then pad, then uniform scale, then centre:

        canvas = (original - bbox_origin + pad) * scale + canvas_offset
    """

    bbox_x0: float
    bbox_y0: float
    pad_x: float
    pad_y: float
    scale: float
    offset_x: float
    offset_y: float

    @classmethod
    def from_manifest(cls, manifest: dict) -> CanvasTransform:
        """Build from a Phase 4 preprocessing manifest."""
        bbox = manifest.get("content_bbox_original") or [0, 0, 0, 0]
        return cls(
            bbox_x0=float(bbox[0]),
            bbox_y0=float(bbox[1]),
            pad_x=float(manifest["pad_x"]),
            pad_y=float(manifest["pad_y"]),
            scale=float(manifest["scale_to_512"]),
            offset_x=float(manifest["canvas_offset_x"]),
            offset_y=float(manifest["canvas_offset_y"]),
        )

    @classmethod
    def derive(
        cls,
        content_bbox: tuple[float, float, float, float],
        margin: float = 0.20,
    ) -> CanvasTransform:
        """Recompute the transform from a content bbox alone.

        Mirrors ``preprocess_crop512_margin20_truepad``. Integer truncation is
        reproduced exactly (``int()``, ``//``) so this agrees with the pipeline
        rather than merely approximating it.
        """
        x0, y0, x1, y1 = content_bbox
        content_w = max(int(x1) - int(x0), 1)
        content_h = max(int(y1) - int(y0), 1)
        pad_x = max(int(content_w * margin), 1)
        pad_y = max(int(content_h * margin), 1)
        padded_w = content_w + 2 * pad_x
        padded_h = content_h + 2 * pad_y
        scale = CANVAS_SIZE / max(padded_w, padded_h)
        scaled_w = int(padded_w * scale)
        scaled_h = int(padded_h * scale)
        return cls(
            bbox_x0=float(x0),
            bbox_y0=float(y0),
            pad_x=float(pad_x),
            pad_y=float(pad_y),
            scale=float(scale),
            offset_x=float((CANVAS_SIZE - scaled_w) // 2),
            offset_y=float((CANVAS_SIZE - scaled_h) // 2),
        )

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.bbox_x0 + self.pad_x) * self.scale + self.offset_x,
            (y - self.bbox_y0 + self.pad_y) * self.scale + self.offset_y,
        )

    def apply_length(self, length: float) -> float:
        """Scale a scalar length (an opening width) into canvas units."""
        return length * self.scale

    def as_dict(self) -> dict[str, float]:
        return {
            "bbox_x0": self.bbox_x0,
            "bbox_y0": self.bbox_y0,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "scale": self.scale,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
        }


def identity() -> CanvasTransform:
    """A no-op transform, for geometry already in canvas space."""
    return CanvasTransform(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
