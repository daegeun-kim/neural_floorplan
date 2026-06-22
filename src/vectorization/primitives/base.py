"""Abstract base class for all CAD-like component primitives."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

from .scale import ScaleInfo

__all__ = ["BasePrimitive", "ScaleInfo"]


class BasePrimitive(ABC):
    """Parametric CAD primitive that can generate SVG geometry and report its bounds.

    Shared fields per spec_v007 SS7: ``kind`` identifies the primitive type for
    debugging/metrics, ``source_class_ids`` records which CNN class IDs the
    evidence came from, and ``source_evidence_bbox_px``/``source_evidence_area_px``
    keep the raw pixel-space evidence available for debugging even after the
    primitive has been converted to clean (possibly metric) geometry.
    """

    def __init__(
        self,
        primitive_id: str,
        confidence: float = 1.0,
        scale_info: Optional[ScaleInfo] = None,
        kind: Optional[str] = None,
        source_class_ids: Optional[list[int]] = None,
        source_evidence_bbox_px: Optional[tuple[float, float, float, float]] = None,
        source_evidence_area_px: Optional[float] = None,
    ) -> None:
        self.primitive_id = primitive_id
        self.confidence = confidence
        self.scale_info: ScaleInfo = scale_info or ScaleInfo()
        self.kind = kind or self.__class__.__name__
        self.source_class_ids = source_class_ids or []
        self.source_evidence_bbox_px = source_evidence_bbox_px
        self.source_evidence_area_px = source_evidence_area_px

    @abstractmethod
    def to_svg(self) -> str:
        """Return SVG element string(s) representing this primitive."""

    @abstractmethod
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) bounding box."""

    def transform(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        angle_deg: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
    ) -> None:
        """Apply translate/rotate/scale in-place. Subclasses override as needed."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement transform()"
        )

    @staticmethod
    def _rotate_point(
        x: float, y: float, cx: float, cy: float, angle_rad: float
    ) -> tuple[float, float]:
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        dx = x - cx
        dy = y - cy
        return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a

    @staticmethod
    def _svg_color_for_confidence(confidence: float) -> str:
        if confidence >= 0.8:
            return "#222222"
        if confidence >= 0.5:
            return "#888888"
        return "#cccccc"
