"""WindowPrimitive: a wall-hosted window rendered as a blue centerline segment."""

from __future__ import annotations

import math
from typing import Optional

from ..wall_geometry import buffer_segment_polygon_svg
from .base import BasePrimitive, ScaleInfo


class WindowPrimitive(BasePrimitive):
    """A wall-hosted window, rendered as a filled blue polygon (task08), with
    a total thickness of its own - 100mm vs the host wall's 200mm (task09),
    not the same thickness as the wall it replaces.

    ``width`` is always the pixel-space width used for geometry; ``width_mm``
    is set (and possibly snapped to a common window module) only when scale
    is resolved/estimated with sufficient confidence - see
    primitives.scale.snap_to_module_mm.
    """

    # Matches DEBUG_COLORS["window"] in src/generate_semantic_masks.py.
    COLOR = "#3c78dc"

    def __init__(
        self,
        primitive_id: str,
        center: tuple[float, float],
        width: float,
        orientation_angle: float = 0.0,
        thickness: float = 8.0,
        width_mm: Optional[float] = None,
        host_wall_id: Optional[str] = None,
        confidence: float = 1.0,
        scale_info: Optional[ScaleInfo] = None,
        **base_kwargs,
    ) -> None:
        super().__init__(primitive_id, confidence, scale_info, **base_kwargs)
        self.center = center
        self.width = width
        self.orientation_angle = orientation_angle
        self.thickness = thickness
        self.width_mm = width_mm
        self.host_wall_id = host_wall_id

    def _endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        half = self.width / 2.0
        a = math.radians(self.orientation_angle)
        cx, cy = self.center
        return (
            (cx - half * math.cos(a), cy - half * math.sin(a)),
            (cx + half * math.cos(a), cy + half * math.sin(a)),
        )

    def to_svg(self) -> str:
        s, e = self._endpoints()
        return buffer_segment_polygon_svg(
            s, e, self.thickness / 2.0, self.COLOR,
            extra_attrs=f'id="{self.primitive_id}" data-type="window"',
        )

    def bounds(self) -> tuple[float, float, float, float]:
        s, e = self._endpoints()
        half_t = self.thickness / 2.0
        return (
            min(s[0], e[0]) - half_t,
            min(s[1], e[1]) - half_t,
            max(s[0], e[0]) + half_t,
            max(s[1], e[1]) + half_t,
        )

    def transform(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        angle_deg: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
    ) -> None:
        self.center = (self.center[0] * sx + dx, self.center[1] * sy + dy)
        self.width *= sx
        self.orientation_angle += angle_deg
