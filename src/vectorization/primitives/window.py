"""WindowPrimitive: a wall-hosted window rendered as a blue centerline segment."""

from __future__ import annotations

import math
from typing import Optional

from .base import BasePrimitive, ScaleInfo


class WindowPrimitive(BasePrimitive):
    """A linear wall-hosted window, styled like a wall centerline but blue."""

    COLOR = "#3355cc"

    def __init__(
        self,
        primitive_id: str,
        center: tuple[float, float],
        width: float,
        orientation_angle: float = 0.0,
        thickness: float = 8.0,
        host_wall_id: Optional[str] = None,
        confidence: float = 1.0,
        scale_info: Optional[ScaleInfo] = None,
    ) -> None:
        super().__init__(primitive_id, confidence, scale_info)
        self.center = center
        self.width = width
        self.orientation_angle = orientation_angle
        self.thickness = thickness
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
        return (
            f'<line id="{self.primitive_id}" data-type="window" '
            f'x1="{s[0]:.2f}" y1="{s[1]:.2f}" '
            f'x2="{e[0]:.2f}" y2="{e[1]:.2f}" '
            f'stroke="{self.COLOR}" stroke-width="{self.thickness:.2f}" '
            f'stroke-linecap="square" />'
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
