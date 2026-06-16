"""WindowPrimitive: a generic wall-hosted window symbol."""

from __future__ import annotations

import math
from typing import Optional

from .base import BasePrimitive, ScaleInfo


class WindowPrimitive(BasePrimitive):
    """A linear wall-hosted window (scalable along the wall axis)."""

    SILL_DEPTH: float = 6.0

    def __init__(
        self,
        primitive_id: str,
        center: tuple[float, float],
        width: float,
        orientation_angle: float = 0.0,
        host_wall_id: Optional[str] = None,
        confidence: float = 1.0,
        scale_info: Optional[ScaleInfo] = None,
    ) -> None:
        super().__init__(primitive_id, confidence, scale_info)
        self.center = center
        self.width = width
        self.orientation_angle = orientation_angle
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
        color = "#3355cc"
        s, e = self._endpoints()
        a = math.radians(self.orientation_angle)
        perp_cos = math.cos(a + math.pi / 2.0)
        perp_sin = math.sin(a + math.pi / 2.0)
        d = self.SILL_DEPTH / 2.0

        outer1 = (s[0] - d * perp_cos, s[1] - d * perp_sin)
        outer2 = (e[0] - d * perp_cos, e[1] - d * perp_sin)
        inner1 = (s[0] + d * perp_cos, s[1] + d * perp_sin)
        inner2 = (e[0] + d * perp_cos, e[1] + d * perp_sin)

        def pt(p: tuple[float, float]) -> str:
            return f"{p[0]:.2f},{p[1]:.2f}"

        border = (
            f'<polygon points="{pt(outer1)} {pt(outer2)} {pt(inner2)} {pt(inner1)}" '
            f'fill="#d0e8ff" stroke="{color}" stroke-width="1" />'
        )
        midline = (
            f'<line x1="{s[0]:.2f}" y1="{s[1]:.2f}" '
            f'x2="{e[0]:.2f}" y2="{e[1]:.2f}" '
            f'stroke="{color}" stroke-width="1.5" />'
        )
        return (
            f'<g id="{self.primitive_id}" data-type="window">'
            f'{border}{midline}'
            f'</g>'
        )

    def bounds(self) -> tuple[float, float, float, float]:
        s, e = self._endpoints()
        half_d = self.SILL_DEPTH / 2.0
        return (
            min(s[0], e[0]) - half_d,
            min(s[1], e[1]) - half_d,
            max(s[0], e[0]) + half_d,
            max(s[1], e[1]) + half_d,
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
