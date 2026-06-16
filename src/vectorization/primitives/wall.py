"""WallPrimitive: centerline + thickness representation of a wall segment."""

from __future__ import annotations

import math
from typing import Optional

from .base import BasePrimitive, ScaleInfo


class WallPrimitive(BasePrimitive):
    """A straight wall segment stored as centerline endpoints + thickness."""

    def __init__(
        self,
        primitive_id: str,
        start: tuple[float, float],
        end: tuple[float, float],
        thickness: float = 8.0,
        confidence: float = 1.0,
        scale_info: Optional[ScaleInfo] = None,
    ) -> None:
        super().__init__(primitive_id, confidence, scale_info)
        self.start = start
        self.end = end
        self.thickness = thickness

    @property
    def orientation_angle(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.degrees(math.atan2(dy, dx))

    @property
    def length(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.hypot(dx, dy)

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.start[0] + self.end[0]) / 2.0,
            (self.start[1] + self.end[1]) / 2.0,
        )

    def to_svg(self) -> str:
        color = self._svg_color_for_confidence(self.confidence)
        return (
            f'<line id="{self.primitive_id}" '
            f'x1="{self.start[0]:.2f}" y1="{self.start[1]:.2f}" '
            f'x2="{self.end[0]:.2f}" y2="{self.end[1]:.2f}" '
            f'stroke="{color}" stroke-width="{self.thickness:.2f}" '
            f'stroke-linecap="square" />'
        )

    def bounds(self) -> tuple[float, float, float, float]:
        half = self.thickness / 2.0
        x_min = min(self.start[0], self.end[0]) - half
        y_min = min(self.start[1], self.end[1]) - half
        x_max = max(self.start[0], self.end[0]) + half
        y_max = max(self.start[1], self.end[1]) + half
        return x_min, y_min, x_max, y_max

    def transform(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        angle_deg: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
    ) -> None:
        cx, cy = self.center
        angle_rad = math.radians(angle_deg)
        sx0, sy0 = self.start
        ex0, ey0 = self.end
        sx0, sy0 = self._rotate_point(sx0, sy0, cx, cy, angle_rad)
        ex0, ey0 = self._rotate_point(ex0, ey0, cx, cy, angle_rad)
        self.start = (sx0 * sx + dx, sy0 * sy + dy)
        self.end = (ex0 * sx + dx, ey0 * sy + dy)
        self.thickness *= (sx + sy) / 2.0
