"""DoorPrimitive: parametric single-hinged door symbol (leaf + quarter-arc)."""

from __future__ import annotations

import math
from typing import Literal, Optional

from .base import BasePrimitive, ScaleInfo

SwingSide = Literal["left", "right"]


class DoorPrimitive(BasePrimitive):
    """Single hinged door: leaf line + quarter-circle swing arc.

    Subclass for sliding, double, or folding variants later.
    """

    def __init__(
        self,
        primitive_id: str,
        hinge_point: tuple[float, float],
        width: float,
        orientation_angle: float = 0.0,
        swing_direction: SwingSide = "left",
        host_wall_id: Optional[str] = None,
        confidence: float = 1.0,
        scale_info: Optional[ScaleInfo] = None,
    ) -> None:
        super().__init__(primitive_id, confidence, scale_info)
        self.hinge_point = hinge_point
        self.width = width
        self.orientation_angle = orientation_angle
        self.swing_direction = swing_direction
        self.host_wall_id = host_wall_id

    def _leaf_end(self) -> tuple[float, float]:
        """Tip of the door leaf when fully closed (along the wall)."""
        angle_rad = math.radians(self.orientation_angle)
        sign = 1.0 if self.swing_direction == "left" else -1.0
        return (
            self.hinge_point[0] + sign * self.width * math.cos(angle_rad),
            self.hinge_point[1] + sign * self.width * math.sin(angle_rad),
        )

    def to_svg(self) -> str:
        color = self._svg_color_for_confidence(self.confidence)
        hx, hy = self.hinge_point
        leaf_end = self._leaf_end()

        angle_rad = math.radians(self.orientation_angle)
        sign = 1.0 if self.swing_direction == "left" else -1.0
        perp_angle = angle_rad + sign * math.pi / 2.0
        arc_end_x = hx + self.width * math.cos(perp_angle)
        arc_end_y = hy + self.width * math.sin(perp_angle)

        # SVG arc: sweep clockwise for right swing, counter for left
        sweep = 0 if self.swing_direction == "left" else 1

        leaf = (
            f'<line x1="{hx:.2f}" y1="{hy:.2f}" '
            f'x2="{leaf_end[0]:.2f}" y2="{leaf_end[1]:.2f}" '
            f'stroke="{color}" stroke-width="1.5" />'
        )
        arc = (
            f'<path d="M {leaf_end[0]:.2f} {leaf_end[1]:.2f} '
            f'A {self.width:.2f} {self.width:.2f} 0 0 {sweep} '
            f'{arc_end_x:.2f} {arc_end_y:.2f}" '
            f'fill="none" stroke="{color}" stroke-width="1" stroke-dasharray="3 2" />'
        )
        return (
            f'<g id="{self.primitive_id}" data-type="door">'
            f'{leaf}{arc}'
            f'</g>'
        )

    def bounds(self) -> tuple[float, float, float, float]:
        hx, hy = self.hinge_point
        r = self.width
        return hx - r, hy - r, hx + r, hy + r

    def transform(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        angle_deg: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
    ) -> None:
        self.hinge_point = (
            self.hinge_point[0] * sx + dx,
            self.hinge_point[1] * sy + dy,
        )
        self.width *= sx
        self.orientation_angle += angle_deg
