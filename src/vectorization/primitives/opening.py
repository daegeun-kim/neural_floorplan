"""OpeningPrimitive: a generic wall-hosted opening (gap in a wall)."""

from __future__ import annotations

import math
from typing import Literal, Optional

from .base import BasePrimitive, ScaleInfo

OpeningType = Literal["generic", "door_candidate", "window_candidate"]


class OpeningPrimitive(BasePrimitive):
    """A generic interruption hosted on a wall segment."""

    def __init__(
        self,
        primitive_id: str,
        center: tuple[float, float],
        width: float,
        orientation_angle: float = 0.0,
        host_wall_id: Optional[str] = None,
        opening_type: OpeningType = "generic",
        confidence: float = 1.0,
        scale_info: Optional[ScaleInfo] = None,
    ) -> None:
        super().__init__(primitive_id, confidence, scale_info)
        self.center = center
        self.width = width
        self.orientation_angle = orientation_angle
        self.host_wall_id = host_wall_id
        self.opening_type = opening_type

    @property
    def start(self) -> tuple[float, float]:
        half = self.width / 2.0
        angle_rad = math.radians(self.orientation_angle)
        return (
            self.center[0] - half * math.cos(angle_rad),
            self.center[1] - half * math.sin(angle_rad),
        )

    @property
    def end(self) -> tuple[float, float]:
        half = self.width / 2.0
        angle_rad = math.radians(self.orientation_angle)
        return (
            self.center[0] + half * math.cos(angle_rad),
            self.center[1] + half * math.sin(angle_rad),
        )

    def to_svg(self) -> str:
        s = self.start
        e = self.end
        color = "#cc4444" if self.host_wall_id else "#ff8800"
        label = self.opening_type
        return (
            f'<line id="{self.primitive_id}" '
            f'x1="{s[0]:.2f}" y1="{s[1]:.2f}" '
            f'x2="{e[0]:.2f}" y2="{e[1]:.2f}" '
            f'stroke="{color}" stroke-width="2" '
            f'stroke-dasharray="4 2" '
            f'data-type="{label}" />'
        )

    def bounds(self) -> tuple[float, float, float, float]:
        s, e = self.start, self.end
        return min(s[0], e[0]), min(s[1], e[1]), max(s[0], e[0]), max(s[1], e[1])

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
