"""WallPrimitive: centerline + thickness representation of a wall segment.

Also defines OuterWallLoopPrimitive, a closed-polyline wrapper around the
building envelope used as a topology reference for floor generation and for
validating loop closure (spec_v007 SS9.1). It is not drawn directly in the
final SVG - the per-edge WallPrimitive segments (wall_type="outer") already
render the loop, and they are the segments that get split at openings.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from .base import BasePrimitive, ScaleInfo

WallType = Literal["outer", "inner", "unknown"]


class WallPrimitive(BasePrimitive):
    """A straight wall segment stored as centerline endpoints + thickness."""

    def __init__(
        self,
        primitive_id: str,
        start: tuple[float, float],
        end: tuple[float, float],
        thickness: float = 8.0,
        thickness_mm: Optional[float] = None,
        wall_type: WallType = "unknown",
        confidence: float = 1.0,
        scale_info: Optional[ScaleInfo] = None,
        **base_kwargs,
    ) -> None:
        super().__init__(primitive_id, confidence, scale_info, **base_kwargs)
        self.start = start
        self.end = end
        self.thickness = thickness
        self.thickness_mm = thickness_mm
        self.wall_type: WallType = wall_type

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
            f'<line id="{self.primitive_id}" data-wall-type="{self.wall_type}" '
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


class OuterWallLoopPrimitive(BasePrimitive):
    """The closed polyline representing the building's exterior envelope.

    Topology reference object only - used to build FloorPrimitive and to
    validate loop closure. Not rendered as part of the final ``wall`` SVG
    group (the per-edge WallPrimitive outer segments are rendered instead).
    """

    def __init__(
        self,
        primitive_id: str,
        centerline: list[tuple[float, float]],
        thickness: float = 8.0,
        thickness_mm: Optional[float] = None,
        confidence: float = 1.0,
        scale_info: Optional[ScaleInfo] = None,
        **base_kwargs,
    ) -> None:
        super().__init__(primitive_id, confidence, scale_info, **base_kwargs)
        self.centerline = centerline
        self.thickness = thickness
        self.thickness_mm = thickness_mm

    def is_closed(self) -> bool:
        """A closed loop here is a ring of >= 3 distinct vertices enclosing a
        non-degenerate area (no duplicated first/last point is required - the
        wrap-around edge is implicit)."""
        pts = self.centerline
        n = len(pts)
        if n < 3:
            return False
        area2 = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))
        return abs(area2) > 1e-6

    def to_svg(self) -> str:
        """Debug-only representation - the loop is not part of the final wall group."""
        pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in self.centerline)
        return (
            f'<polygon id="{self.primitive_id}" data-type="outer_wall_loop" '
            f'points="{pts_str}" fill="none" stroke="#999999" '
            f'stroke-width="1" stroke-dasharray="2 2" />'
        )

    def bounds(self) -> tuple[float, float, float, float]:
        if not self.centerline:
            return 0.0, 0.0, 0.0, 0.0
        xs = [p[0] for p in self.centerline]
        ys = [p[1] for p in self.centerline]
        return min(xs), min(ys), max(xs), max(ys)

    def transform(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        angle_deg: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
    ) -> None:
        self.centerline = [(x * sx + dx, y * sy + dy) for x, y in self.centerline]
