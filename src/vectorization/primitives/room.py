"""RoomPrimitive: simplified polygon representing a room or space region."""

from __future__ import annotations

import math
from typing import Optional

from .base import BasePrimitive, ScaleInfo


class RoomPrimitive(BasePrimitive):
    """A room region stored as a simplified polygon."""

    def __init__(
        self,
        primitive_id: str,
        polygon: list[tuple[float, float]],
        confidence: float = 1.0,
        scale_info: Optional[ScaleInfo] = None,
    ) -> None:
        super().__init__(primitive_id, confidence, scale_info)
        self.polygon = polygon

    @property
    def area(self) -> float:
        pts = self.polygon
        n = len(pts)
        if n < 3:
            return 0.0
        total = 0.0
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            total += x1 * y2 - x2 * y1
        return abs(total) / 2.0

    @property
    def centroid(self) -> tuple[float, float]:
        if not self.polygon:
            return (0.0, 0.0)
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def to_svg(self) -> str:
        if len(self.polygon) < 3:
            return ""
        pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in self.polygon)
        alpha = max(0.1, min(0.4, self.confidence * 0.4))
        return (
            f'<polygon id="{self.primitive_id}" '
            f'points="{pts_str}" '
            f'fill="rgba(80,160,220,{alpha:.2f})" '
            f'stroke="#3388aa" stroke-width="1" />'
        )

    def bounds(self) -> tuple[float, float, float, float]:
        if not self.polygon:
            return 0.0, 0.0, 0.0, 0.0
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    def transform(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        angle_deg: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
    ) -> None:
        cx, cy = self.centroid
        angle_rad = math.radians(angle_deg)
        new_pts = []
        for x, y in self.polygon:
            rx, ry = self._rotate_point(x, y, cx, cy, angle_rad)
            new_pts.append((rx * sx + dx, ry * sy + dy))
        self.polygon = new_pts
