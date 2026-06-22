"""FloorPrimitive: filled building footprint polygon behind all other elements."""

from __future__ import annotations

from typing import Optional

from .base import BasePrimitive, ScaleInfo


class FloorPrimitive(BasePrimitive):
    """Filled rectilinear footprint polygon — rendered behind all linework."""

    # task08: floor must render pure white, not the CNN debug palette's
    # off-white, to read as a clean architectural drawing.
    FILL_COLOR = "#ffffff"

    def __init__(
        self,
        primitive_id: str,
        polygon: list[tuple[float, float]],
        confidence: float = 1.0,
        scale_info: Optional[ScaleInfo] = None,
        **base_kwargs,
    ) -> None:
        super().__init__(primitive_id, confidence, scale_info, **base_kwargs)
        self.polygon = polygon

    @property
    def area(self) -> float:
        pts = self.polygon
        n = len(pts)
        if n < 3:
            return 0.0
        return abs(
            sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
                for i in range(n))
        ) / 2.0

    def to_svg(self) -> str:
        pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in self.polygon)
        return (
            f'<polygon id="{self.primitive_id}" class="floor" '
            f'points="{pts_str}" '
            f'fill="{self.FILL_COLOR}" stroke="none" />'
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
        self.polygon = [(x * sx + dx, y * sy + dy) for x, y in self.polygon]
