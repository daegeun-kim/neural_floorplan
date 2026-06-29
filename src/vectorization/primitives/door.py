"""Door primitives reconstructed from door_origin / door_leaf / door_arc evidence.

Per spec_v007 SS11 / spec_v008 SS9, a door is composed of three primitives:

- DoorOriginPrimitive: the wall-hosted threshold segment (replaces a wall
  segment, like a window). Has two endpoints measured directly from
  door_origin mask evidence projected onto the host wall.
- DoorLeafPrimitive: generated procedurally from one endpoint of the origin
  (the "hinge", chosen from door_leaf/door_arc evidence density) - a segment
  perpendicular to the host wall, with length equal to the origin's width.
- DoorArcPrimitive: a 90-degree arc from the origin's far endpoint to the
  leaf's end, centered on the hinge, radius equal to the door width.

The leaf/arc perpendicular-vector math is unchanged from the previous
single-class DoorPrimitive implementation - only the data model changed.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from .base import BasePrimitive, ScaleInfo

SwingSide = Literal["left", "right"]


def _perp_end(
    hinge: tuple[float, float],
    width: float,
    orientation_angle: float,
    swing_direction: SwingSide,
) -> tuple[float, float]:
    """Endpoint of a `width`-long segment perpendicular to `orientation_angle`."""
    angle_rad = math.radians(orientation_angle)
    sign = 1.0 if swing_direction == "left" else -1.0
    perp_angle = angle_rad + sign * math.pi / 2.0
    return (
        hinge[0] + width * math.cos(perp_angle),
        hinge[1] + width * math.sin(perp_angle),
    )


class DoorOriginPrimitive(BasePrimitive):
    """Wall-hosted door threshold segment - replaces a wall segment.

    Mirrors WindowPrimitive's shape (center/width/orientation_angle/host_wall_id)
    so the shared wall-hosting/projection/splitting helpers work on both.
    """

    COLOR = "#a046b4"

    def __init__(
        self,
        primitive_id: str,
        center: tuple[float, float],
        width: float,
        orientation_angle: float = 0.0,
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
        self.width_mm = width_mm
        self.host_wall_id = host_wall_id

    @property
    def start(self) -> tuple[float, float]:
        half = self.width / 2.0
        a = math.radians(self.orientation_angle)
        return (self.center[0] - half * math.cos(a), self.center[1] - half * math.sin(a))

    @property
    def end(self) -> tuple[float, float]:
        half = self.width / 2.0
        a = math.radians(self.orientation_angle)
        return (self.center[0] + half * math.cos(a), self.center[1] + half * math.sin(a))

    def to_svg(self) -> str:
        s, e = self.start, self.end
        return (
            f'<line id="{self.primitive_id}" data-type="door_origin" '
            f'x1="{s[0]:.2f}" y1="{s[1]:.2f}" x2="{e[0]:.2f}" y2="{e[1]:.2f}" '
            f'stroke="{self.COLOR}" stroke-width="2" stroke-linecap="butt" />'
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


class DoorLeafPrimitive(BasePrimitive):
    """Door leaf in the open position - perpendicular to the door origin.

    Rendered as a thin symbolic SVG line (task09 supersedes task08's
    closed-polygon decision for this primitive) - a leaf panel is a
    symbolic indicator, not wall-thickness geometry.
    """

    COLOR = "#eb8c50"

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
        **base_kwargs,
    ) -> None:
        super().__init__(primitive_id, confidence, scale_info, **base_kwargs)
        self.hinge_point = hinge_point
        self.width = width
        self.orientation_angle = orientation_angle
        self.swing_direction = swing_direction
        self.host_wall_id = host_wall_id

    @property
    def leaf_end(self) -> tuple[float, float]:
        return _perp_end(self.hinge_point, self.width, self.orientation_angle, self.swing_direction)

    def to_svg(self) -> str:
        hx, hy = self.hinge_point
        ex, ey = self.leaf_end
        return (
            f'<line id="{self.primitive_id}" data-type="door_leaf" '
            f'x1="{hx:.2f}" y1="{hy:.2f}" x2="{ex:.2f}" y2="{ey:.2f}" '
            f'stroke="{self.COLOR}" stroke-width="1.5" />'
        )

    def bounds(self) -> tuple[float, float, float, float]:
        hx, hy = self.hinge_point
        ex, ey = self.leaf_end
        return min(hx, ex), min(hy, ey), max(hx, ex), max(hy, ey)

    def transform(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        angle_deg: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
    ) -> None:
        self.hinge_point = (self.hinge_point[0] * sx + dx, self.hinge_point[1] * sy + dy)
        self.width *= sx
        self.orientation_angle += angle_deg


class DoorArcPrimitive(BasePrimitive):
    """90-degree swing arc from the door origin's far end to the leaf's end.

    Rendered as a stroked arc (task08 allows "arc or closed arc primitive"
    for this component, unlike wall/window/door-origin/door-leaf which must
    be filled polygons).
    """

    COLOR = "#dc5a5a"

    def __init__(
        self,
        primitive_id: str,
        hinge_point: tuple[float, float],
        origin_far_point: tuple[float, float],
        width: float,
        orientation_angle: float = 0.0,
        swing_direction: SwingSide = "left",
        host_wall_id: Optional[str] = None,
        confidence: float = 1.0,
        scale_info: Optional[ScaleInfo] = None,
        **base_kwargs,
    ) -> None:
        super().__init__(primitive_id, confidence, scale_info, **base_kwargs)
        self.hinge_point = hinge_point
        self.origin_far_point = origin_far_point
        self.width = width
        self.orientation_angle = orientation_angle
        self.swing_direction = swing_direction
        self.host_wall_id = host_wall_id

    @property
    def leaf_end(self) -> tuple[float, float]:
        return _perp_end(self.hinge_point, self.width, self.orientation_angle, self.swing_direction)

    def to_svg(self) -> str:
        hx, hy = self.hinge_point
        ox, oy = self.origin_far_point
        ex, ey = self.leaf_end

        # The SVG elliptical-arc command lets the renderer pick either of two
        # valid circle centers for a given radius/endpoint pair; large-arc=0
        # always selects the minor (<=180deg) arc, and sweep selects which of
        # the two centers is used. origin_far_point and leaf_end are both
        # exactly `width` from hinge_point and 90deg apart by construction,
        # so hinge_point is the correct center only for one specific sweep
        # value - computed here from the actual angles rather than guessed
        # from swing_direction, so the arc is centered on the hinge for any
        # wall orientation (fixes the previously-reversed arcs).
        theta_start = math.atan2(oy - hy, ox - hx)
        theta_end = math.atan2(ey - hy, ex - hx)
        delta = (theta_end - theta_start + math.pi) % (2 * math.pi) - math.pi
        sweep = 1 if delta > 0 else 0

        return (
            f'<path id="{self.primitive_id}" data-type="door_arc" '
            f'd="M {ox:.2f} {oy:.2f} A {self.width:.2f} {self.width:.2f} 0 0 {sweep} '
            f'{ex:.2f} {ey:.2f}" '
            f'fill="none" stroke="{self.COLOR}" stroke-width="1.5" />'
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
        self.hinge_point = (self.hinge_point[0] * sx + dx, self.hinge_point[1] * sy + dy)
        self.origin_far_point = (self.origin_far_point[0] * sx + dx, self.origin_far_point[1] * sy + dy)
        self.width *= sx
        self.orientation_angle += angle_deg
