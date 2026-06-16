"""Basic geometric cleanup: snap walls, merge collinear segments, project openings."""

from __future__ import annotations

import math

from .primitives import OpeningPrimitive, WallPrimitive


def snap_walls_to_cardinal(
    walls: list[WallPrimitive], snap_threshold_deg: float = 8.0
) -> list[WallPrimitive]:
    """Snap near-horizontal and near-vertical wall lines to exact cardinal angles."""
    for wall in walls:
        angle = wall.orientation_angle
        cx, cy = wall.center
        length = wall.length
        half = length / 2.0

        # Determine snap target
        abs_angle = angle % 180.0
        if abs_angle > 90.0:
            abs_angle = 180.0 - abs_angle
        snap_to_horizontal = abs_angle <= snap_threshold_deg
        snap_to_vertical = abs(abs_angle - 90.0) <= snap_threshold_deg

        if snap_to_horizontal:
            wall.start = (cx - half, cy)
            wall.end = (cx + half, cy)
        elif snap_to_vertical:
            wall.start = (cx, cy - half)
            wall.end = (cx, cy + half)

    return walls


def project_opening_onto_wall(
    opening: OpeningPrimitive, wall: WallPrimitive
) -> OpeningPrimitive:
    """Project opening center onto the host wall centerline."""
    x1, y1 = wall.start
    x2, y2 = wall.end
    cx, cy = opening.center
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return opening
    t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    opening.center = (proj_x, proj_y)
    opening.orientation_angle = wall.orientation_angle
    return opening


def apply_geometry_rules(
    walls: list[WallPrimitive],
    openings: list[OpeningPrimitive],
    snap_threshold_deg: float = 8.0,
) -> tuple[list[WallPrimitive], list[OpeningPrimitive]]:
    walls = snap_walls_to_cardinal(walls, snap_threshold_deg)
    wall_map = {w.primitive_id: w for w in walls}
    for op in openings:
        if op.host_wall_id and op.host_wall_id in wall_map:
            project_opening_onto_wall(op, wall_map[op.host_wall_id])
    return walls, openings
