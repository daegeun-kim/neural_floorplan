"""Basic geometric cleanup: snap walls, merge collinear segments, project openings."""

from __future__ import annotations

import math
from typing import Union

from .primitives import OpeningPrimitive, WallPrimitive, WindowPrimitive


def snap_walls_to_45(
    walls: list[WallPrimitive], snap_threshold_deg: float = 8.0
) -> list[WallPrimitive]:
    """Snap wall angles to the nearest 45-degree increment.

    Walls near a cardinal direction (0/90/180/270) within `snap_threshold_deg`
    snap to that cardinal, preferring horizontal/vertical output when the
    evidence is near orthogonal. All other walls snap to the nearest 45-degree
    multiple so every wall reads as architectural linework.
    """
    for wall in walls:
        angle = wall.orientation_angle
        cx, cy = wall.center
        half = wall.length / 2.0

        norm = angle % 180.0
        nearest_cardinal = round(norm / 90.0) * 90.0
        if abs(norm - nearest_cardinal) <= snap_threshold_deg:
            target = nearest_cardinal % 180.0
        else:
            target = round(norm / 45.0) * 45.0 % 180.0

        target_rad = math.radians(target)
        dx, dy = math.cos(target_rad) * half, math.sin(target_rad) * half

        # Preserve the original segment's direction sign.
        orig_rad = math.radians(angle)
        if math.cos(orig_rad) * math.cos(target_rad) + math.sin(orig_rad) * math.sin(target_rad) < 0:
            dx, dy = -dx, -dy

        wall.start = (cx - dx, cy - dy)
        wall.end = (cx + dx, cy + dy)

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


def split_walls_at_openings(
    walls: list[WallPrimitive],
    hosted: list[Union[OpeningPrimitive, WindowPrimitive]],
    min_segment_px: float = 3.0,
) -> list[WallPrimitive]:
    """Split wall segments at projected opening/window intervals.

    For each hosted opening or window, the wall that contains it is split into
    two segments on either side of the gap (the opening span is removed).
    Walls with no hosted openings are returned unchanged.

    Args:
        walls:          Input wall primitives.
        hosted:         Openings/windows that have a ``host_wall_id``.
        min_segment_px: Minimum length for a resulting wall segment to be kept.

    Returns:
        New list of WallPrimitive objects with splits applied.
    """
    # Collect gap intervals per wall: {wall_id: [(t_start, t_end), ...]}
    gaps: dict[str, list[tuple[float, float]]] = {w.primitive_id: [] for w in walls}
    wall_map = {w.primitive_id: w for w in walls}

    for obj in hosted:
        wid = getattr(obj, "host_wall_id", None)
        if not wid or wid not in wall_map:
            continue
        wall = wall_map[wid]
        x1, y1 = wall.start
        x2, y2 = wall.end
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-6:
            continue

        cx, cy = obj.center
        t_center = ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq
        seg_len = math.sqrt(seg_len_sq)
        half_t = (obj.width / 2.0) / seg_len

        t_start = max(0.0, t_center - half_t)
        t_end   = min(1.0, t_center + half_t)
        if t_end > t_start:
            gaps[wid].append((t_start, t_end))

    result: list[WallPrimitive] = []
    for wall in walls:
        wall_gaps = gaps.get(wall.primitive_id, [])
        if not wall_gaps:
            result.append(wall)
            continue

        # Sort and merge overlapping gaps
        wall_gaps.sort()
        merged: list[tuple[float, float]] = [wall_gaps[0]]
        for gs, ge in wall_gaps[1:]:
            ms, me = merged[-1]
            if gs <= me:
                merged[-1] = (ms, max(me, ge))
            else:
                merged.append((gs, ge))

        # Build solid segment intervals (complement of gaps)
        solid_intervals: list[tuple[float, float]] = []
        prev = 0.0
        for gs, ge in merged:
            if gs - prev > 1e-4:
                solid_intervals.append((prev, gs))
            prev = ge
        if 1.0 - prev > 1e-4:
            solid_intervals.append((prev, 1.0))

        x1, y1 = wall.start
        x2, y2 = wall.end
        for idx, (ta, tb) in enumerate(solid_intervals):
            sa = (x1 + ta * (x2 - x1), y1 + ta * (y2 - y1))
            sb = (x1 + tb * (x2 - x1), y1 + tb * (y2 - y1))
            seg_len = math.hypot(sb[0] - sa[0], sb[1] - sa[1])
            if seg_len < min_segment_px:
                continue
            result.append(WallPrimitive(
                primitive_id=f"{wall.primitive_id}_s{idx}",
                start=sa,
                end=sb,
                thickness=wall.thickness,
                confidence=wall.confidence,
                scale_info=wall.scale_info,
            ))

    return result if result else walls


def apply_geometry_rules(
    walls: list[WallPrimitive],
    openings: list[OpeningPrimitive],
    snap_threshold_deg: float = 8.0,
) -> tuple[list[WallPrimitive], list[OpeningPrimitive]]:
    walls = snap_walls_to_45(walls, snap_threshold_deg)
    wall_map = {w.primitive_id: w for w in walls}
    for op in openings:
        if op.host_wall_id and op.host_wall_id in wall_map:
            project_opening_onto_wall(op, wall_map[op.host_wall_id])
    return walls, openings
