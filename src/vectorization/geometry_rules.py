"""Basic geometric cleanup: snap walls, merge collinear segments, host/project/split openings."""

from __future__ import annotations

import math
from typing import Union

import numpy as np

from .primitives import OpeningPrimitive, WallPrimitive
from .primitives.door import DoorOriginPrimitive
from .primitives.window import WindowPrimitive

HostedOpening = Union[WindowPrimitive, DoorOriginPrimitive]


def snap_walls_to_45(
    walls: list[WallPrimitive],
    ortho_snap_deg: float = 20.0,
    diagonal_snap_deg: float = 10.0,
) -> list[WallPrimitive]:
    """Snap wall angles to the nearest 45-degree increment, orthogonal-first.

    Priority (task09): horizontal/vertical is the default interpretation for
    ambiguous evidence; a 45-degree diagonal is only used when the evidence
    is close to *exactly* 45/135 degrees. Concretely: walls within
    `ortho_snap_deg` of a cardinal direction (0/90/180/270) snap to that
    cardinal; walls within `diagonal_snap_deg` of an exact diagonal
    (45/135) snap to that diagonal; anything in between - genuinely
    ambiguous evidence that is closer to a diagonal than to a cardinal, but
    not close enough to the diagonal to call it explicit - still snaps to
    the nearest cardinal rather than the diagonal, per "do not convert
    noisy or ambiguous pixels into diagonal walls when an orthogonal
    interpretation is plausible."
    """
    for wall in walls:
        angle = wall.orientation_angle
        cx, cy = wall.center
        half = wall.length / 2.0

        norm = angle % 180.0
        nearest_cardinal = round(norm / 90.0) * 90.0 % 180.0
        dist_to_cardinal = min(abs(norm - nearest_cardinal), 180.0 - abs(norm - nearest_cardinal))
        nearest_diagonal = 45.0 if min(abs(norm - 45.0), abs(norm - 135.0)) == abs(norm - 45.0) else 135.0
        dist_to_diagonal = abs(norm - nearest_diagonal)

        if dist_to_cardinal <= ortho_snap_deg:
            target = nearest_cardinal
        elif dist_to_diagonal <= diagonal_snap_deg:
            target = nearest_diagonal
        else:
            target = nearest_cardinal

        target_rad = math.radians(target)
        dx, dy = math.cos(target_rad) * half, math.sin(target_rad) * half

        # Preserve the original segment's direction sign.
        orig_rad = math.radians(angle)
        if math.cos(orig_rad) * math.cos(target_rad) + math.sin(orig_rad) * math.sin(target_rad) < 0:
            dx, dy = -dx, -dy

        wall.start = (cx - dx, cy - dy)
        wall.end = (cx + dx, cy + dy)

    return walls


def _point_to_wall_distance(point: tuple[float, float], wall: WallPrimitive) -> float:
    """Shortest distance from `point` to the wall's clamped centerline segment."""
    cx, cy = point
    x1, y1 = wall.start
    x2, y2 = wall.end
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return math.hypot(cx - x1, cy - y1)
    t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(cx - proj_x, cy - proj_y)


def nearest_wall(
    center: tuple[float, float],
    walls: list[WallPrimitive],
    max_dist: float = 40.0,
) -> WallPrimitive | None:
    """Find the wall whose centerline is closest to `center` (within `max_dist`)."""
    if not walls:
        return None
    best_wall = None
    best_dist = max_dist
    for wall in walls:
        dist = _point_to_wall_distance(center, wall)
        if dist < best_dist:
            best_dist = dist
            best_wall = wall
    return best_wall


def _dominant_axis_angle_deg(pixel_coords: np.ndarray) -> float:
    """Angle (degrees) of the dominant axis of a point cloud, via PCA."""
    pts = pixel_coords - pixel_coords.mean(axis=0)
    if pts.shape[0] < 2:
        return 0.0
    cov = np.cov(pts.T)
    if np.ndim(cov) == 0 or not np.all(np.isfinite(cov)):
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    dominant = eigvecs[:, int(np.argmax(eigvals))]
    return math.degrees(math.atan2(dominant[1], dominant[0]))


def _hosting_probability(
    pixel_coords: np.ndarray, wall: WallPrimitive, min_remainder_px: float
) -> float:
    """Composite tie-break score in [0, 3] for how well `wall` hosts this opening
    evidence (task10 corner-ambiguity rule): orientation alignment + evidence
    overlap with the wall body + non-degenerate remainder after hosting,
    each in [0, 1] and simply summed - this is only used to break a tie
    between two already-close candidate walls, not the primary hosting
    decision, so a fitted/weighted model would be overkill.
    """
    x1, y1 = wall.start
    x2, y2 = wall.end
    dx, dy = x2 - x1, y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-6:
        return 0.0
    ux, uy = dx / seg_len, dy / seg_len

    evidence_angle = _dominant_axis_angle_deg(pixel_coords)
    diff = abs((evidence_angle - wall.orientation_angle) % 180.0)
    if diff > 90.0:
        diff = 180.0 - diff
    orientation_term = max(0.0, 1.0 - diff / 90.0)

    rel_x = pixel_coords[:, 0] - x1
    rel_y = pixel_coords[:, 1] - y1
    t_px = rel_x * ux + rel_y * uy
    perp_dist = np.abs(rel_x * uy - rel_y * ux)

    overlap_slack_px = 4.0
    overlap_term = float(np.mean(perp_dist <= (wall.thickness / 2.0 + overlap_slack_px)))

    t_min, t_max = float(t_px.min()), float(t_px.max())
    remainder_left = max(0.0, t_min)
    remainder_right = max(0.0, seg_len - t_max)
    remainder_px = min(remainder_left, remainder_right)
    remainder_term = max(0.0, min(1.0, remainder_px / (4.0 * max(min_remainder_px, 1e-6))))

    return orientation_term + overlap_term + remainder_term


def select_host_wall_for_opening(
    pixel_coords: np.ndarray,
    walls: list[WallPrimitive],
    max_dist: float = 40.0,
    corner_ambiguity_px: float = 25.0,
    min_remainder_px: float = 3.0,
) -> WallPrimitive | None:
    """Pick exactly one host wall for an opening's pixel evidence (task10).

    Cheap path: the single nearest wall, same as `nearest_wall`. Only when a
    second wall is within `corner_ambiguity_px` of the nearest one (the
    opening genuinely sits near where two walls meet, e.g. a corner) do we
    score both candidates with `_hosting_probability` and pick the higher
    scorer - this avoids ever straddling an opening across two walls or
    splitting a wall into a near-zero-length stub, by committing the opening
    fully to one wall up front.
    """
    if walls is None or len(walls) == 0 or pixel_coords is None or len(pixel_coords) == 0:
        return None
    center = (float(pixel_coords[:, 0].mean()), float(pixel_coords[:, 1].mean()))

    scored = [
        (dist, wall)
        for wall in walls
        if (dist := _point_to_wall_distance(center, wall)) <= max_dist
    ]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    nearest_dist, nearest = scored[0]
    if len(scored) == 1 or (scored[1][0] - nearest_dist) > corner_ambiguity_px:
        return nearest

    second_dist, second = scored[1]
    score_nearest = _hosting_probability(pixel_coords, nearest, min_remainder_px)
    score_second = _hosting_probability(pixel_coords, second, min_remainder_px)
    return nearest if score_nearest >= score_second else second


def project_pixels_onto_wall(
    pixel_coords: np.ndarray, wall: WallPrimitive
) -> tuple[tuple[float, float], float, float, float]:
    """Project a mask component's pixel coordinates onto a wall's centerline.

    `pixel_coords` is an (N, 2) array of (x, y) pixel coordinates for one
    connected component (e.g. a window or door_origin component). Returns
    (center, width, t_min, t_max): the hosted segment's center point and
    width along the wall, and the raw parametric extent on the wall line.

    This is the shared "locate the boundary/transition points between the
    component and the host wall" step used by both window and door_origin
    extraction (spec_v008 SS8/SS9.1).
    """
    x1, y1 = wall.start
    x2, y2 = wall.end
    dx, dy = x2 - x1, y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-6:
        cx = float(pixel_coords[:, 0].mean())
        cy = float(pixel_coords[:, 1].mean())
        return (cx, cy), 0.0, 0.0, 0.0

    ux, uy = dx / seg_len, dy / seg_len
    rel_x = pixel_coords[:, 0] - x1
    rel_y = pixel_coords[:, 1] - y1
    t_px = rel_x * ux + rel_y * uy  # signed projection length in px along wall

    t_min = float(t_px.min())
    t_max = float(t_px.max())
    width = max(t_max - t_min, 1.0)
    t_mid = (t_min + t_max) / 2.0
    center = (x1 + ux * t_mid, y1 + uy * t_mid)
    return center, width, t_min / seg_len, t_max / seg_len


def project_opening_onto_wall(
    opening: OpeningPrimitive | HostedOpening, wall: WallPrimitive
) -> OpeningPrimitive | HostedOpening:
    """Project an opening/window/door-origin's center onto the host wall centerline."""
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
    hosted: list[HostedOpening],
    min_segment_px: float = 3.0,
) -> list[WallPrimitive]:
    """Split wall segments at projected window/door-origin intervals.

    For each hosted window or door origin, the wall that contains it is split
    into two segments on either side of the gap (the opening span is
    removed). Walls with no hosted openings are returned unchanged.

    Args:
        walls:          Input wall primitives.
        hosted:         Windows/door-origins that have a ``host_wall_id``.
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
        t_end = min(1.0, t_center + half_t)
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
                thickness_mm=wall.thickness_mm,
                wall_type=wall.wall_type,
                confidence=wall.confidence,
                scale_info=wall.scale_info,
            ))

    return result if result else walls


def apply_geometry_rules(
    walls: list[WallPrimitive],
    openings: list[OpeningPrimitive],
    ortho_snap_deg: float = 20.0,
    diagonal_snap_deg: float = 10.0,
) -> tuple[list[WallPrimitive], list[OpeningPrimitive]]:
    walls = snap_walls_to_45(walls, ortho_snap_deg, diagonal_snap_deg)
    wall_map = {w.primitive_id: w for w in walls}
    for op in openings:
        if op.host_wall_id and op.host_wall_id in wall_map:
            project_opening_onto_wall(op, wall_map[op.host_wall_id])
    return walls, openings
