"""Extract WallPrimitive objects from a cleaned wall binary mask.

Two-stage, procedural extraction:
  1. Outer wall loop  - largest external contour of the wall (+room) evidence,
     rectilinearized into a closed axis-aligned ring of centerline segments.
  2. Inner walls      - skeleton + Hough line evidence from whatever wall
     mask remains after erasing a band around the outer loop.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .primitives import ScaleInfo, WallPrimitive


def _skeletonize_wall(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    skel = skeletonize(binary).astype(np.uint8) * 255
    return skel


def _snap_angle(angle_deg: float, snap_threshold: float = 8.0) -> float:
    """Snap angle to nearest cardinal (0, 90, 180, 270) if within threshold."""
    cardinals = [0.0, 90.0, 180.0, -90.0, -180.0]
    for c in cardinals:
        if abs(angle_deg - c) <= snap_threshold:
            return c
    return angle_deg


def _segment_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _segment_length(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def _estimate_wall_thickness(wall_mask: np.ndarray, skel: np.ndarray) -> float:
    dist = cv2.distanceTransform(wall_mask, cv2.DIST_L2, 5)
    skel_pts = skel > 0
    if skel_pts.any():
        return float(np.median(dist[skel_pts]) * 2.0)
    return 8.0


def _merge_collinear_segments(
    segments: list[tuple[float, float, float, float]],
    merge_dist: float = 6.0,
) -> list[tuple[float, float, float, float]]:
    """Merge segments that are nearly collinear and close together."""
    if not segments:
        return []
    merged = list(segments)
    changed = True
    while changed:
        changed = False
        out: list[tuple[float, float, float, float]] = []
        used = [False] * len(merged)
        for i, seg_i in enumerate(merged):
            if used[i]:
                continue
            x1, y1, x2, y2 = seg_i
            ang_i = _snap_angle(_segment_angle(x1, y1, x2, y2))
            for j, seg_j in enumerate(merged):
                if i == j or used[j]:
                    continue
                x3, y3, x4, y4 = seg_j
                ang_j = _snap_angle(_segment_angle(x3, y3, x4, y4))
                if abs(ang_i - ang_j) > 15.0:
                    continue
                # Check if endpoints are close enough to merge
                close = (
                    math.hypot(x2 - x3, y2 - y3) <= merge_dist
                    or math.hypot(x1 - x4, y1 - y4) <= merge_dist
                    or math.hypot(x1 - x3, y1 - y3) <= merge_dist
                    or math.hypot(x2 - x4, y2 - y4) <= merge_dist
                )
                if close:
                    pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    if abs(ang_i) <= 45 or abs(ang_i) >= 135:
                        min_x = min(xs)
                        max_x = max(xs)
                        ys_sorted = sorted(ys)
                        mid_y = (ys_sorted[1] + ys_sorted[2]) / 2.0
                        x1, y1, x2, y2 = min_x, mid_y, max_x, mid_y
                    else:
                        min_y = min(ys)
                        max_y = max(ys)
                        xs_sorted = sorted(xs)
                        mid_x = (xs_sorted[1] + xs_sorted[2]) / 2.0
                        x1, y1, x2, y2 = mid_x, min_y, mid_x, max_y
                    used[j] = True
                    changed = True
            out.append((x1, y1, x2, y2))
            used[i] = True
        merged = out
    return merged


def _rectilinearize_contour(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Snap each contour segment to exact horizontal or vertical direction.

    Classifies each segment as H or V based on original coordinates, then
    cascades the snapping so every resulting segment is axis-aligned.
    """
    n = len(pts)
    if n < 3:
        return list(pts)

    # Classify each segment based on original coordinates
    is_horizontal: list[bool] = []
    for i in range(n):
        j = (i + 1) % n
        dx = abs(pts[j][0] - pts[i][0])
        dy = abs(pts[j][1] - pts[i][1])
        is_horizontal.append(dx >= dy)

    # Build snapped polygon by cascading: each point inherits one coordinate
    # from the previous snapped point
    snapped: list[tuple[float, float]] = [pts[0]]
    for i in range(1, n):
        prev = snapped[-1]
        ox, oy = pts[i]
        if is_horizontal[i - 1]:
            snapped.append((ox, prev[1]))  # same y as previous
        else:
            snapped.append((prev[0], oy))  # same x as previous

    # Close the polygon: align the first point with the last so the wrap-around
    # edge (snapped[-1] -> snapped[0]) is also axis-aligned.
    if is_horizontal[n - 1]:  # closing segment is horizontal -> share y with last
        snapped[0] = (snapped[0][0], snapped[-1][1])
    else:  # closing segment is vertical -> share x with last
        snapped[0] = (snapped[-1][0], snapped[0][1])

    # Remove near-duplicate consecutive points
    result: list[tuple[float, float]] = [snapped[0]]
    for pt in snapped[1:]:
        if abs(pt[0] - result[-1][0]) > 0.5 or abs(pt[1] - result[-1][1]) > 0.5:
            result.append(pt)

    if (
        len(result) > 1
        and abs(result[-1][0] - result[0][0]) < 0.5
        and abs(result[-1][1] - result[0][1]) < 0.5
    ):
        result.pop()

    return result if len(result) >= 3 else list(pts)


def extract_outer_wall_loop(
    wall_mask: np.ndarray,
    room_mask: np.ndarray | None = None,
    thickness: float | None = None,
    dilate_px: int = 8,
    simplify_epsilon_ratio: float = 0.008,
    scale_info: ScaleInfo | None = None,
) -> tuple[list[WallPrimitive], list[tuple[float, float]]]:
    """Build the closed, rectilinear outer wall loop from wall (+ room) evidence.

    Returns (outer_wall_segments, outer_polygon). The outer wall loop is the
    most outer wall and is generated before any inner walls.
    """
    fg = wall_mask
    if room_mask is not None:
        fg = np.maximum(wall_mask, room_mask)
    if fg is None or not fg.any():
        return [], []

    k_size = dilate_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], []

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return [], []

    perimeter = cv2.arcLength(contour, True)
    epsilon = max(simplify_epsilon_ratio * perimeter, 3.0)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    raw_pts = [(float(p[0][0]), float(p[0][1])) for p in approx]

    polygon = _rectilinearize_contour(raw_pts)
    if len(polygon) < 3:
        return [], []

    if thickness is None:
        thickness = _estimate_wall_thickness(wall_mask, _skeletonize_wall(wall_mask))

    n = len(polygon)
    outer_walls: list[WallPrimitive] = []
    for i in range(n):
        start = polygon[i]
        end = polygon[(i + 1) % n]
        if math.hypot(end[0] - start[0], end[1] - start[1]) < 1.0:
            continue
        outer_walls.append(
            WallPrimitive(
                primitive_id=f"wall_outer_{i:04d}",
                start=start,
                end=end,
                thickness=thickness,
                confidence=1.0,
                scale_info=scale_info,
            )
        )

    return outer_walls, polygon


def _erase_outer_band(
    wall_mask: np.ndarray, outer_polygon: list[tuple[float, float]], thickness: float
) -> np.ndarray:
    """Remove the band of wall evidence already claimed by the outer loop."""
    if not outer_polygon:
        return wall_mask
    band = np.zeros_like(wall_mask)
    pts = np.array(outer_polygon, dtype=np.int32)
    band_thickness = max(int(thickness * 1.5) + 4, 6)
    cv2.polylines(band, [pts], isClosed=True, color=255, thickness=band_thickness)
    remainder = wall_mask.copy()
    remainder[band > 0] = 0
    return remainder


def extract_inner_walls(
    wall_mask: np.ndarray,
    outer_polygon: list[tuple[float, float]],
    thickness: float,
    snap_angle_deg: float = 8.0,
    merge_distance_px: float = 6.0,
    min_wall_length_px: float = 10.0,
    scale_info: ScaleInfo | None = None,
) -> list[WallPrimitive]:
    """Extract interior wall segments from wall evidence not claimed by the outer loop."""
    remainder = _erase_outer_band(wall_mask, outer_polygon, thickness)
    if not remainder.any():
        return []

    skel = _skeletonize_wall(remainder)
    lines = cv2.HoughLinesP(
        skel,
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        minLineLength=int(min_wall_length_px),
        maxLineGap=int(merge_distance_px),
    )
    if lines is None:
        return []

    raw_segments: list[tuple[float, float, float, float]] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        raw_segments.append((float(x1), float(y1), float(x2), float(y2)))

    merged = _merge_collinear_segments(raw_segments, merge_dist=merge_distance_px)

    inner_walls: list[WallPrimitive] = []
    for idx, (x1, y1, x2, y2) in enumerate(merged):
        length = _segment_length(x1, y1, x2, y2)
        if length < min_wall_length_px:
            continue
        wall = WallPrimitive(
            primitive_id=f"wall_inner_{idx:04d}",
            start=(x1, y1),
            end=(x2, y2),
            thickness=thickness,
            confidence=min(1.0, length / 50.0),
            scale_info=scale_info,
        )
        inner_walls.append(wall)

    return inner_walls


def extract_walls(
    wall_mask: np.ndarray,
    room_mask: np.ndarray | None = None,
    snap_angle_deg: float = 8.0,
    merge_distance_px: float = 6.0,
    min_wall_length_px: float = 10.0,
    scale_info: ScaleInfo | None = None,
) -> tuple[list[WallPrimitive], list[WallPrimitive], list[tuple[float, float]]]:
    """Procedurally extract walls: outer rectilinear loop first, then inner walls.

    Returns (outer_walls, inner_walls, outer_polygon).
    """
    if wall_mask is None or not wall_mask.any():
        return [], [], []

    thickness = _estimate_wall_thickness(wall_mask, _skeletonize_wall(wall_mask))

    outer_walls, outer_polygon = extract_outer_wall_loop(
        wall_mask, room_mask, thickness=thickness, scale_info=scale_info
    )
    inner_walls = extract_inner_walls(
        wall_mask,
        outer_polygon,
        thickness,
        snap_angle_deg=snap_angle_deg,
        merge_distance_px=merge_distance_px,
        min_wall_length_px=min_wall_length_px,
        scale_info=scale_info,
    )

    return outer_walls, inner_walls, outer_polygon
