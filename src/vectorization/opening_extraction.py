"""Extract OpeningPrimitive candidates from the opening binary mask."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .primitives import OpeningPrimitive, ScaleInfo, WallPrimitive


def _nearest_wall(
    center: tuple[float, float],
    walls: list[WallPrimitive],
    max_dist: float = 40.0,
) -> WallPrimitive | None:
    if not walls:
        return None
    cx, cy = center
    best_wall = None
    best_dist = max_dist
    for wall in walls:
        x1, y1 = wall.start
        x2, y2 = wall.end
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            dist = math.hypot(cx - x1, cy - y1)
        else:
            t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = math.hypot(cx - proj_x, cy - proj_y)
        if dist < best_dist:
            best_dist = dist
            best_wall = wall
    return best_wall


def extract_openings(
    opening_mask: np.ndarray,
    walls: list[WallPrimitive],
    min_area: int = 8,
    max_wall_dist: float = 40.0,
    scale_info: ScaleInfo | None = None,
) -> list[OpeningPrimitive]:
    if not opening_mask.any():
        return []

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        opening_mask, connectivity=8
    )

    primitives: list[OpeningPrimitive] = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx, cy = centroids[i]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        width = float(max(w, h))
        center = (float(cx), float(cy))

        host_wall = _nearest_wall(center, walls, max_dist=max_wall_dist)
        host_wall_id = host_wall.primitive_id if host_wall else None
        orientation = host_wall.orientation_angle if host_wall else 0.0

        primitives.append(
            OpeningPrimitive(
                primitive_id=f"opening_{i:04d}",
                center=center,
                width=width,
                orientation_angle=orientation,
                host_wall_id=host_wall_id,
                opening_type="generic",
                confidence=1.0 if host_wall else 0.3,
                scale_info=scale_info,
            )
        )

    return primitives
