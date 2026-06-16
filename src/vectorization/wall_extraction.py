"""Extract WallPrimitive objects from a cleaned wall binary mask."""

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


def extract_walls(
    wall_mask: np.ndarray,
    snap_angle_deg: float = 8.0,
    merge_distance_px: float = 6.0,
    min_wall_length_px: float = 10.0,
    scale_info: ScaleInfo | None = None,
) -> list[WallPrimitive]:
    if not wall_mask.any():
        return []

    thickness = _estimate_wall_thickness(wall_mask, _skeletonize_wall(wall_mask))
    skel = _skeletonize_wall(wall_mask)

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

    primitives: list[WallPrimitive] = []
    for idx, (x1, y1, x2, y2) in enumerate(merged):
        length = _segment_length(x1, y1, x2, y2)
        if length < min_wall_length_px:
            continue
        angle = _snap_angle(_segment_angle(x1, y1, x2, y2), snap_angle_deg)
        wall = WallPrimitive(
            primitive_id=f"wall_{idx:04d}",
            start=(x1, y1),
            end=(x2, y2),
            thickness=thickness,
            confidence=min(1.0, length / 50.0),
            scale_info=scale_info,
        )
        primitives.append(wall)

    return primitives
