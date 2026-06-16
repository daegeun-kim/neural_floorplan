"""Extract RoomPrimitive objects from a cleaned room binary mask."""

from __future__ import annotations

import cv2
import numpy as np

from .primitives import RoomPrimitive, ScaleInfo


def _simplify_contour(contour: np.ndarray, epsilon_factor: float = 0.02) -> list[tuple[float, float]]:
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
    return pts


def extract_rooms(
    room_mask: np.ndarray,
    min_area: int = 100,
    scale_info: ScaleInfo | None = None,
) -> list[RoomPrimitive]:
    if not room_mask.any():
        return []

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        room_mask, connectivity=8
    )

    primitives: list[RoomPrimitive] = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        comp_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        pts = _simplify_contour(contour)
        if len(pts) < 3:
            continue

        confidence = min(1.0, area / 2000.0)
        primitives.append(
            RoomPrimitive(
                primitive_id=f"room_{i:04d}",
                polygon=pts,
                confidence=confidence,
                scale_info=scale_info,
            )
        )

    return primitives
