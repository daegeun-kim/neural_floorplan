"""Extract IconPrimitive objects from a cleaned icon binary mask."""

from __future__ import annotations

import cv2
import numpy as np

from .primitives import IconPrimitive, ScaleInfo


def _simplify_contour(contour: np.ndarray, epsilon_factor: float = 0.02) -> list[tuple[float, float]]:
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
    return pts


def extract_icons(
    icon_mask: np.ndarray,
    min_area: int = 20,
    scale_info: ScaleInfo | None = None,
) -> list[IconPrimitive]:
    """Extract simplified filled icon/furniture shapes from the icon mask."""
    if icon_mask is None or not icon_mask.any():
        return []

    n, labels, stats, _ = cv2.connectedComponentsWithStats(icon_mask, connectivity=8)

    primitives: list[IconPrimitive] = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        comp_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        pts = _simplify_contour(contour)
        if len(pts) < 3:
            continue

        confidence = min(1.0, area / 500.0)
        primitives.append(
            IconPrimitive(
                primitive_id=f"icon_{i:04d}",
                polygon=pts,
                confidence=confidence,
                scale_info=scale_info,
            )
        )

    return primitives
