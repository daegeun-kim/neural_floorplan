"""Extract WindowPrimitive objects directly from window mask evidence.

The active 7-class CNN already separates `window` from the door classes, so
this module hosts window components on the nearest wall and projects the
component's own pixels onto that wall to find the two transition points
(spec_v008 SS8) - no more generic-opening + heuristic aspect-ratio
classification step.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry_rules import project_pixels_onto_wall, select_host_wall_for_opening
from .primitives import OpeningPrimitive, ScaleInfo, WallPrimitive, WindowPrimitive
from .primitives.scale import WINDOW_MODULES_MM, snap_to_module_mm


def extract_windows(
    window_mask: np.ndarray,
    walls: list[WallPrimitive],
    min_area: int = 8,
    max_wall_dist: float = 40.0,
    min_hosted_width_px: float = 10.0,
    min_confidence_for_metric: float = 0.70,
    scale_info: ScaleInfo | None = None,
    min_width_mm: float = 300.0,
    corner_ambiguity_px: float = 25.0,
    min_remainder_px: float = 3.0,
) -> tuple[list[WindowPrimitive], list[OpeningPrimitive]]:
    """Return (hosted windows, unresolved debug markers for unhosted evidence)."""
    scale_info = scale_info or ScaleInfo()
    if not window_mask.any():
        return [], []

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        window_mask, connectivity=8
    )

    windows: list[WindowPrimitive] = []
    unresolved: list[OpeningPrimitive] = []

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx, cy = centroids[i]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y0 = stats[i, cv2.CC_STAT_TOP]
        center = (float(cx), float(cy))
        bbox_px = (float(x0), float(y0), float(x0 + bw), float(y0 + bh))

        ys, xs = np.where(labels == i)
        pixel_coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

        host_wall = select_host_wall_for_opening(
            pixel_coords, walls, max_dist=max_wall_dist,
            corner_ambiguity_px=corner_ambiguity_px, min_remainder_px=min_remainder_px,
        )
        if host_wall is None:
            unresolved.append(
                OpeningPrimitive(
                    primitive_id=f"window_unresolved_{i:04d}",
                    center=center,
                    width=float(max(bw, bh)),
                    orientation_angle=0.0,
                    host_wall_id=None,
                    opening_type="unresolved_window",
                    confidence=0.3,
                    scale_info=scale_info,
                    source_class_ids=[3],
                    source_evidence_bbox_px=bbox_px,
                    source_evidence_area_px=float(area),
                )
            )
            continue

        hosted_center, width_px, _t_min, _t_max = project_pixels_onto_wall(
            pixel_coords, host_wall
        )

        if width_px < min_hosted_width_px:
            unresolved.append(
                OpeningPrimitive(
                    primitive_id=f"window_too_narrow_{i:04d}",
                    center=hosted_center,
                    width=width_px,
                    orientation_angle=host_wall.orientation_angle,
                    host_wall_id=host_wall.primitive_id,
                    opening_type="unresolved_window",
                    confidence=0.2,
                    scale_info=scale_info,
                    source_class_ids=[3],
                    source_evidence_bbox_px=bbox_px,
                    source_evidence_area_px=float(area),
                )
            )
            continue

        if scale_info.px_to_mm is None or scale_info.scale_status not in ("resolved", "estimated"):
            unresolved.append(
                OpeningPrimitive(
                    primitive_id=f"window_scale_blocked_{i:04d}",
                    center=hosted_center,
                    width=width_px,
                    orientation_angle=host_wall.orientation_angle,
                    host_wall_id=host_wall.primitive_id,
                    opening_type="unresolved_window_scale_blocked",
                    confidence=0.4,
                    scale_info=scale_info,
                    source_class_ids=[3],
                    source_evidence_bbox_px=bbox_px,
                    source_evidence_area_px=float(area),
                )
            )
            continue

        if width_px * scale_info.px_to_mm < min_width_mm:
            unresolved.append(
                OpeningPrimitive(
                    primitive_id=f"window_too_narrow_mm_{i:04d}",
                    center=hosted_center,
                    width=width_px,
                    orientation_angle=host_wall.orientation_angle,
                    host_wall_id=host_wall.primitive_id,
                    opening_type="unresolved_window_too_narrow_mm",
                    confidence=0.2,
                    scale_info=scale_info,
                    source_class_ids=[3],
                    source_evidence_bbox_px=bbox_px,
                    source_evidence_area_px=float(area),
                )
            )
            continue

        width_mm, _ = snap_to_module_mm(
            width_px, scale_info, WINDOW_MODULES_MM, min_confidence_for_metric
        )

        windows.append(
            WindowPrimitive(
                primitive_id=f"window_{i:04d}",
                center=hosted_center,
                width=width_px,
                orientation_angle=host_wall.orientation_angle,
                # Window total width is 100mm vs the wall's 200mm (task09) -
                # exactly half, derived proportionally from the wall's own
                # measured thickness so the px-domain fallback rule holds
                # even when metric scale isn't resolved.
                thickness=host_wall.thickness / 2.0,
                width_mm=width_mm,
                host_wall_id=host_wall.primitive_id,
                confidence=1.0,
                scale_info=scale_info,
                source_class_ids=[3],
                source_evidence_bbox_px=bbox_px,
                source_evidence_area_px=float(area),
            )
        )

    return windows, unresolved
