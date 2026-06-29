"""Connected-component extraction and metadata for the v008 point-graph
pipeline (spec_v008 SS6/SS8).

Each class mask is lightly cleaned (small-noise removal, plus a touch of
morphological closing only where it cannot corrupt a measurable length),
then split into connected components. Components below the configured area
threshold are dropped and recorded as ``RejectedEvidence`` instead of being
silently discarded, per spec_v008 SS8: "Remove components below configured
area thresholds, but record removed components in metrics."

Cleanup policy per class (ported from the retired cleanup.py, unchanged):
wall and door_arc tolerate light closing (rect/ellipse kernel) to bridge
segmentation speckle; window, door_leaf, and door_origin must not be closed -
door_origin in particular is a thin stroked line whose length is the
evidence scale.py clusters against the 700/900mm door modules, so closing it
would corrupt that measurement.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .graph_types import ComponentRecord, RejectedEvidence

DEFAULT_MIN_AREA_PX: dict[str, float] = {
    "wall": 4.0,
    "window": 4.0,
    "door_arc": 4.0,
    "door_leaf": 2.0,
    "door_origin": 2.0,
}

_CLEAN_CONFIG: dict[str, dict] = {
    "wall": {"close_gap_px": 3, "shape": cv2.MORPH_RECT},
    "door_arc": {"close_gap_px": 3, "shape": cv2.MORPH_ELLIPSE},
    "window": {"close_gap_px": 0},
    "door_leaf": {"close_gap_px": 0},
    "door_origin": {"close_gap_px": 0},
}


def clean_class_mask(
    mask: np.ndarray, class_name: str, close_gap_px: int | None = None
) -> np.ndarray:
    cfg = _CLEAN_CONFIG.get(class_name, {})
    gap = close_gap_px if close_gap_px is not None else cfg.get("close_gap_px", 0)
    if not gap or gap <= 1:
        return mask
    shape = cfg.get("shape", cv2.MORPH_RECT)
    kernel = cv2.getStructuringElement(shape, (gap, gap))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _skeleton_and_endpoints(
    component_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Skeletonize one component and find its degree<=1 endpoint pixels.

    point_detection.py walks this skeleton to build the wall point graph
    (spec_v008 SS9.1): degree-1 pixels are candidate node endpoints, degree>=3
    pixels are candidate junctions.
    """
    skel = skeletonize(component_mask > 0)
    ys, xs = np.nonzero(skel)
    skeleton_points = list(zip(xs.tolist(), ys.tolist()))
    if not skeleton_points:
        return [], []
    skel_set = set(skeleton_points)
    endpoints = []
    for x, y in skeleton_points:
        degree = sum(
            (x + dx, y + dy) in skel_set
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
        )
        if degree <= 1:
            endpoints.append((x, y))
    return skeleton_points, endpoints


def extract_components(
    mask: np.ndarray,
    class_name: str,
    min_area_px: float = 4.0,
    close_gap_px: int | None = None,
    compute_skeleton: bool | None = None,
) -> tuple[list[ComponentRecord], list[RejectedEvidence]]:
    """Clean one class mask and extract its connected components.

    ``compute_skeleton`` defaults to True only for "wall", since that is the
    only class point_detection.py walks as a skeleton graph - window and
    door evidence are instead projected onto host wall topology directly.
    """
    cleaned = clean_class_mask(mask, class_name, close_gap_px)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    if compute_skeleton is None:
        compute_skeleton = class_name == "wall"

    components: list[ComponentRecord] = []
    rejected: list[RejectedEvidence] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        bbox = (x0, y0, x0 + w, y0 + h)
        cx, cy = centroids[i]

        if area < min_area_px:
            rejected.append(
                RejectedEvidence(
                    kind=f"{class_name}_component_too_small",
                    reason=f"area {area:.1f}px < min {min_area_px:.1f}px",
                    class_name=class_name,
                    bbox=bbox,
                    centroid=(float(cx), float(cy)),
                    component_id=i,
                )
            )
            continue

        component_mask = (labels == i).astype(np.uint8) * 255
        ys, xs = np.nonzero(component_mask)
        pts = np.column_stack([xs, ys]).astype(np.float32)

        rect_size = None
        rect_angle = None
        if len(pts) >= 2:
            (_rcx, _rcy), (rw, rh), rangle = cv2.minAreaRect(pts)
            rect_size = (float(max(rw, rh)), float(min(rw, rh)))
            rect_angle = float(rangle)

        skeleton_points: list[tuple[int, int]] = []
        endpoints: list[tuple[int, int]] = []
        if compute_skeleton:
            skeleton_points, endpoints = _skeleton_and_endpoints(component_mask)

        components.append(
            ComponentRecord(
                class_name=class_name,
                component_id=i,
                area_px=area,
                bbox=bbox,
                centroid=(float(cx), float(cy)),
                rect_size=rect_size,
                rect_angle=rect_angle,
                skeleton_points=skeleton_points,
                endpoints=endpoints,
                mask=component_mask,
            )
        )

    return components, rejected


def extract_all_components(
    masks: dict[str, np.ndarray],
    min_area_px: dict[str, float] | None = None,
    close_gap_px: dict[str, int] | None = None,
) -> tuple[dict[str, list[ComponentRecord]], list[RejectedEvidence]]:
    """Extract components for every class present in ``masks`` (spec_v008 SS8).

    Callers exclude "floor" before calling this for the v008 restart - floor
    is ignored entirely (spec_v008 SS1/SS2).
    """
    min_area_px = min_area_px or {}
    close_gap_px = close_gap_px or {}
    components: dict[str, list[ComponentRecord]] = {}
    rejected: list[RejectedEvidence] = []
    for class_name, mask in masks.items():
        min_area = min_area_px.get(class_name, DEFAULT_MIN_AREA_PX.get(class_name, 4.0))
        comps, rej = extract_components(
            mask,
            class_name,
            min_area_px=min_area,
            close_gap_px=close_gap_px.get(class_name),
        )
        components[class_name] = comps
        rejected.extend(rej)
    return components, rejected
