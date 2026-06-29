"""Detect door and window opening candidates from segmentation components.

Implements spec_v008 §7 (Door Generation) and §8 (Window Generation).

For doors:
    - Use red door_arc connected-component bbox
    - Identify the wall-facing bbox edge by proximity to aligned R2G graph
    - Two adjacent vertices on the wall-facing edge are the raw door points

For windows:
    - Use blue window connected-component
    - Estimate major axis via minAreaRect
    - Two endpoints along the major axis are the raw window points
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..graph_types import ComponentRecord


@dataclass
class DoorCandidate:
    """Raw door candidate before graph hosting."""
    component_id: int
    bbox: tuple[int, int, int, int]       # x0, y0, x1, y1
    bbox_long_edge_px: float
    raw_points: list[tuple[float, float]]  # the two wall-facing bbox vertices
    wall_facing_edge: str                  # "top","bottom","left","right"
    confidence: float = 1.0
    rejection_reason: str = ""


@dataclass
class WindowCandidate:
    """Raw window candidate before graph hosting."""
    component_id: int
    bbox: tuple[int, int, int, int]
    raw_points: list[tuple[float, float]]  # two major-axis endpoints
    major_axis_px: float
    confidence: float = 1.0
    rejection_reason: str = ""


def _bbox_edges(bbox: tuple[int, int, int, int]) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Return the 4 bbox edges as {name: (pt_a, pt_b)}."""
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return {
        "top":    ((float(x0), float(y0)), (float(x1), float(y0))),
        "bottom": ((float(x0), float(y1)), (float(x1), float(y1))),
        "left":   ((float(x0), float(y0)), (float(x0), float(y1))),
        "right":  ((float(x1), float(y0)), (float(x1), float(y1))),
    }


def _min_dist_edge_to_graph(
    pt_a: tuple[float, float],
    pt_b: tuple[float, float],
    graph_edges: list[list[float]],
) -> float:
    """Minimum distance from the midpoint of a bbox edge to any graph edge."""
    mx = (pt_a[0] + pt_b[0]) / 2.0
    my = (pt_a[1] + pt_b[1]) / 2.0
    if not graph_edges:
        return float("inf")
    best = float("inf")
    for ex1, ey1, ex2, ey2 in graph_edges:
        # distance from midpoint to segment
        dx, dy = ex2 - ex1, ey2 - ey1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-9:
            d = math.hypot(mx - ex1, my - ey1)
        else:
            t = max(0.0, min(1.0, ((mx - ex1) * dx + (my - ey1) * dy) / seg_len_sq))
            px, py = ex1 + t * dx, ey1 + t * dy
            d = math.hypot(mx - px, my - py)
        if d < best:
            best = d
    return best


def detect_door_candidates(
    door_arc_components: list[ComponentRecord],
    aligned_graph_edges: list[list[float]],
    max_bbox_aspect_ratio: float = 2.0,
    min_area_px: float = 4.0,
) -> tuple[list[DoorCandidate], list[DoorCandidate]]:
    """Detect raw door candidates from door_arc segmentation components.

    Returns:
        (accepted, rejected) lists of DoorCandidate
    """
    accepted: list[DoorCandidate] = []
    rejected: list[DoorCandidate] = []

    for comp in door_arc_components:
        x0, y0, x1, y1 = comp.bbox
        w, h = x1 - x0, y1 - y0
        long_edge = float(max(w, h))
        short_edge = float(min(w, h))

        candidate = DoorCandidate(
            component_id=comp.component_id,
            bbox=comp.bbox,
            bbox_long_edge_px=long_edge,
            raw_points=[],
            wall_facing_edge="",
        )

        # Reject implausible aspect ratios
        if short_edge < 1e-3 or long_edge / short_edge > max_bbox_aspect_ratio:
            candidate.rejection_reason = (
                f"bbox aspect ratio {long_edge/max(short_edge,1e-3):.2f} > {max_bbox_aspect_ratio}"
            )
            rejected.append(candidate)
            continue

        # Identify wall-facing bbox edge: the one closest to the aligned graph
        edges = _bbox_edges(comp.bbox)
        best_edge_name = min(
            edges.keys(),
            key=lambda name: _min_dist_edge_to_graph(edges[name][0], edges[name][1], aligned_graph_edges),
        )
        pt_a, pt_b = edges[best_edge_name]
        candidate.raw_points = [pt_a, pt_b]
        candidate.wall_facing_edge = best_edge_name
        accepted.append(candidate)

    return accepted, rejected


def detect_window_candidates(
    window_components: list[ComponentRecord],
    min_major_axis_px: float = 5.0,
) -> tuple[list[WindowCandidate], list[WindowCandidate]]:
    """Detect raw window candidates from window segmentation components.

    Returns:
        (accepted, rejected) lists of WindowCandidate
    """
    accepted: list[WindowCandidate] = []
    rejected: list[WindowCandidate] = []

    for comp in window_components:
        if comp.mask is None:
            continue

        ys, xs = np.nonzero(comp.mask)
        if len(xs) < 2:
            rejected.append(WindowCandidate(
                component_id=comp.component_id,
                bbox=comp.bbox,
                raw_points=[],
                major_axis_px=0.0,
                rejection_reason="too few pixels for axis estimation",
            ))
            continue

        pts = np.column_stack([xs, ys]).astype(np.float32)
        _center, (rw, rh), angle = cv2.minAreaRect(pts)
        major_len = float(max(rw, rh))
        minor_len = float(min(rw, rh))

        if major_len < min_major_axis_px:
            rejected.append(WindowCandidate(
                component_id=comp.component_id,
                bbox=comp.bbox,
                raw_points=[],
                major_axis_px=major_len,
                rejection_reason=f"major axis {major_len:.1f}px < {min_major_axis_px}px",
            ))
            continue

        # The major axis direction
        # minAreaRect angle is the angle of the longest side from x-axis.
        # When rw >= rh, the angle is along rw (horizontal-ish direction).
        if rw >= rh:
            angle_rad = math.radians(angle)
        else:
            angle_rad = math.radians(angle + 90.0)

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        half = major_len / 2.0
        dx = math.cos(angle_rad) * half
        dy = math.sin(angle_rad) * half

        pt_a = (cx - dx, cy - dy)
        pt_b = (cx + dx, cy + dy)

        accepted.append(WindowCandidate(
            component_id=comp.component_id,
            bbox=comp.bbox,
            raw_points=[pt_a, pt_b],
            major_axis_px=major_len,
        ))

    return accepted, rejected
