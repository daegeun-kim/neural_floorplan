"""Host door and window opening candidates onto the aligned wall graph.

Implements spec_v008 §7 step 7-10 (door) and §8 step 4-6 (window).

Non-negotiable rule (spec §14.3):
    Both endpoints of one opening MUST snap to the SAME wall edge or the same
    wall chain interval.  If that is impossible, the opening is rejected.

Algorithm per opening:
    1. For each aligned graph edge, compute the projection distance of both
       raw endpoints onto that edge.
    2. A compatible host edge is one where BOTH projections land within the
       edge extent (t in [0,1]) and the perpendicular distances are small.
    3. Score compatible edges by total proximity; choose the best.
    4. If no single edge satisfies the constraint, reject.
    5. Snap the two projected points onto the host edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .opening_detection import DoorCandidate, WindowCandidate
from ..primitives.scale import DOOR_MODULES_MM, ScaleInfo, snap_to_module_mm


@dataclass
class HostedOpening:
    """An opening successfully hosted on one wall edge."""
    opening_type: str           # "door" or "window"
    source_component_id: int
    host_edge_idx: int          # index into aligned_edges list
    host_edge_raw: list[float]  # [x1, y1, x2, y2]
    raw_points: list[tuple[float, float]]
    snapped_points: list[tuple[float, float]]
    width_px: float
    width_mm: Optional[float]
    confidence: float
    # For doors: which module was snapped to
    snapped_module_mm: Optional[float] = None


@dataclass
class RejectedOpening:
    """An opening that could not be hosted on any single wall edge."""
    opening_type: str
    source_component_id: int
    raw_points: list[tuple[float, float]]
    rejection_reason: str
    debug_confidence: float = 0.0


def _pt_to_segment_dist_and_t(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float]:
    """Perpendicular distance from point to segment, and t in [0,1]."""
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-9:
        return math.hypot(px - x1, py - y1), 0.0
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, t))
    proj_x = x1 + t_clamped * dx
    proj_y = y1 + t_clamped * dy
    return math.hypot(px - proj_x, py - proj_y), t


def _project_onto_edge(
    pt: tuple[float, float], x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float]:
    """Project point onto segment, clamped to segment extent."""
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-9:
        return x1, y1
    t = ((pt[0] - x1) * dx + (pt[1] - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    return x1 + t * dx, y1 + t * dy


def _try_host_on_edge(
    pt_a: tuple[float, float],
    pt_b: tuple[float, float],
    edge: list[float],
    max_perp_dist_px: float = 20.0,
    min_width_px: float = 5.0,
) -> Optional[tuple[tuple[float, float], tuple[float, float], float, float]]:
    """Try to host both opening points onto one edge.

    Returns (snapped_a, snapped_b, avg_perp_dist, width_px) or None if
    either point exceeds max_perp_dist_px or the resulting width is too small.
    """
    x1, y1, x2, y2 = edge[0], edge[1], edge[2], edge[3]
    dist_a, t_a = _pt_to_segment_dist_and_t(pt_a[0], pt_a[1], x1, y1, x2, y2)
    dist_b, t_b = _pt_to_segment_dist_and_t(pt_b[0], pt_b[1], x1, y1, x2, y2)

    if dist_a > max_perp_dist_px or dist_b > max_perp_dist_px:
        return None

    snap_a = _project_onto_edge(pt_a, x1, y1, x2, y2)
    snap_b = _project_onto_edge(pt_b, x1, y1, x2, y2)
    width = math.hypot(snap_b[0] - snap_a[0], snap_b[1] - snap_a[1])
    if width < min_width_px:
        return None

    return snap_a, snap_b, (dist_a + dist_b) / 2.0, width


def host_openings(
    door_candidates: list[DoorCandidate],
    window_candidates: list[WindowCandidate],
    aligned_graph_edges: list[list[float]],
    scale_info: ScaleInfo,
    max_perp_dist_px: float = 20.0,
    min_door_width_px: float = 5.0,
    min_window_width_px: float = 5.0,
    door_modules_mm: tuple[float, ...] = DOOR_MODULES_MM,
    min_window_width_mm: float = 300.0,
) -> tuple[list[HostedOpening], list[RejectedOpening]]:
    """Host all door and window candidates onto the aligned wall graph.

    Returns:
        (hosted, rejected)
    """
    hosted: list[HostedOpening] = []
    rejected: list[RejectedOpening] = []

    def _host_one(
        pt_a: tuple[float, float],
        pt_b: tuple[float, float],
        opening_type: str,
        component_id: int,
        confidence: float,
        min_width_px: float,
    ) -> None:
        best_result = None
        best_edge_idx = -1
        best_avg_dist = float("inf")

        for ei, edge in enumerate(aligned_graph_edges):
            result = _try_host_on_edge(pt_a, pt_b, edge, max_perp_dist_px, min_width_px)
            if result is not None:
                snap_a, snap_b, avg_dist, width = result
                if avg_dist < best_avg_dist:
                    best_avg_dist = avg_dist
                    best_result = (snap_a, snap_b, width)
                    best_edge_idx = ei

        if best_result is None:
            rejected.append(RejectedOpening(
                opening_type=opening_type,
                source_component_id=component_id,
                raw_points=[pt_a, pt_b],
                rejection_reason=(
                    "no single wall edge could host both endpoints "
                    f"within {max_perp_dist_px:.0f}px"
                ),
                debug_confidence=confidence,
            ))
            return

        snap_a, snap_b, width_px = best_result
        host_edge = aligned_graph_edges[best_edge_idx]

        # Scale to mm if resolved
        width_mm: Optional[float] = None
        snapped_module: Optional[float] = None
        if scale_info.px_to_mm is not None and scale_info.scale_status in ("resolved", "estimated"):
            width_mm = width_px * scale_info.px_to_mm
            if opening_type == "door":
                # Snap door width to nearest allowed module
                mod, _orig_px = snap_to_module_mm(width_px, scale_info, door_modules_mm)
                snapped_module = mod
                if mod is not None:
                    # Adjust snapped_b along the host edge so width matches the module
                    snapped_width_px = mod / scale_info.px_to_mm
                    ex1, ey1, ex2, ey2 = host_edge
                    seg_len = math.hypot(ex2 - ex1, ey2 - ey1)
                    if seg_len > 1e-6:
                        dir_x = (ex2 - ex1) / seg_len
                        dir_y = (ey2 - ey1) / seg_len
                        # Keep snap_a fixed, adjust snap_b
                        snap_b = (snap_a[0] + dir_x * snapped_width_px,
                                  snap_a[1] + dir_y * snapped_width_px)
                        # Re-project snap_b to keep it on the edge
                        snap_b = _project_onto_edge(snap_b, ex1, ey1, ex2, ey2)
                        width_px = math.hypot(snap_b[0] - snap_a[0], snap_b[1] - snap_a[1])
                    width_mm = mod

            # Window minimum width check
            if opening_type == "window" and width_mm is not None and width_mm < min_window_width_mm:
                rejected.append(RejectedOpening(
                    opening_type=opening_type,
                    source_component_id=component_id,
                    raw_points=[pt_a, pt_b],
                    rejection_reason=f"window width {width_mm:.0f}mm < min {min_window_width_mm:.0f}mm",
                    debug_confidence=confidence,
                ))
                return

        hosted.append(HostedOpening(
            opening_type=opening_type,
            source_component_id=component_id,
            host_edge_idx=best_edge_idx,
            host_edge_raw=host_edge,
            raw_points=[pt_a, pt_b],
            snapped_points=[snap_a, snap_b],
            width_px=width_px,
            width_mm=width_mm,
            confidence=confidence,
            snapped_module_mm=snapped_module,
        ))

    # Host doors
    for door in door_candidates:
        if len(door.raw_points) < 2:
            rejected.append(RejectedOpening(
                opening_type="door",
                source_component_id=door.component_id,
                raw_points=door.raw_points,
                rejection_reason="no raw points from detection",
                debug_confidence=0.0,
            ))
            continue
        _host_one(
            door.raw_points[0], door.raw_points[1],
            "door", door.component_id, door.confidence, min_door_width_px,
        )

    # Host windows
    for win in window_candidates:
        if len(win.raw_points) < 2:
            rejected.append(RejectedOpening(
                opening_type="window",
                source_component_id=win.component_id,
                raw_points=win.raw_points,
                rejection_reason="no raw points from detection",
                debug_confidence=0.0,
            ))
            continue
        _host_one(
            win.raw_points[0], win.raw_points[1],
            "window", win.component_id, win.confidence, min_window_width_px,
        )

    return hosted, rejected
