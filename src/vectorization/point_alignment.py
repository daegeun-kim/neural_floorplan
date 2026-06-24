"""Align searched points onto shared orthogonal axes (spec_v008 SS11).

Window and door-origin pairs already share an exact axis by construction
(point_detection.py derives both endpoints from the same host-wall
projection), so this module's main job for those pairs is to confirm and
re-assert that axis from the opening direction (SS11.3), which takes
priority over generic coordinate clustering. For plain wall points
(1/2/3/4_wall_point) that are not part of an opening pair, this module
clusters near-equal coordinates onto one shared axis only when wall pixel
evidence actually runs between them (a bbox corridor check) - bare
coordinate proximity is never sufficient (SS11.2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .graph_types import ComponentRecord, GraphPoint, ValidationIssue

if TYPE_CHECKING:
    from .point_detection import WallSkeletonEdge

DEFAULT_AXIS_ALIGNMENT_TOLERANCE_MM = 500.0
DEFAULT_PX_FALLBACK_TOLERANCE = 6.0
DEFAULT_CORRIDOR_SLACK_PX = 20.0
DEFAULT_MAX_WALL_CLUSTER_TOLERANCE_PX = 15.0


def _tolerance_px(scale_info, tolerance_mm: float, px_fallback: float) -> tuple[float, bool]:
    """Returns (tolerance_px, scale_blocked)."""
    if (
        scale_info is not None
        and scale_info.px_to_mm is not None
        and scale_info.scale_status in ("resolved", "estimated")
    ):
        return tolerance_mm / scale_info.px_to_mm, False
    return px_fallback, True


def _opening_pair_groups(points: list[GraphPoint]) -> list[list[GraphPoint]]:
    """Group wall_window_point and door hinge/end points sharing source
    component ids - these are the already-known opening pairs (SS11.3)."""
    groups: dict[tuple, list[GraphPoint]] = {}
    for p in points:
        if p.point_type not in ("wall_window_point", "wall_door_hinge_point", "wall_door_end_point"):
            continue
        key = tuple(sorted(p.source_component_ids))
        groups.setdefault(key, []).append(p)
    return list(groups.values())


def _assert_opening_axis(group: list[GraphPoint]) -> None:
    """Re-assert the shared axis implied by the opening attachment direction:
    horizontal opening (left/right) -> shared y; vertical (up/down) -> shared x."""
    if len(group) != 2:
        return
    a, b = group
    a_open = a.attachment_of("window") or a.attachment_of("door_origin")
    if a_open is None:
        return
    axis_index = 1 if a_open.direction in ("left", "right") else 0
    avg = (a.coordinate[axis_index] + b.coordinate[axis_index]) / 2.0
    for p in group:
        coord = list(p.coordinate)
        coord[axis_index] = avg
        p.coordinate = (coord[0], coord[1])


def _assert_wall_edge_axes(points: list[GraphPoint], wall_skeleton_edges: list["WallSkeletonEdge"]) -> None:
    """Snap every wall-skeleton-edge-connected pair of points onto a shared
    axis from the edge's own direction (SS11.1) - the highest-confidence
    alignment evidence there is, since the two points are connected by one
    continuous wall chain. Without this, slightly-noisy raw skeleton pixel
    coordinates flow straight into the final polygon and the buffer/mitre
    step can produce wild non-orthogonal spikes at every joint.
    """
    points_by_id = {p.id: p for p in points}
    for edge in wall_skeleton_edges:
        pa = points_by_id.get(edge.point_id_at_start)
        pb = points_by_id.get(edge.point_id_at_end)
        if pa is None or pb is None or pa is pb:
            continue
        axis_index = 1 if edge.dir_from_start in ("left", "right") else 0
        avg = (pa.coordinate[axis_index] + pb.coordinate[axis_index]) / 2.0
        for p in (pa, pb):
            coord = list(p.coordinate)
            coord[axis_index] = avg
            p.coordinate = (coord[0], coord[1])


def _corridor_has_wall_evidence(
    p: GraphPoint, q: GraphPoint, axis: str, wall_components: list[ComponentRecord], slack_px: float
) -> bool:
    """True if some wall component's bbox runs in the corridor strip between
    p and q along the shared axis (SS11.2: "wall pixels along the axis")."""
    (px, py), (qx, qy) = p.coordinate, q.coordinate
    if axis == "x":
        shared = (px + qx) / 2.0
        lo, hi = sorted((py, qy))
        for comp in wall_components:
            x0, y0, x1, y1 = comp.bbox
            if x0 - slack_px <= shared <= x1 + slack_px and not (y1 < lo or y0 > hi):
                return True
    else:
        shared = (py + qy) / 2.0
        lo, hi = sorted((px, qx))
        for comp in wall_components:
            x0, y0, x1, y1 = comp.bbox
            if y0 - slack_px <= shared <= y1 + slack_px and not (x1 < lo or x0 > hi):
                return True
    return False


def _cluster_axis(
    candidates: list[GraphPoint], axis_index: int, tol_px: float,
    wall_components: list[ComponentRecord], corridor_slack_px: float,
) -> None:
    """Merge near-equal coordinates only when every pair in the resulting
    group is mutually within tolerance and corridor-supported - a transitive
    "i matches j, j matches k" chain through a long, loosely-toleranced wall
    bbox is exactly how unrelated points end up averaged together, so a
    candidate only joins a group if it agrees with *every* existing member,
    not just the group's seed point.
    """
    axis_name = "x" if axis_index == 0 else "y"

    def _compatible(pi: GraphPoint, pj: GraphPoint) -> bool:
        return abs(pi.coordinate[axis_index] - pj.coordinate[axis_index]) <= tol_px and _corridor_has_wall_evidence(
            pi, pj, axis_name, wall_components, corridor_slack_px
        )

    used = [False] * len(candidates)
    for i in range(len(candidates)):
        if used[i]:
            continue
        group = [i]
        for j in range(i + 1, len(candidates)):
            if used[j]:
                continue
            if all(_compatible(candidates[k], candidates[j]) for k in group):
                group.append(j)
        if len(group) < 2:
            continue
        for k in group:
            used[k] = True
        avg = sum(candidates[k].coordinate[axis_index] for k in group) / len(group)
        for k in group:
            p = candidates[k]
            coord = list(p.coordinate)
            coord[axis_index] = avg
            p.coordinate = (coord[0], coord[1])


def align_points(
    points: list[GraphPoint],
    wall_components: list[ComponentRecord],
    scale_info=None,
    config: Optional[dict] = None,
    wall_skeleton_edges: Optional[list["WallSkeletonEdge"]] = None,
) -> tuple[list[GraphPoint], list[ValidationIssue]]:
    """Align points onto shared orthogonal axes (spec_v008 SS11 / SS7 step 7).

    Mutates and returns the same point objects (a fresh list each pipeline
    run), plus any scale-blocked notices.
    """
    cfg = config or {}
    tolerance_mm = cfg.get("axis_alignment_tolerance_mm", DEFAULT_AXIS_ALIGNMENT_TOLERANCE_MM)
    px_fallback = cfg.get("px_fallback_tolerance", DEFAULT_PX_FALLBACK_TOLERANCE)
    corridor_slack_px = cfg.get("corridor_slack_px", DEFAULT_CORRIDOR_SLACK_PX)

    tol_px, scale_blocked = _tolerance_px(scale_info, tolerance_mm, px_fallback)
    issues: list[ValidationIssue] = []
    if scale_blocked:
        issues.append(
            ValidationIssue(
                rule="axis_alignment_scale_blocked",
                message=f"scale unresolved - using px fallback tolerance {px_fallback}px instead of {tolerance_mm}mm",
                severity="info",
            )
        )

    # 0. Wall-skeleton-edge-connected pairs - exact, highest-confidence evidence.
    if wall_skeleton_edges:
        _assert_wall_edge_axes(points, wall_skeleton_edges)

    # 1. Opening pairs (window / door) - axis from opening direction, highest priority.
    for group in _opening_pair_groups(points):
        _assert_opening_axis(group)

    opening_ids = {
        id(p) for p in points
        if p.point_type in ("wall_window_point", "wall_door_hinge_point", "wall_door_end_point")
    }
    wall_only = [p for p in points if id(p) not in opening_ids]

    # 2. Plain wall points - evidence-gated clustering only. Capped well below
    # the full mm-derived tolerance: that figure is meant for confirming an
    # opening's own axis (already handled above with no coordinate-distance
    # check at all), not for blindly merging any two wall points that happen
    # to be within half a meter of each other anywhere in the floor plan.
    cluster_tol_px = min(tol_px, cfg.get("max_wall_cluster_tolerance_px", DEFAULT_MAX_WALL_CLUSTER_TOLERANCE_PX))
    _cluster_axis(wall_only, 0, cluster_tol_px, wall_components, corridor_slack_px)
    _cluster_axis(wall_only, 1, cluster_tol_px, wall_components, corridor_slack_px)

    return points, issues
