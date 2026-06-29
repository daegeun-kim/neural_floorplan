"""Align searched points onto shared orthogonal axes (spec_v008 SS11).

Window and door-origin pairs already share an exact axis by construction
(point_detection.py derives both endpoints from the same host-wall
projection), so this module first confirms and re-asserts that axis from
the opening direction (SS11.3), which takes priority over generic
coordinate clustering.

task17 (door-first): red `door_arc` bbox vertices (`wall_door_hinge_point`/
`wall_door_end_point`) are trusted anchors and are never moved by this
module - only `_assert_opening_axis`'s own-pair exactness touches them, and
that is already satisfied by construction (both vertices come from the same
bbox). `wall_point`/`wall_window_point` ("followers") first cluster among
themselves the same distance-only way task16 introduced (no wall-pixel-
corridor gate), then any follower within `axis_alignment_tolerance_mm`
(default `500 mm`, restored from task16's `1000 mm`) of a door anchor on a
given axis is snapped (overridden) onto that anchor's exact value - door
anchors always win, never the other way around.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .graph_types import ComponentRecord, GraphPoint, ValidationIssue

if TYPE_CHECKING:
    from .point_detection import WallSkeletonEdge

DEFAULT_AXIS_ALIGNMENT_TOLERANCE_MM = 500.0
DEFAULT_PX_FALLBACK_TOLERANCE = 6.0
FOLLOWER_POINT_TYPES = ("wall_point", "wall_window_point")
DOOR_ANCHOR_POINT_TYPES = ("wall_door_hinge_point", "wall_door_end_point")


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
    component ids - these are the already-known opening pairs (SS11.3).

    Component ids are assigned independently per mask class (components.py),
    so a window component and a door_arc component can legitimately share
    the same id number - the group key must include the opening category,
    not just the id, or an unrelated window pair and door pair that happen
    to collide on id would be merged into one bogus group of 4.
    """
    groups: dict[tuple, list[GraphPoint]] = {}
    for p in points:
        if p.point_type not in ("wall_window_point", "wall_door_hinge_point", "wall_door_end_point"):
            continue
        category = "window" if p.point_type == "wall_window_point" else "door"
        key = (category, tuple(sorted(p.source_component_ids)))
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


def _same_opening_pair(a: GraphPoint, b: GraphPoint) -> bool:
    """True if ``a``/``b`` are the two endpoints of the same window or
    door-origin opening. ``_assert_opening_axis`` above already fixed their
    one *shared* axis exactly; the broad distance-based clustering below
    must never also merge them on their other axis - that one is the
    opening's own width/length, and forcing it equal would collapse the
    door/window to a single point instead of just aligning it onto the wall.
    """
    if not set(a.source_component_ids) & set(b.source_component_ids):
        return False
    types = {a.point_type, b.point_type}
    return types == {"wall_window_point"} or types == {"wall_door_hinge_point", "wall_door_end_point"}


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


def _cluster_axis(candidates: list[GraphPoint], axis_index: int, tol_px: float) -> None:
    """Merge near-equal coordinates whenever every pair in the resulting
    group is mutually within ``tol_px`` on this one axis (task16: the only
    condition - no corridor/wall-evidence gate, independent of the other
    axis). A candidate only joins a group if it agrees with *every* existing
    member, not just the group's seed point, so a transitive "i matches j,
    j matches k" chain can't silently average together two points that are
    each individually within tolerance of the middle one but not of each
    other.
    """

    def _compatible(pi: GraphPoint, pj: GraphPoint) -> bool:
        if _same_opening_pair(pi, pj):
            return False
        return abs(pi.coordinate[axis_index] - pj.coordinate[axis_index]) <= tol_px

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


def _snap_followers_to_door_anchors(
    followers: list[GraphPoint], door_anchors: list[GraphPoint], axis_index: int, tol_px: float
) -> None:
    """task17: door bbox vertices are trusted anchors - a follower
    (`wall_point`/`wall_window_point`) within ``tol_px`` of a door anchor on
    this one axis snaps (overrides, not averages) onto that anchor's exact
    value. Run after follower-to-follower clustering so a door anchor always
    has the final say over any follower it can reach, never the reverse.
    """
    if not door_anchors:
        return
    for f in followers:
        best_anchor = None
        best_dist = tol_px
        for a in door_anchors:
            dist = abs(f.coordinate[axis_index] - a.coordinate[axis_index])
            if dist <= best_dist:
                best_dist = dist
                best_anchor = a
        if best_anchor is None:
            continue
        coord = list(f.coordinate)
        coord[axis_index] = best_anchor.coordinate[axis_index]
        f.coordinate = (coord[0], coord[1])


def align_points(
    points: list[GraphPoint],
    wall_components: list[ComponentRecord],
    scale_info=None,
    config: Optional[dict] = None,
    wall_skeleton_edges: Optional[list["WallSkeletonEdge"]] = None,
) -> tuple[list[GraphPoint], list[ValidationIssue]]:
    """Align points onto shared orthogonal axes (spec_v008 SS11 / SS7 step 7).

    ``wall_components`` is accepted for call-site compatibility but no
    longer used here (task16 dropped the wall-pixel-corridor gate from
    alignment itself - point_connection.py's separate axis *connection*
    step still uses real wall components for that purpose).

    Mutates and returns the same point objects (a fresh list each pipeline
    run), plus any scale-blocked notices.
    """
    del wall_components
    cfg = config or {}
    tolerance_mm = cfg.get("axis_alignment_tolerance_mm", DEFAULT_AXIS_ALIGNMENT_TOLERANCE_MM)
    px_fallback = cfg.get("px_fallback_tolerance", DEFAULT_PX_FALLBACK_TOLERANCE)

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

    # 1. Opening pairs (window / door) - axis from opening direction, exact.
    # Door hinge/end vertices come straight from the trusted red door_arc
    # bbox (point_detection.py) and are never moved past this point.
    for group in _opening_pair_groups(points):
        _assert_opening_axis(group)

    # 2. Followers (wall_point/wall_window_point) cluster among themselves -
    # same distance-only rule task16 introduced (no wall-pixel-corridor
    # gate), but door anchors never participate here, so they cannot be
    # pulled off their bbox-derived position by averaging with a follower.
    followers = [p for p in points if p.point_type in FOLLOWER_POINT_TYPES]
    door_anchors = [p for p in points if p.point_type in DOOR_ANCHOR_POINT_TYPES]
    _cluster_axis(followers, 0, tol_px)
    _cluster_axis(followers, 1, tol_px)

    # 3. task17: door vertices are trusted anchors - any follower within
    # tolerance of a door anchor on a given axis snaps onto that anchor's
    # exact value, overriding whatever step 2 decided. Anchors always win.
    _snap_followers_to_door_anchors(followers, door_anchors, 0, tol_px)
    _snap_followers_to_door_anchors(followers, door_anchors, 1, tol_px)

    # 4. Step 3 can move only one of a window pair's two points (only the
    # one near a door anchor), breaking their own shared axis - re-assert it
    # once more. Doors are unaffected (already exact and untouched).
    for group in _opening_pair_groups(points):
        _assert_opening_axis(group)

    return points, issues
