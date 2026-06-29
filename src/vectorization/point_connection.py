"""Connect aligned points into wall/window/door-origin graph edges
(spec_v008 SS12).

Wall edges are reconstructed directly from the wall skeleton edges
point_detection.py already discovered: each skeleton chain connected two
node coordinates that have now resolved into final points (a junction, a
free end, or a window/door point that superseded a free end), so every
skeleton chain becomes exactly one wall ``GraphEdge`` between whichever
final points now sit at its two ends. Window and door-origin edges connect
the already-paired ``wall_window_point`` / hinge+end points point_detection
produced - no further metric validation is needed here since both were
already validated (300mm minimum, 700/900mm snap) before being created as
points.
"""

from __future__ import annotations

import math
from typing import Optional

from .graph_types import Attachment, ComponentRecord, GraphEdge, GraphPoint, OPPOSITE_DIRECTION, ValidationIssue
from .point_alignment import _corridor_has_wall_evidence
from .point_detection import WallSkeletonEdge, _project_point_onto_line

DEFAULT_NODE_MATCH_TOLERANCE_PX = 12.0
DEFAULT_CORRIDOR_SLACK_PX = 20.0
OPENING_POINT_TYPES = ("wall_window_point", "wall_door_hinge_point", "wall_door_end_point")
CONNECTABLE_WALL_TYPES = ("wall_point", "wall_window_point", "wall_door_hinge_point", "wall_door_end_point")


def _nearest_point(points: list[GraphPoint], coord: tuple[float, float], tol_px: float) -> Optional[GraphPoint]:
    """Fallback only - used when a skeleton edge wasn't linked to a point id
    (e.g. in hand-built test fixtures). Prefer point_id_at_start/end, which
    are exact and immune to point_alignment.py having since moved the point
    by more than a tight pixel tolerance."""
    best = None
    best_dist = tol_px
    for p in points:
        dist = math.hypot(p.coordinate[0] - coord[0], p.coordinate[1] - coord[1])
        if dist <= best_dist:
            best_dist = dist
            best = p
    return best


def _local_pair_direction(left: GraphPoint, right: GraphPoint) -> Optional[str]:
    """Cardinal direction from ``left`` to ``right``, by their actual
    coordinates - used instead of a shared skeleton edge's own overall
    ``dir_from_start``/``dir_from_end`` labels when chaining 3+ stops along
    one edge. Those labels only describe the relationship at the edge's
    true two endpoints; a host point inserted beyond either end (e.g. an
    opening point whose real boundary lands past the wall mask's own noisy
    edge) sits *outside* that simple two-label relationship, so checking
    every consecutive pair against the same two labels mis-rejects (or, in
    principle, could mis-accept) pairs that aren't the edge's true ends.
    """
    dx = right.coordinate[0] - left.coordinate[0]
    dy = right.coordinate[1] - left.coordinate[1]
    if abs(dx) >= abs(dy):
        if abs(dx) < 1e-6:
            return None
        return "right" if dx > 0 else "left"
    if abs(dy) < 1e-6:
        return None
    return "down" if dy > 0 else "up"


def _wall_direction_attachment(point: GraphPoint, direction: str):
    for a in point.attachments:
        if a.type == "wall" and a.direction == direction:
            return a
    return None


def _same_opening_pair(a: GraphPoint, b: GraphPoint) -> bool:
    """True if ``a``/``b`` are the two endpoints of the same window or
    door-origin opening - such a pair must never get a parallel ``wall``
    edge bridging them, since the opening already replaces the wall
    interval between them (rules 76-78/82-83). Without this guard, once both
    endpoints land on the same aligned axis (the normal case - they are the
    two ends of one straight opening span) ``_connect_axis_aligned_points``
    would otherwise see them as an ordinary same-axis, evidence-backed wall
    pair and bridge them, duplicating the opening with a solid wall edge
    across its own gap.
    """
    if not set(a.source_component_ids) & set(b.source_component_ids):
        return False
    types = {a.point_type, b.point_type}
    return types == {"wall_window_point"} or types == {"wall_door_hinge_point", "wall_door_end_point"}


def _geometric_interior_opening_points(
    se: WallSkeletonEdge, opening_points: list[GraphPoint], exclude_ids: set[str], tol_px: float,
) -> list[tuple[float, GraphPoint]]:
    """Legacy fallback for opening points with no recorded
    ``host_wall_edge_id`` (e.g. hand-built test fixtures): window/door
    points whose projected location falls strictly between ``se.start`` and
    ``se.end``, found by geometry instead of the exact host-edge reference.
    """
    seg_dx, seg_dy = se.end[0] - se.start[0], se.end[1] - se.start[1]
    seg_len = math.hypot(seg_dx, seg_dy)
    if seg_len < 1e-6:
        return []
    found: list[tuple[float, GraphPoint]] = []
    for p in opening_points:
        if p.id in exclude_ids or p.host_wall_edge_id is not None:
            continue
        _proj, dist, t = _project_point_onto_line(p.coordinate, se.start, se.end)
        # Perpendicular distance must use a tight, fixed tolerance, not
        # se.thickness/2 - that field is the *whole wall component's*
        # overall minAreaRect short axis (components.py), which can be huge
        # for a large/complex multi-segment component and would otherwise
        # wrongly pull in unrelated points from other doors/windows far down
        # the same component.
        if 0.0 < t < 1.0 and dist <= tol_px:
            found.append((t * seg_len, p))
    return found


def build_wall_edges(
    points: list[GraphPoint],
    wall_skeleton_edges: list[WallSkeletonEdge],
    tol_px: float = DEFAULT_NODE_MATCH_TOLERANCE_PX,
    opening_match_tolerance_px: float = DEFAULT_NODE_MATCH_TOLERANCE_PX,
) -> list[GraphEdge]:
    """Wall GraphEdges from each skeleton chain (spec_v008 SS12.1), split at
    any window/door point hosted on it so the opening replaces the wall
    interval at its location instead of being spanned by one uninterrupted
    wall edge (rules 77/78, 82/83).

    Every stop along a chain - its own natural skeleton endpoints plus every
    window/door point point_detection.py recorded as hosted on this exact
    edge (``host_wall_edge_id``) - is ordered by position along the chain's
    line and connected to its neighbors. A window/door's real boundary
    commonly lands beyond the chain's own noisy skeleton-pixel endpoint (the
    wall mask doesn't extend all the way to the opening) - using the
    recorded host edge id rather than a fixed pixel tolerance means that
    connection is never missed regardless of how large that gap is (rules
    76/104/105: no floating windows/doors). ``opening_match_tolerance_px``
    and the geometric fallback below remain for points with no recorded host
    edge (hand-built test fixtures).
    """
    edges: list[GraphEdge] = []
    seen_pairs: set[frozenset] = set()
    counter = 0
    points_by_id = {p.id: p for p in points}
    opening_points = [p for p in points if p.point_type in OPENING_POINT_TYPES]
    host_points_by_edge: dict[str, list[GraphPoint]] = {}
    for p in opening_points:
        if p.host_wall_edge_id is not None:
            host_points_by_edge.setdefault(p.host_wall_edge_id, []).append(p)

    for se in wall_skeleton_edges:
        pa = points_by_id.get(se.point_id_at_start) if se.point_id_at_start else None
        pb = points_by_id.get(se.point_id_at_end) if se.point_id_at_end else None
        if pa is None:
            pa = _nearest_point(points, se.start, tol_px) or _nearest_point(opening_points, se.start, opening_match_tolerance_px)
        if pb is None:
            pb = _nearest_point(points, se.end, tol_px) or _nearest_point(opening_points, se.end, opening_match_tolerance_px)

        seg_dx, seg_dy = se.end[0] - se.start[0], se.end[1] - se.start[1]
        seg_len = math.hypot(seg_dx, seg_dy)

        stops: list[tuple[float, GraphPoint]] = []
        seen_ids: set[str] = set()
        if pa is not None:
            stops.append((0.0, pa))
            seen_ids.add(pa.id)
        if pb is not None and pb.id not in seen_ids:
            stops.append((seg_len, pb))
            seen_ids.add(pb.id)
        for p in host_points_by_edge.get(se.id, []):
            if p.id in seen_ids:
                continue
            if seg_len < 1e-6:
                t_abs = 0.0
            else:
                _proj, _dist, t = _project_point_onto_line(p.coordinate, se.start, se.end)
                t_abs = t * seg_len
            stops.append((t_abs, p))
            seen_ids.add(p.id)
        for t_abs, p in _geometric_interior_opening_points(se, opening_points, seen_ids, tol_px):
            stops.append((t_abs, p))
            seen_ids.add(p.id)

        if len(stops) < 2:
            continue
        stops.sort(key=lambda item: item[0])

        for (left_t, left), (right_t, right) in zip(stops[:-1], stops[1:]):
            if left.id == right.id:
                continue
            if 0.0 <= left_t <= seg_len and 0.0 <= right_t <= seg_len:
                # Both stops fall within the chain's own true extent - this is
                # the common case (including the chain's real two endpoints),
                # checked the well-tested way against the edge's own overall
                # direction labels.
                left_dir, right_dir = se.dir_from_start, se.dir_from_end
            else:
                # At least one stop is a host point extrapolated beyond the
                # edge's real extent (e.g. an opening whose true boundary
                # lands past the wall mask's own noisy edge - see module
                # docstring). se.dir_from_start/dir_from_end only describe
                # the relationship at the edge's true two endpoints, not an
                # extrapolated pair, so derive the direction from this pair's
                # own coordinates instead.
                local_dir = _local_pair_direction(left, right)
                if local_dir is None:
                    continue
                left_dir, right_dir = local_dir, OPPOSITE_DIRECTION[local_dir]
            if _wall_direction_attachment(left, left_dir) is None or _wall_direction_attachment(right, right_dir) is None:
                # The point's wall attachment doesn't face this chain - alignment
                # must have shifted something incompatible; skip rather than
                # connect a wall edge with no supporting attachment evidence.
                continue

            key = frozenset((left.id, right.id))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            counter += 1
            edges.append(
                GraphEdge(
                    id=f"walledge_final_{counter}",
                    edge_type="wall",
                    point_a_id=left.id,
                    point_b_id=right.id,
                    start=left.coordinate,
                    end=right.coordinate,
                    source_component_ids=[se.component_id],
                    thickness_px=se.thickness,
                )
            )
    return edges


def _existing_wall_neighbor_directions(
    point_id: str, edges: list[GraphEdge], points_by_id: dict[str, GraphPoint]
) -> set[str]:
    """Cardinal directions in which ``point_id`` already has a wall edge -
    used instead of the point's own pre-existing ``Attachment`` list, since a
    true free end's only recorded attachment points back into its own chain,
    never into a gap that might still need bridging."""
    p = points_by_id.get(point_id)
    if p is None:
        return set()
    dirs: set[str] = set()
    for e in edges:
        if e.edge_type != "wall":
            continue
        if e.point_a_id == point_id:
            other_id = e.point_b_id
        elif e.point_b_id == point_id:
            other_id = e.point_a_id
        else:
            continue
        other = points_by_id.get(other_id)
        if other is None:
            continue
        dx, dy = other.coordinate[0] - p.coordinate[0], other.coordinate[1] - p.coordinate[1]
        if abs(dx) >= abs(dy):
            dirs.add("right" if dx > 0 else "left")
        else:
            dirs.add("down" if dy > 0 else "up")
    return dirs


def _connect_axis_aligned_points(
    points: list[GraphPoint],
    existing_wall_edges: list[GraphEdge],
    wall_components: list[ComponentRecord],
    corridor_slack_px: float,
) -> list[GraphEdge]:
    """task15 problem 1: after axis alignment, connect any two
    wall-participating points (generic ``wall_point``, ``wall_window_point``,
    ``wall_door_hinge_point``, ``wall_door_end_point``) that share an exact
    aligned axis and have real wall pixel evidence between them, even when
    they were never part of the same originally-discovered skeleton chain -
    e.g. a free end on one side of a noisy mask break and a junction/free end
    on the other, which ``build_wall_edges`` above can never connect since it
    only ever walks one pre-built ``WallSkeletonEdge`` chain at a time.

    Connecting strictly *exact*-axis, corridor-evidence-backed, consecutive
    (no skip-over) pairs implements requirement 2's "if two points are not
    aligned onto the same axis, do not connect them to form a wall on that
    axis" - there is no separate "close enough" connection path here.
    """
    points_by_id = {p.id: p for p in points}
    candidates = [p for p in points if p.point_type in CONNECTABLE_WALL_TYPES]
    edges = list(existing_wall_edges)
    connected_pairs = {frozenset((e.point_a_id, e.point_b_id)) for e in edges}

    new_edges: list[GraphEdge] = []
    counter = 0
    for axis_index in (0, 1):
        axis_name = "x" if axis_index == 0 else "y"
        other_index = 1 - axis_index
        forward = "down" if axis_index == 0 else "right"
        backward = "up" if axis_index == 0 else "left"

        groups: dict[float, list[GraphPoint]] = {}
        for p in candidates:
            groups.setdefault(p.coordinate[axis_index], []).append(p)

        for group in groups.values():
            if len(group) < 2:
                continue
            ordered = sorted(group, key=lambda p: p.coordinate[other_index])
            for left, right in zip(ordered[:-1], ordered[1:]):
                if left.id == right.id:
                    continue
                if _same_opening_pair(left, right):
                    continue
                key = frozenset((left.id, right.id))
                if key in connected_pairs:
                    continue
                if forward in _existing_wall_neighbor_directions(left.id, edges, points_by_id):
                    continue
                if backward in _existing_wall_neighbor_directions(right.id, edges, points_by_id):
                    continue
                if not _corridor_has_wall_evidence(left, right, axis_name, wall_components, corridor_slack_px):
                    continue

                left.attachments.append(Attachment(type="wall", direction=forward, source="wall"))
                right.attachments.append(Attachment(type="wall", direction=backward, source="wall"))
                counter += 1
                new_edge = GraphEdge(
                    id=f"walledge_axis_{counter}",
                    edge_type="wall",
                    point_a_id=left.id,
                    point_b_id=right.id,
                    start=left.coordinate,
                    end=right.coordinate,
                    source_component_ids=sorted(set(left.source_component_ids) | set(right.source_component_ids)),
                )
                new_edges.append(new_edge)
                edges.append(new_edge)
                connected_pairs.add(key)

    return new_edges


def _opening_edges(points: list[GraphPoint], edge_type: str, point_types: set[str]) -> list[GraphEdge]:
    groups: dict[tuple, list[GraphPoint]] = {}
    for p in points:
        if p.point_type not in point_types:
            continue
        key = tuple(sorted(p.source_component_ids))
        groups.setdefault(key, []).append(p)

    attachment_type = "window" if edge_type == "window" else "door_origin"
    edges: list[GraphEdge] = []
    counter = 0
    for key, group in groups.items():
        if len(group) != 2:
            continue
        a, b = group
        thickness = None
        host_attachment = a.attachment_of(attachment_type)
        if host_attachment is not None:
            thickness = host_attachment.host_thickness_px
        counter += 1
        edges.append(
            GraphEdge(
                id=f"{edge_type}edge_{counter}",
                edge_type=edge_type,
                point_a_id=a.id,
                point_b_id=b.id,
                start=a.coordinate,
                end=b.coordinate,
                source_component_ids=list(key),
                thickness_px=thickness,
            )
        )
    return edges


def connect_points(
    points: list[GraphPoint],
    wall_skeleton_edges: list[WallSkeletonEdge],
    scale_info=None,
    config: Optional[dict] = None,
    wall_components: Optional[list[ComponentRecord]] = None,
) -> tuple[list[GraphEdge], list[ValidationIssue]]:
    """Build the final wall/window/door-origin graph (spec_v008 SS12 / SS7 step 8)."""
    cfg = config or {}
    tol_px = cfg.get("node_match_tolerance_px", DEFAULT_NODE_MATCH_TOLERANCE_PX)
    opening_match_tolerance_px = cfg.get("opening_match_tolerance_px", tol_px)
    corridor_slack_px = cfg.get("corridor_slack_px", DEFAULT_CORRIDOR_SLACK_PX)

    wall_edges = build_wall_edges(points, wall_skeleton_edges, tol_px, opening_match_tolerance_px)
    wall_edges = wall_edges + _connect_axis_aligned_points(points, wall_edges, wall_components or [], corridor_slack_px)
    window_edges = _opening_edges(points, "window", {"wall_window_point"})
    door_edges = _opening_edges(points, "door_origin", {"wall_door_hinge_point", "wall_door_end_point"})
    edges = wall_edges + window_edges + door_edges

    if scale_info is not None and scale_info.px_to_mm is not None and scale_info.scale_status in ("resolved", "estimated"):
        for e in edges:
            e.length_mm = e.length_px * scale_info.px_to_mm

    return edges, validate_graph(points, edges)


def validate_graph(points: list[GraphPoint], edges: list[GraphEdge]) -> list[ValidationIssue]:
    """Enforce spec_v008 SS10/SS17 graph-level invariants.

    A window/door point's *own* opening edge (window or door_origin) must
    appear exactly once - zero means orphaned, more than one is a real
    conflict. Separately, rules 75-78/82-83/104/105 require that point also
    be hosted on wall topology (the adjacent wall interval it replaces): zero
    wall edges touching it means it is floating, not actually connected to
    the wall graph. A healthy window/door point normally has exactly one
    wall edge *and* one opening edge - that is not a conflict, unlike the
    previous flat-coverage-count check assumed.

    Window hosting is checked per-endpoint (rule 78: "window endpoints must
    connect exactly to adjacent wall geometry", plural/each). Door hosting
    (rule 82) is phrased at the door-origin level, not per-endpoint - a door
    sitting right at the end of a wall run legitimately has only one side
    (its hinge or its end) continuing into more wall topology; as long as
    that one side anchors the door into the wall graph the door as a whole
    is hosted, so the other side alone having no further wall edge is not a
    floating-door error.
    """
    issues: list[ValidationIssue] = []
    coverage_by_type: dict[tuple[str, str], int] = {}
    for e in edges:
        coverage_by_type[(e.point_a_id, e.edge_type)] = coverage_by_type.get((e.point_a_id, e.edge_type), 0) + 1
        coverage_by_type[(e.point_b_id, e.edge_type)] = coverage_by_type.get((e.point_b_id, e.edge_type), 0) + 1

    door_partner: dict[str, str] = {}
    for e in edges:
        if e.edge_type == "door_origin":
            door_partner[e.point_a_id] = e.point_b_id
            door_partner[e.point_b_id] = e.point_a_id

    own_edge_type = {
        "wall_window_point": "window",
        "wall_door_hinge_point": "door_origin",
        "wall_door_end_point": "door_origin",
    }

    for p in points:
        opening_type = own_edge_type.get(p.point_type)
        if opening_type is None:
            continue
        own_count = coverage_by_type.get((p.id, opening_type), 0)
        wall_count = coverage_by_type.get((p.id, "wall"), 0)
        if own_count == 0:
            rule = "orphan_window_point" if opening_type == "window" else "orphan_door_point"
            edge_label = "window" if opening_type == "window" else "door_origin"
            issues.append(ValidationIssue(rule, f"{p.id} has no {edge_label} edge", [p.id]))
        elif own_count > 1:
            issues.append(ValidationIssue("opening_point_multiple_edges", f"{p.id} covered by {own_count} {opening_type} edges", [p.id]))
        if own_count > 0 and wall_count == 0:
            if opening_type == "door_origin":
                partner_id = door_partner.get(p.id)
                partner_wall_count = coverage_by_type.get((partner_id, "wall"), 0) if partner_id else 0
                if partner_wall_count > 0:
                    continue
                rule = "floating_door_point"
            else:
                rule = "floating_window_point"
            issues.append(ValidationIssue(rule, f"{p.id} has no wall edge - not hosted on wall topology", [p.id]))

    return issues
