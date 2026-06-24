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

from .graph_types import GraphEdge, GraphPoint, ValidationIssue
from .point_detection import WallSkeletonEdge, _project_point_onto_line

DEFAULT_NODE_MATCH_TOLERANCE_PX = 12.0
OPENING_POINT_TYPES = ("wall_window_point", "wall_door_hinge_point", "wall_door_end_point")


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


def _wall_direction_attachment(point: GraphPoint, direction: str):
    for a in point.attachments:
        if a.type == "wall" and a.direction == direction:
            return a
    return None


def _interior_opening_points_on_chain(
    se: WallSkeletonEdge, opening_points: list[GraphPoint], exclude_ids: set[str], tol_px: float,
) -> list[tuple[float, GraphPoint]]:
    """Window/door points whose projected location falls strictly between
    ``se.start`` and ``se.end`` (not at either pre-existing skeleton node).

    A window/door's host wall is often a single uninterrupted skeleton
    chain (the CNN wall mask has no real pixel gap there) - the opening's
    point sits mid-chain, not at one of the chain's own node pixels. Without
    this, the wall interval at the opening's location never gets replaced
    (rules 77/83) and the opening point ends up with no wall edge at all
    (rules 76/104/105: floating).
    """
    seg_dx, seg_dy = se.end[0] - se.start[0], se.end[1] - se.start[1]
    seg_len = math.hypot(seg_dx, seg_dy)
    if seg_len < 1e-6:
        return []
    found: list[tuple[float, GraphPoint]] = []
    for p in opening_points:
        if p.id in exclude_ids:
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
    any window/door point hosted mid-chain so the opening replaces the wall
    interval at its location instead of being spanned by one uninterrupted
    wall edge (rules 77/78, 82/83).

    A chain whose natural skeleton free end is real CNN segmentation noise
    short of where the window/door evidence actually starts is rejected
    upstream as a ``1_wall_point`` (point_detection.py's
    ``wall_end_near_opening_evidence`` - it correctly recognizes the two are
    related) but, at the tight exact/near-coordinate tolerance used above,
    never actually gets reconnected to that opening point - leaving the
    opening floating (rules 76/104/105). ``opening_match_tolerance_px``
    (normally the same `free_end_opening_proximity_px` config value that
    upstream check already uses) is a last-resort fallback that completes
    that connection, scoped to opening points only so it cannot wrongly
    merge unrelated plain wall corners across a large distance.
    """
    edges: list[GraphEdge] = []
    seen_pairs: set[frozenset] = set()
    counter = 0
    points_by_id = {p.id: p for p in points}
    opening_points = [p for p in points if p.point_type in OPENING_POINT_TYPES]

    for se in wall_skeleton_edges:
        pa = points_by_id.get(se.point_id_at_start) if se.point_id_at_start else None
        pb = points_by_id.get(se.point_id_at_end) if se.point_id_at_end else None
        if pa is None:
            pa = _nearest_point(points, se.start, tol_px) or _nearest_point(opening_points, se.start, opening_match_tolerance_px)
        if pb is None:
            pb = _nearest_point(points, se.end, tol_px) or _nearest_point(opening_points, se.end, opening_match_tolerance_px)
        if pa is None or pb is None or pa.id == pb.id:
            continue

        interior = _interior_opening_points_on_chain(se, opening_points, {pa.id, pb.id}, tol_px)
        interior.sort(key=lambda item: item[0])
        chain = [pa] + [p for _t, p in interior] + [pb]

        for left, right in zip(chain[:-1], chain[1:]):
            if left.id == right.id:
                continue
            if _wall_direction_attachment(left, se.dir_from_start) is None or _wall_direction_attachment(right, se.dir_from_end) is None:
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
) -> tuple[list[GraphEdge], list[ValidationIssue]]:
    """Build the final wall/window/door-origin graph (spec_v008 SS12 / SS7 step 8)."""
    cfg = config or {}
    tol_px = cfg.get("node_match_tolerance_px", DEFAULT_NODE_MATCH_TOLERANCE_PX)
    opening_match_tolerance_px = cfg.get("opening_match_tolerance_px", tol_px)

    wall_edges = build_wall_edges(points, wall_skeleton_edges, tol_px, opening_match_tolerance_px)
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
    """
    issues: list[ValidationIssue] = []
    coverage_by_type: dict[tuple[str, str], int] = {}
    for e in edges:
        coverage_by_type[(e.point_a_id, e.edge_type)] = coverage_by_type.get((e.point_a_id, e.edge_type), 0) + 1
        coverage_by_type[(e.point_b_id, e.edge_type)] = coverage_by_type.get((e.point_b_id, e.edge_type), 0) + 1

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
            rule = "floating_window_point" if opening_type == "window" else "floating_door_point"
            issues.append(ValidationIssue(rule, f"{p.id} has no wall edge - not hosted on wall topology", [p.id]))

    return issues
