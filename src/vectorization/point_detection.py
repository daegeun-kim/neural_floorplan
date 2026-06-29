"""Search directly for the four allowed point types (spec_v008 SS9, as
revised by task15 - see module note below).

Wall points come from walking each wall component's skeleton as a graph:
degree-1 pixels are candidate free ends, degree>=2 pixels are junctions -
all finalize as the single generic ``wall_point`` type (task15: wall graph
construction must not depend on accurate pre-classification of a point's
eventual degree; the actual attachment directions are still recorded on
each point, just not pre-baked into four separate type labels). Window
points are located by projecting their own component evidence onto the
nearest wall skeleton edge - reusing the same projection/hosting math the
old (retired) window_extraction.py/geometry_rules.py already implemented
correctly, ported here as private helpers operating on the new
skeleton-graph edges instead of pre-built WallPrimitive objects. Door hinge
and end points are simpler (task16): they are 2 of the red door_arc
bounding box's 4 vertices, chosen directly from purple/black/orange
evidence around the bbox - see ``select_door_hinge_end_from_bbox``.

A wall free end whose coordinate coincides with a window/door point (the
common case: an opening sits where the wall mask itself has a real gap) is
superseded by that window/door point rather than also being emitted as a
redundant ``wall_point`` - spec_v008 SS10 requires every location to
resolve to exactly one final point type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .graph_types import (
    OPPOSITE_DIRECTION,
    ALL_POINT_TYPES,
    Attachment,
    ComponentRecord,
    Direction,
    DoorCandidateRecord,
    GraphPoint,
    RejectedEvidence,
    ValidationIssue,
)

DEFAULTS: dict = {
    "cardinal_tolerance_deg": 25.0,
    # task15 problem 4: door recognition (is there a host wall at all) must
    # not be gated by a tight search radius - only must-rule 17's 200mm
    # arc-bbox-proximity floor (door_point_max_dist_from_arc_mm below) is a
    # fixed, must-rule-mandated number.
    "max_wall_dist": 100000.0,
    "min_hosted_width_px": 3.0,
    "corner_ambiguity_px": 25.0,
    "min_remainder_px": 3.0,
    # task16: probe band (px) around each door_arc bbox edge/corner used to
    # score purple/black/orange evidence when picking the hinge/end vertex
    # pair - no longer a mask-intersection search radius.
    "hinge_probe_radius": 14.0,
    "hinge_snap_to_wall_max_dist_px": 100000.0,
    "door_width_modules_mm": (700.0, 900.0),
    "free_end_merge_tol_px": 8.0,
    "min_window_width_mm": 300.0,
    "free_end_opening_proximity_px": 20.0,
    "door_point_max_dist_from_arc_mm": 200.0,
}


# ---------------------------------------------------------------------------
# Raw wall skeleton graph (internal - point_connection.py builds the real,
# aligned GraphEdges later; this is just enough structure to host window and
# door evidence the same way an already-built WallPrimitive list would).
# ---------------------------------------------------------------------------


@dataclass
class WallSkeletonEdge:
    id: str
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float
    component_id: int
    dir_from_start: Optional[Direction]
    dir_from_end: Optional[Direction]
    point_id_at_start: Optional[str] = None
    point_id_at_end: Optional[str] = None

    @property
    def orientation_angle(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.degrees(math.atan2(dy, dx))

    @property
    def length(self) -> float:
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    @property
    def primitive_id(self) -> str:
        return self.id


def _neighbors8(pt: tuple[int, int], pts_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    x, y = pt
    return [
        (x + dx, y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if not (dx == 0 and dy == 0) and (x + dx, y + dy) in pts_set
    ]


def _cardinal_direction(dx: float, dy: float, tolerance_deg: float) -> Optional[Direction]:
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    if abs(dx) >= abs(dy):
        off = math.degrees(math.atan2(abs(dy), abs(dx)))
        if off > tolerance_deg:
            return None
        return "right" if dx > 0 else "left"
    off = math.degrees(math.atan2(abs(dx), abs(dy)))
    if off > tolerance_deg:
        return None
    return "down" if dy > 0 else "up"


def _walk_chain(
    node_pixels: set[tuple[int, int]],
    skel_set: set[tuple[int, int]],
    start: tuple[int, int],
    first: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [start, first]
    prev, curr = start, first
    while curr not in node_pixels:
        nbrs = [n for n in _neighbors8(curr, skel_set) if n != prev]
        if not nbrs:
            break
        nxt = nbrs[0]
        path.append(nxt)
        prev, curr = curr, nxt
    return path


def _chain_length(path: list[tuple[int, int]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:]))


def _local_axis_labels(path: list[tuple[int, int]], window: int = 5) -> list[str]:
    """Per-pixel dominant local axis ("h" or "v"), from a window of pixels on
    either side - robust to the single-pixel diagonal step skeletonize often
    inserts exactly at a right-angle corner."""
    n = len(path)
    labels = []
    for i in range(n):
        a = path[max(0, i - window)]
        b = path[min(n - 1, i + window)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        labels.append("h" if abs(dx) >= abs(dy) else "v")
    return labels


def _split_path_at_corners(path: list[tuple[int, int]], window: int = 5) -> list[list[tuple[int, int]]]:
    """Split a skeleton chain wherever its dominant local axis changes.

    skeletonize represents an L-shaped wall corner as one continuous
    degree-2 path with a turn in it, not a branch - so a chain between two
    true node pixels can itself contain a corner. Checking the chain's
    overall start-to-end direction would see that as diagonal and reject it;
    splitting at the turn first lets each resulting sub-path be checked (and
    pass) as its own clean cardinal run.
    """
    if len(path) < 3:
        return [path]
    labels = _local_axis_labels(path, window)
    split_indices = sorted({0, *(i for i in range(1, len(labels)) if labels[i] != labels[i - 1]), len(path) - 1})
    sub_paths = [path[a:b + 1] for a, b in zip(split_indices[:-1], split_indices[1:]) if b > a]
    return sub_paths if sub_paths else [path]


def build_wall_skeleton_graph(
    wall_components: list[ComponentRecord], cardinal_tolerance_deg: float = 25.0
) -> tuple[dict[tuple[int, int], list[WallSkeletonEdge]], list[RejectedEvidence]]:
    """Walk each wall component's skeleton into a node/edge graph.

    Returns ``{node_pixel: [incident WallSkeletonEdge, ...]}``. Chains whose
    overall direction is not within ``cardinal_tolerance_deg`` of a cardinal
    axis are rejected (spec_v008 SS4: no diagonal final wall evidence).
    """
    node_edges: dict[tuple[int, int], list[WallSkeletonEdge]] = {}
    rejected: list[RejectedEvidence] = []
    edge_counter = 0

    for comp in wall_components:
        skel_pts = comp.skeleton_points
        if len(skel_pts) < 2:
            continue
        skel_set = set(skel_pts)
        # Local thickness per sub-chain, not comp.rect_size[1]: that field is
        # the *whole component's* overall minAreaRect short axis, which is a
        # meaningless wall-thickness estimate for a large, non-rectangular
        # (L-shaped, multi-segment, or branching) component such as a full
        # outer-wall loop - it can be hundreds of px, wildly inflating the
        # final polygon buffer width (rules 13/123: thickness must resolve to
        # 100mm or 200mm). cv2.distanceTransform gives, per wall pixel, the
        # distance to the nearest non-wall pixel; doubling that sampled along
        # each sub-chain's own pixels is a real local thickness measurement.
        dist_transform = None
        if comp.mask is not None:
            dist_transform = cv2.distanceTransform((comp.mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        degree = {p: len(_neighbors8(p, skel_set)) for p in skel_pts}
        node_pixels = {p for p, d in degree.items() if d != 2}
        if not node_pixels:
            node_pixels = {skel_pts[0], skel_pts[-1]}

        visited_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for node in list(node_pixels):
            for nbr in _neighbors8(node, skel_set):
                key = (node, nbr)
                if key in visited_pairs:
                    continue
                path = _walk_chain(node_pixels, skel_set, node, nbr)
                end = path[-1]
                visited_pairs.add((node, nbr))
                if len(path) >= 2:
                    visited_pairs.add((end, path[-2]))
                if end == node and len(path) < 3:
                    continue

                for sub_path in _split_path_at_corners(path):
                    sub_start, sub_end = sub_path[0], sub_path[-1]
                    if sub_start == sub_end:
                        continue
                    dx, dy = sub_end[0] - sub_start[0], sub_end[1] - sub_start[1]
                    dir_from_start = _cardinal_direction(dx, dy, cardinal_tolerance_deg)
                    dir_from_end = _cardinal_direction(-dx, -dy, cardinal_tolerance_deg)
                    if dir_from_start is None or dir_from_end is None:
                        rejected.append(
                            RejectedEvidence(
                                kind="diagonal_wall_edge",
                                reason="wall skeleton sub-chain is not within cardinal tolerance",
                                class_name="wall",
                                bbox=comp.bbox,
                                centroid=((sub_start[0] + sub_end[0]) / 2.0, (sub_start[1] + sub_end[1]) / 2.0),
                                component_id=comp.component_id,
                            )
                        )
                        continue

                    local_thickness = None
                    if dist_transform is not None:
                        sample_dists = [
                            float(dist_transform[y, x])
                            for x, y in sub_path
                            if 0 <= y < dist_transform.shape[0] and 0 <= x < dist_transform.shape[1]
                        ]
                        if sample_dists:
                            local_thickness = 2.0 * float(np.median(sample_dists))
                    if not local_thickness:
                        local_thickness = comp.rect_size[1] if comp.rect_size else 8.0

                    edge_counter += 1
                    edge = WallSkeletonEdge(
                        id=f"walledge_{comp.component_id}_{edge_counter}",
                        start=(float(sub_start[0]), float(sub_start[1])),
                        end=(float(sub_end[0]), float(sub_end[1])),
                        thickness=local_thickness,
                        component_id=comp.component_id,
                        dir_from_start=dir_from_start,
                        dir_from_end=dir_from_end,
                    )
                    node_edges.setdefault(sub_start, []).append(edge)
                    node_edges.setdefault(sub_end, []).append(edge)

    return node_edges, rejected


def _classify_wall_nodes(
    node_edges: dict[tuple[int, int], list[WallSkeletonEdge]],
) -> tuple[list[GraphPoint], dict[tuple[int, int], WallSkeletonEdge]]:
    """Junctions (>=2 incident edges) finalize immediately as the generic
    ``wall_point`` type (task15: wall graph construction must not depend on
    accurate pre-classification of a point's eventual degree); degree-1 nodes
    are returned separately as free-end candidates, since window/door search
    must get a chance to reclassify them first."""
    points: list[GraphPoint] = []
    free_ends: dict[tuple[int, int], WallSkeletonEdge] = {}
    counter = 0

    for node, edges in node_edges.items():
        if not edges:
            continue
        if len(edges) == 1:
            free_ends[node] = edges[0]
            continue
        counter += 1
        attachments = []
        for e in edges:
            d = e.dir_from_start if e.start == (float(node[0]), float(node[1])) else e.dir_from_end
            attachments.append(Attachment(type="wall", direction=d, source="wall", evidence_length_px=e.length))
        points.append(
            GraphPoint(
                id=f"wallpt_{counter}",
                point_type="wall_point",
                coordinate=(float(node[0]), float(node[1])),
                attachments=attachments,
                source_component_ids=[e.component_id for e in edges],
            )
        )
    return points, free_ends


def _free_end_near_opening_evidence(node: tuple[int, int], masks: dict[str, np.ndarray], radius_px: float) -> bool:
    """True if window/door_arc/door_leaf/door_origin mask evidence exists
    within ``radius_px`` of a candidate wall free-end pixel - such an end is
    not a true peninsula and must not finalize as a ``1_wall_point`` (task12
    SS2.1: "no window or door evidence touches or sits immediately near that
    wall end")."""
    nx, ny = node
    r = int(round(radius_px))
    for key in ("window", "door_arc", "door_leaf", "door_origin"):
        mask = masks.get(key)
        if mask is None:
            continue
        h, w = mask.shape
        x0, x1 = max(0, nx - r), min(w, nx + r + 1)
        y0, y1 = max(0, ny - r), min(h, ny + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        if mask[y0:y1, x0:x1].any():
            return True
    return False


def _finalize_free_ends(
    free_ends: dict[tuple[int, int], WallSkeletonEdge],
    other_points: list[GraphPoint],
    tol_px: float,
    masks: dict[str, np.ndarray],
    opening_proximity_px: float,
) -> tuple[list[GraphPoint], list[RejectedEvidence]]:
    other_coords = [p.coordinate for p in other_points]
    points: list[GraphPoint] = []
    rejected: list[RejectedEvidence] = []
    counter = 0
    for node, edge in free_ends.items():
        nx, ny = float(node[0]), float(node[1])
        if any(math.hypot(nx - ox, ny - oy) <= tol_px for ox, oy in other_coords):
            continue
        if _free_end_near_opening_evidence(node, masks, opening_proximity_px):
            rejected.append(
                RejectedEvidence(
                    kind="wall_end_near_opening_evidence",
                    reason="wall free end is near window/door mask evidence - not a true 1_wall_point",
                    class_name="wall",
                    centroid=(nx, ny),
                    component_id=edge.component_id,
                )
            )
            continue
        counter += 1
        d = edge.dir_from_start if edge.start == (nx, ny) else edge.dir_from_end
        points.append(
            GraphPoint(
                id=f"wallpt_free_{counter}",
                point_type="wall_point",
                coordinate=(nx, ny),
                attachments=[Attachment(type="wall", direction=d, source="wall", evidence_length_px=edge.length)],
                source_component_ids=[edge.component_id],
            )
        )
    return points, rejected


# ---------------------------------------------------------------------------
# Projection/hosting helpers (ported from the retired geometry_rules.py -
# pure geometry, duck-typed against any object with .start/.end/.thickness/
# .orientation_angle/.length/.primitive_id, so WallSkeletonEdge works unchanged).
# ---------------------------------------------------------------------------


def _project_point_onto_line(
    point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> tuple[tuple[float, float], float, float]:
    ax, ay = a
    bx, by = b
    px, py = point
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-9:
        return a, math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    proj = (ax + t * dx, ay + t * dy)
    dist = math.hypot(px - proj[0], py - proj[1])
    return proj, dist, t


def _point_to_wall_distance(point: tuple[float, float], wall) -> float:
    cx, cy = point
    x1, y1 = wall.start
    x2, y2 = wall.end
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return math.hypot(cx - x1, cy - y1)
    t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq))
    proj_x, proj_y = x1 + t * dx, y1 + t * dy
    return math.hypot(cx - proj_x, cy - proj_y)


def _min_pixel_distance_to_wall(pixel_coords: np.ndarray, wall) -> float:
    """Minimum distance from any of the opening's own pixels to ``wall``,
    not the opening's centroid.

    A long/tall opening's centroid can sit far from its real host wall
    whenever that wall is itself split into two short chains (one above and
    one below the opening's real pixel gap) - the centroid-distance check
    then sees neither chain as "near" even though the opening's own
    extremity is right next to one of them (rule 75: every final window
    must be hosted by wall topology).
    """
    x1, y1 = wall.start
    x2, y2 = wall.end
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-9:
        return float(np.min(np.hypot(pixel_coords[:, 0] - x1, pixel_coords[:, 1] - y1)))
    t = np.clip(((pixel_coords[:, 0] - x1) * dx + (pixel_coords[:, 1] - y1) * dy) / seg_len_sq, 0.0, 1.0)
    proj_x, proj_y = x1 + t * dx, y1 + t * dy
    return float(np.min(np.hypot(pixel_coords[:, 0] - proj_x, pixel_coords[:, 1] - proj_y)))


def nearest_wall(center: tuple[float, float], walls: list, max_dist: float = 40.0):
    if not walls:
        return None
    best_wall = None
    best_dist = max_dist
    for wall in walls:
        dist = _point_to_wall_distance(center, wall)
        if dist < best_dist:
            best_dist = dist
            best_wall = wall
    return best_wall


def _nearest_wall_matching_orientation(center: tuple[float, float], walls: list, orientation: str, max_dist: float):
    """Prefer the nearest wall edge whose own running ``orientation``
    (``"horizontal"`` = ``dir_from_start`` in left/right, ``"vertical"`` =
    up/down) matches; fall back to plain nearest-by-distance only when no
    matching-orientation wall exists within ``max_dist`` (task17 - see
    ``_detect_door_points``'s hosting call for why this matters)."""
    matching = [
        w for w in walls
        if ("horizontal" if w.dir_from_start in ("left", "right") else "vertical") == orientation
    ]
    host = nearest_wall(center, matching, max_dist=max_dist)
    if host is not None:
        return host
    return nearest_wall(center, walls, max_dist=max_dist)


def _dominant_axis_angle_deg(pixel_coords: np.ndarray) -> float:
    pts = pixel_coords - pixel_coords.mean(axis=0)
    if pts.shape[0] < 2:
        return 0.0
    cov = np.cov(pts.T)
    if np.ndim(cov) == 0 or not np.all(np.isfinite(cov)):
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    dominant = eigvecs[:, int(np.argmax(eigvals))]
    return math.degrees(math.atan2(dominant[1], dominant[0]))


def _hosting_probability(pixel_coords: np.ndarray, wall, min_remainder_px: float) -> float:
    x1, y1 = wall.start
    x2, y2 = wall.end
    dx, dy = x2 - x1, y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-6:
        return 0.0
    ux, uy = dx / seg_len, dy / seg_len

    evidence_angle = _dominant_axis_angle_deg(pixel_coords)
    diff = abs((evidence_angle - wall.orientation_angle) % 180.0)
    if diff > 90.0:
        diff = 180.0 - diff
    orientation_term = max(0.0, 1.0 - diff / 90.0)

    rel_x = pixel_coords[:, 0] - x1
    rel_y = pixel_coords[:, 1] - y1
    t_px = rel_x * ux + rel_y * uy
    perp_dist = np.abs(rel_x * uy - rel_y * ux)

    overlap_slack_px = 4.0
    overlap_term = float(np.mean(perp_dist <= (wall.thickness / 2.0 + overlap_slack_px)))

    t_min, t_max = float(t_px.min()), float(t_px.max())
    remainder_left = max(0.0, t_min)
    remainder_right = max(0.0, seg_len - t_max)
    remainder_px = min(remainder_left, remainder_right)
    remainder_term = max(0.0, min(1.0, remainder_px / (4.0 * max(min_remainder_px, 1e-6))))

    return orientation_term + overlap_term + remainder_term


def select_host_wall_for_opening(
    pixel_coords: np.ndarray,
    walls: list,
    max_dist: float = 40.0,
    corner_ambiguity_px: float = 25.0,
    min_remainder_px: float = 3.0,
):
    if not walls or pixel_coords is None or len(pixel_coords) == 0:
        return None
    scored = [(dist, wall) for wall in walls if (dist := _min_pixel_distance_to_wall(pixel_coords, wall)) <= max_dist]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    nearest_dist, nearest = scored[0]
    if len(scored) == 1 or (scored[1][0] - nearest_dist) > corner_ambiguity_px:
        return nearest
    _second_dist, second = scored[1]
    score_nearest = _hosting_probability(pixel_coords, nearest, min_remainder_px)
    score_second = _hosting_probability(pixel_coords, second, min_remainder_px)
    return nearest if score_nearest >= score_second else second


def project_pixels_onto_wall(pixel_coords: np.ndarray, wall) -> tuple[tuple[float, float], float, float, float]:
    x1, y1 = wall.start
    x2, y2 = wall.end
    dx, dy = x2 - x1, y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-6:
        cx, cy = float(pixel_coords[:, 0].mean()), float(pixel_coords[:, 1].mean())
        return (cx, cy), 0.0, 0.0, 0.0
    ux, uy = dx / seg_len, dy / seg_len
    rel_x = pixel_coords[:, 0] - x1
    rel_y = pixel_coords[:, 1] - y1
    t_px = rel_x * ux + rel_y * uy
    t_min, t_max = float(t_px.min()), float(t_px.max())
    width = max(t_max - t_min, 1.0)
    t_mid = (t_min + t_max) / 2.0
    center = (x1 + ux * t_mid, y1 + uy * t_mid)
    return center, width, t_min / seg_len, t_max / seg_len


# ---------------------------------------------------------------------------
# Window point search (spec_v008 SS9.2)
# ---------------------------------------------------------------------------


def _detect_window_points(
    window_components: list[ComponentRecord], wall_edges: list, cfg: dict
) -> tuple[list[GraphPoint], list[RejectedEvidence]]:
    points: list[GraphPoint] = []
    rejected: list[RejectedEvidence] = []
    counter = 0

    for comp in window_components:
        ys, xs = np.nonzero(comp.mask)
        pixel_coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

        host_edge = select_host_wall_for_opening(
            pixel_coords, wall_edges, max_dist=cfg["max_wall_dist"],
            corner_ambiguity_px=cfg["corner_ambiguity_px"], min_remainder_px=cfg["min_remainder_px"],
        )
        if host_edge is None:
            rejected.append(
                RejectedEvidence(kind="window_unhosted", reason="no wall within max_wall_dist",
                                  class_name="window", bbox=comp.bbox, centroid=comp.centroid,
                                  component_id=comp.component_id)
            )
            continue

        _center, width_px, t_min, t_max = project_pixels_onto_wall(pixel_coords, host_edge)
        if width_px < cfg["min_hosted_width_px"]:
            rejected.append(
                RejectedEvidence(kind="window_too_narrow", reason=f"{width_px:.1f}px hosted width",
                                  class_name="window", bbox=comp.bbox, centroid=comp.centroid,
                                  component_id=comp.component_id)
            )
            continue

        scale_info = cfg.get("scale_info")
        if (
            scale_info is None
            or scale_info.px_to_mm is None
            or scale_info.scale_status not in ("resolved", "estimated")
        ):
            rejected.append(
                RejectedEvidence(kind="window_scale_blocked", reason="scale not resolved",
                                  class_name="window", bbox=comp.bbox, centroid=comp.centroid,
                                  component_id=comp.component_id)
            )
            continue
        if width_px * scale_info.px_to_mm < cfg["min_window_width_mm"]:
            rejected.append(
                RejectedEvidence(kind="window_too_narrow_mm", reason=f"{width_px * scale_info.px_to_mm:.1f}mm hosted width",
                                  class_name="window", bbox=comp.bbox, centroid=comp.centroid,
                                  component_id=comp.component_id)
            )
            continue

        seg_len = host_edge.length
        ux = (host_edge.end[0] - host_edge.start[0]) / seg_len
        uy = (host_edge.end[1] - host_edge.start[1]) / seg_len
        p_min = (host_edge.start[0] + ux * t_min * seg_len, host_edge.start[1] + uy * t_min * seg_len)
        p_max = (host_edge.start[0] + ux * t_max * seg_len, host_edge.start[1] + uy * t_max * seg_len)
        dx, dy = p_max[0] - p_min[0], p_max[1] - p_min[1]
        dir_min_to_max = _cardinal_direction(dx, dy, tolerance_deg=45.0)
        if dir_min_to_max is None:
            rejected.append(
                RejectedEvidence(kind="window_axis_ambiguous", reason="hosted span is not cardinal",
                                  class_name="window", bbox=comp.bbox, centroid=comp.centroid,
                                  component_id=comp.component_id)
            )
            continue
        dir_max_to_min = OPPOSITE_DIRECTION[dir_min_to_max]

        counter += 1
        points.append(
            GraphPoint(
                id=f"winpt_{counter}_a", point_type="wall_window_point", coordinate=p_min,
                attachments=[
                    Attachment(type="wall", direction=dir_max_to_min, source="wall"),
                    Attachment(type="window", direction=dir_min_to_max, source="window", evidence_length_px=width_px,
                               host_thickness_px=host_edge.thickness),
                ],
                source_component_ids=[comp.component_id],
                host_wall_edge_id=host_edge.id,
            )
        )
        counter += 1
        points.append(
            GraphPoint(
                id=f"winpt_{counter}_b", point_type="wall_window_point", coordinate=p_max,
                attachments=[
                    Attachment(type="wall", direction=dir_min_to_max, source="wall"),
                    Attachment(type="window", direction=dir_max_to_min, source="window", evidence_length_px=width_px,
                               host_thickness_px=host_edge.thickness),
                ],
                host_wall_edge_id=host_edge.id,
                source_component_ids=[comp.component_id],
            )
        )

    return points, rejected


# ---------------------------------------------------------------------------
# Door point search (spec_v008 SS9.3). task16: hinge/end are 2 of the red
# door_arc bbox's 4 vertices, picked by purple/black/orange evidence along
# its edges (select_door_hinge_end_from_bbox) - not searched from mask
# intersections or arc geometry.
# ---------------------------------------------------------------------------


def _nearest_door_module_mm(value_mm: float, modules_mm: tuple[float, ...]) -> float:
    best = modules_mm[0]
    best_err = abs(value_mm - best)
    for module in modules_mm[1:]:
        err = abs(value_mm - module)
        if err < best_err:
            best_err = err
            best = module
    return best


def _extend_point(origin: tuple[float, float], through: tuple[float, float], new_distance: float) -> tuple[float, float]:
    ox, oy = origin
    tx, ty = through
    dx, dy = tx - ox, ty - oy
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return through
    ux, uy = dx / dist, dy / dist
    return (ox + ux * new_distance, oy + uy * new_distance)


def _point_to_bbox_distance(point: tuple[float, float], bbox: tuple[int, int, int, int]) -> float:
    """Euclidean distance from ``point`` to the nearest edge/corner of
    ``bbox`` (0.0 when the point lies inside it)."""
    x, y = point
    x0, y0, x1, y1 = bbox
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def _band_pixel_count(mask: Optional[np.ndarray], x0: float, y0: float, x1: float, y1: float) -> int:
    """Count of nonzero ``mask`` pixels in the (clipped) box ``[x0,x1)x[y0,y1)``."""
    if mask is None:
        return 0
    h, w = mask.shape
    ix0, iy0 = max(0, int(round(x0))), max(0, int(round(y0)))
    ix1, iy1 = min(w, int(round(x1))), min(h, int(round(y1)))
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    return int(np.count_nonzero(mask[iy0:iy1, ix0:ix1]))


def _bbox_edges(bbox: tuple[int, int, int, int]) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """The 4 edges of ``bbox``, each as its 2 endpoint vertices."""
    x0, y0, x1, y1 = bbox
    return {
        "top": ((float(x0), float(y0)), (float(x1), float(y0))),
        "bottom": ((float(x0), float(y1)), (float(x1), float(y1))),
        "left": ((float(x0), float(y0)), (float(x0), float(y1))),
        "right": ((float(x1), float(y0)), (float(x1), float(y1))),
    }


def _edge_band_score(
    p1: tuple[float, float], p2: tuple[float, float],
    purple_mask: Optional[np.ndarray], wall_mask: Optional[np.ndarray], probe_px: float,
) -> int:
    """Combined purple (``door_origin``) + black (``wall``) pixel count in a
    band straddling the bbox edge ``p1``-``p2`` (task16: a door's hinge and
    end sit on whichever bbox edge the wall and door-origin evidence
    actually runs along). The band is only widened *perpendicular* to the
    edge by ``probe_px`` - along the edge it stays exactly the edge's own
    span - so a short, real evidence run near one corner of, say, the left
    edge cannot also bleed into and tie with the top/bottom edges' bands.
    """
    x0, y0, x1, y1 = _edge_perpendicular_band(p1, p2, probe_px)
    return _band_pixel_count(purple_mask, x0, y0, x1, y1) + _band_pixel_count(wall_mask, x0, y0, x1, y1)


def _corner_orange_score(corner: tuple[float, float], orange_mask: Optional[np.ndarray], probe_px: float) -> int:
    x, y = corner
    return _band_pixel_count(orange_mask, x - probe_px, y - probe_px, x + probe_px, y + probe_px)


@dataclass
class DoorVertexSelection:
    """Result of scoring a red ``door_arc`` bbox's 4 vertices (task17 ## "Door
    Point Selection" / "Required Metrics"). ``hinge``/``end`` are 2 *adjacent*
    vertices of ``all_vertices`` - the edge between them is ``edge_name``,
    scored ``edge_score`` (the edge's score is attributed to both of its
    vertices: per-vertex scoring and edge-based scoring are the same
    selection, just two ways of describing it)."""

    hinge: tuple[float, float]
    end: tuple[float, float]
    edge_name: str
    edge_score: int
    all_vertices: dict[str, tuple[float, float]]
    host_wall_alignment_score: int = 0


def select_door_hinge_end_from_bbox(
    bbox: tuple[int, int, int, int],
    purple_mask: Optional[np.ndarray],
    wall_mask: Optional[np.ndarray],
    orange_mask: Optional[np.ndarray],
    probe_px: float,
) -> DoorVertexSelection:
    """task16/task17: hinge and end are 2 *adjacent* vertices of the red
    ``door_arc`` bbox, not points searched from raw mask intersections/arc
    geometry. The red bbox is trusted unconditionally (task17 "Red Bbox
    Assumption") - this always returns a selection, never ``None``, even
    when every edge scores 0 (no plausible host-wall geometry is grounds for
    rejecting the resulting *door*, later, via the 200mm-from-bbox floor -
    not for refusing to pick 2 adjacent vertices at all).

    The 2 adjacent vertices chosen are the endpoints of whichever bbox edge
    has the strongest combined purple (``door_origin``) + black (``wall``)
    evidence running along it - the wall-facing edge, since both the hinge
    and the end sit on the wall. Of those 2 endpoints, the one with more
    nearby orange (``door_leaf``) evidence is the hinge (the leaf pivots
    open from the hinge, not from the end). Real wall evidence concentrated
    on one edge already *is* "the bbox edge aligned with the host wall"
    (task17 ## "Door Point Selection" item 6) - a separate skeleton-edge
    lookup turned out to be the wrong place to enforce that: the nearest
    skeleton chain by raw centroid distance is frequently a short, unrelated
    noise fragment, so restricting candidates to *its* orientation overrode
    correct, well-evidenced selections more often than it fixed bad ones
    (see ## 20 "Task17 Debugging Notes").
    """
    x0, y0, x1, y1 = bbox
    all_vertices = {
        "top_left": (float(x0), float(y0)), "top_right": (float(x1), float(y0)),
        "bottom_left": (float(x0), float(y1)), "bottom_right": (float(x1), float(y1)),
    }
    edges = _bbox_edges(bbox)
    scored = [
        (_edge_band_score(p1, p2, purple_mask, wall_mask, probe_px), name, p1, p2)
        for name, (p1, p2) in edges.items()
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, edge_name, p1, p2 = scored[0]
    if _corner_orange_score(p1, orange_mask, probe_px) >= _corner_orange_score(p2, orange_mask, probe_px):
        hinge, end = p1, p2
    else:
        hinge, end = p2, p1
    host_wall_alignment_score = _band_pixel_count(
        wall_mask, *_edge_perpendicular_band(p1, p2, probe_px)
    )
    return DoorVertexSelection(
        hinge=hinge, end=end, edge_name=edge_name, edge_score=best_score, all_vertices=all_vertices,
        host_wall_alignment_score=host_wall_alignment_score,
    )


def _edge_perpendicular_band(
    p1: tuple[float, float], p2: tuple[float, float], probe_px: float
) -> tuple[float, float, float, float]:
    x0, x1 = sorted((p1[0], p2[0]))
    y0, y1 = sorted((p1[1], p2[1]))
    if x0 == x1:
        return x0 - probe_px, y0, x1 + probe_px, y1
    return x0, y0 - probe_px, x1, y1 + probe_px


def _detect_door_points(
    door_arc_components: list[ComponentRecord],
    door_origin_components: list[ComponentRecord],
    door_leaf_mask: Optional[np.ndarray],
    door_origin_mask: Optional[np.ndarray],
    wall_mask: Optional[np.ndarray],
    wall_edges: list,
    cfg: dict,
) -> tuple[list[GraphPoint], list[RejectedEvidence]]:
    """task16/task17: hinge/end come directly from 2 adjacent vertices of the
    door_arc bbox (``select_door_hinge_end_from_bbox``) instead of being
    searched from orange/purple mask intersections and arc geometry. The red
    bbox is trusted unconditionally (task17 "Red Bbox Assumption") - a
    vertex pair is always selected, even with zero evidence; only the
    min-area cleanup (component extraction, upstream), the bbox aspect-ratio
    floor below, and the downstream too-narrow/scale-blocked/200mm-floor
    checks can still reject."""
    points: list[GraphPoint] = []
    rejected: list[RejectedEvidence] = []
    counter = 0
    scale_info = cfg["scale_info"]
    probe_px = cfg["hinge_probe_radius"]
    max_aspect_ratio = cfg.get("max_door_bbox_aspect_ratio", 2.0)

    for arc in door_arc_components:
        if arc.mask is None:
            continue

        x0, y0, x1, y1 = arc.bbox
        width_px, height_px = float(x1 - x0), float(y1 - y0)
        shorter, longer = sorted((width_px, height_px))
        aspect_ratio = longer / shorter if shorter > 0 else float("inf")
        if aspect_ratio > max_aspect_ratio:
            rejected.append(RejectedEvidence(kind="unresolved_door_arc_aspect_ratio",
                                               reason=f"bbox aspect ratio {aspect_ratio:.2f}:1 exceeds {max_aspect_ratio:.2f}:1",
                                               class_name="door_arc", bbox=arc.bbox, centroid=arc.centroid,
                                               component_id=arc.component_id))
            continue

        selection = select_door_hinge_end_from_bbox(arc.bbox, door_origin_mask, wall_mask, door_leaf_mask, probe_px)
        hinge_point, far_point = selection.hinge, selection.end

        raw_width_px = math.hypot(far_point[0] - hinge_point[0], far_point[1] - hinge_point[1])
        if raw_width_px < cfg["min_hosted_width_px"]:
            rejected.append(RejectedEvidence(kind="unresolved_door_too_narrow",
                                               reason=f"{raw_width_px:.1f}px bbox edge span",
                                               class_name="door_arc", bbox=arc.bbox, centroid=hinge_point,
                                               component_id=arc.component_id))
            continue

        if (
            scale_info is None
            or scale_info.px_to_mm is None
            or scale_info.scale_status not in ("resolved", "estimated")
        ):
            rejected.append(RejectedEvidence(kind="unresolved_door_scale_blocked",
                                               reason="scale not resolved",
                                               class_name="door_arc", bbox=arc.bbox, centroid=hinge_point,
                                               component_id=arc.component_id))
            continue

        width_mm = _nearest_door_module_mm(raw_width_px * scale_info.px_to_mm, cfg["door_width_modules_mm"])
        snapped_width_px = width_mm / scale_info.px_to_mm
        max_dist_px = cfg["door_point_max_dist_from_arc_mm"] / scale_info.px_to_mm
        snapped_far_point = _extend_point(hinge_point, far_point, snapped_width_px)

        if scale_info.scale_status == "resolved":
            # task15 problem 3: rules 10/124 require the final door-origin
            # width to be exactly 700mm or 900mm whenever scale is
            # *resolved* (explicit/high-confidence) - always snap, with no
            # silent exception. The unresolved_door_too_far_from_arc check
            # right below then evaluates the *snapped* point, so a span that
            # can't be snapped within rule 17's 200mm floor is correctly
            # rejected (rule 51's "no plausible geometry" case) rather than
            # silently kept unsnapped.
            far_point = snapped_far_point
        else:
            # scale_status == "estimated": snap only when it still lands
            # within rule 17's 200mm floor; otherwise keep the
            # real-evidence-grounded bbox vertex (see task15 notes).
            if _point_to_bbox_distance(snapped_far_point, arc.bbox) <= max_dist_px:
                far_point = snapped_far_point

        # A door commonly sits between two separate wall stub fragments, one
        # on each side - host the hinge and end independently rather than
        # forcing both onto whichever single wall happened to be nearest the
        # arc's overall shape (task15 notes).
        #
        # task17: the hinge/end vector's own orientation (vertical when they
        # share x, horizontal when they share y) must match its host wall's
        # running direction, or point_connection.py's build_wall_edges later
        # finds the point's "wall" attachment direction incompatible with
        # the chain's own direction labels and silently drops the edge,
        # leaving the door floating. Plain nearest-by-distance can pick an
        # orientation-incompatible wall over a slightly farther compatible
        # one (a short, irrelevant skeleton fragment closer to the point
        # than the real host) - prefer an orientation-matching host first.
        door_orientation = "vertical" if hinge_point[0] == far_point[0] else "horizontal"
        hinge_host = _nearest_wall_matching_orientation(
            hinge_point, wall_edges, door_orientation, cfg["hinge_snap_to_wall_max_dist_px"]
        )
        far_host = _nearest_wall_matching_orientation(
            far_point, wall_edges, door_orientation, cfg["hinge_snap_to_wall_max_dist_px"]
        )
        if hinge_host is None or far_host is None:
            rejected.append(RejectedEvidence(kind="unresolved_door_hinge",
                                               reason="hinge/end vertex too far from any wall",
                                               class_name="door_arc", bbox=arc.bbox, centroid=hinge_point,
                                               component_id=arc.component_id))
            continue

        hinge_dist_px = _point_to_bbox_distance(hinge_point, arc.bbox)
        end_dist_px = _point_to_bbox_distance(far_point, arc.bbox)
        if hinge_dist_px > max_dist_px or end_dist_px > max_dist_px:
            rejected.append(RejectedEvidence(kind="unresolved_door_too_far_from_arc",
                                               reason=(f"hinge={hinge_dist_px:.1f}px end={end_dist_px:.1f}px "
                                                       f"exceeds {max_dist_px:.1f}px from door_arc bbox"),
                                               class_name="door_arc", bbox=arc.bbox, centroid=hinge_point,
                                               component_id=arc.component_id))
            continue

        dx, dy = far_point[0] - hinge_point[0], far_point[1] - hinge_point[1]
        dir_hinge_to_far = _cardinal_direction(dx, dy, tolerance_deg=45.0)
        if dir_hinge_to_far is None:
            rejected.append(RejectedEvidence(kind="unresolved_door_axis",
                                               reason="hinge-to-end direction is not cardinal",
                                               class_name="door_arc", bbox=arc.bbox, centroid=hinge_point,
                                               component_id=arc.component_id))
            continue
        dir_far_to_hinge = OPPOSITE_DIRECTION[dir_hinge_to_far]

        counter += 1
        points.append(
            GraphPoint(
                id=f"doorpt_{counter}_hinge", point_type="wall_door_hinge_point", coordinate=hinge_point,
                attachments=[
                    Attachment(type="wall", direction=dir_far_to_hinge, source="wall"),
                    Attachment(type="door_origin", direction=dir_hinge_to_far, source="door_origin",
                               evidence_length_px=snapped_width_px, host_thickness_px=hinge_host.thickness,
                               confidence=1.0),
                ],
                source_component_ids=[arc.component_id],
                host_wall_edge_id=hinge_host.id,
            )
        )
        points.append(
            GraphPoint(
                id=f"doorpt_{counter}_end", point_type="wall_door_end_point", coordinate=far_point,
                attachments=[
                    Attachment(type="wall", direction=dir_hinge_to_far, source="wall"),
                    Attachment(type="door_origin", direction=dir_far_to_hinge, source="door_origin",
                               evidence_length_px=snapped_width_px, host_thickness_px=far_host.thickness,
                               confidence=1.0),
                ],
                source_component_ids=[arc.component_id],
                host_wall_edge_id=far_host.id,
            )
        )

    # task16: door_origin evidence is no longer paired to a specific red
    # cluster (hinge/end come from the cluster's own bbox) - every
    # door_origin component is reported as supporting evidence only,
    # consistent with rule 52 (purple alone never creates a door).
    for comp in door_origin_components:
        rejected.append(RejectedEvidence(kind="unresolved_door_origin",
                                           reason="door_origin evidence is supporting evidence only, not paired to a specific red cluster",
                                           class_name="door_origin", bbox=comp.bbox, centroid=comp.centroid,
                                           component_id=comp.component_id))

    return points, rejected


# ---------------------------------------------------------------------------
# Orchestration + validation
# ---------------------------------------------------------------------------


def detect_points(
    components: dict[str, list[ComponentRecord]],
    masks: dict[str, np.ndarray],
    scale_info,
    config: Optional[dict] = None,
) -> tuple[list[GraphPoint], list[RejectedEvidence], list[WallSkeletonEdge]]:
    """Search directly for the seven allowed point types (spec_v008 SS9/SS7
    step 5). ``masks`` must contain the cleaned per-class masks (at least
    "door_leaf" and "door_origin", used for hinge intersection search).

    Also returns the raw wall skeleton edges, since point_connection.py needs
    the same topology this module already discovered to reconnect whichever
    final point ended up at each skeleton node (junction, free end, or a
    window/door point that superseded a free end).
    """
    cfg = {**DEFAULTS, **(config or {})}
    cfg["scale_info"] = scale_info

    wall_components = components.get("wall", [])
    window_components = components.get("window", [])
    door_arc_components = components.get("door_arc", [])
    door_origin_components = components.get("door_origin", [])

    node_edges, diag_rejected = build_wall_skeleton_graph(wall_components, cfg["cardinal_tolerance_deg"])
    junction_points, free_ends = _classify_wall_nodes(node_edges)

    all_edges: dict[str, WallSkeletonEdge] = {}
    for edges in node_edges.values():
        for e in edges:
            all_edges[e.id] = e
    wall_edge_list = list(all_edges.values())

    window_points, window_rejected = _detect_window_points(window_components, wall_edge_list, cfg)
    door_points, door_rejected = _detect_door_points(
        door_arc_components, door_origin_components,
        masks.get("door_leaf"), masks.get("door_origin"), masks.get("wall"),
        wall_edge_list, cfg,
    )

    free_points, free_end_rejected = _finalize_free_ends(
        free_ends,
        window_points + door_points,
        cfg["free_end_merge_tol_px"],
        masks,
        cfg["free_end_opening_proximity_px"],
    )

    points = junction_points + window_points + door_points + free_points
    rejected = diag_rejected + window_rejected + door_rejected + free_end_rejected
    _link_skeleton_edges_to_points(wall_edge_list, points)
    return points, rejected, wall_edge_list


def _link_skeleton_edges_to_points(wall_edges: list[WallSkeletonEdge], points: list[GraphPoint], tol_px: float = 10.0) -> None:
    """Record which final point id sits at each end of every skeleton edge,
    by exact (pre-alignment) coordinate match. point_connection.py uses these
    ids directly instead of re-matching by coordinate after point_alignment.py
    has potentially moved points by much more than a tight pixel tolerance."""

    def _id_near(coord: tuple[float, float]) -> Optional[str]:
        best_id, best_dist = None, tol_px
        for p in points:
            dist = math.hypot(p.coordinate[0] - coord[0], p.coordinate[1] - coord[1])
            if dist <= best_dist:
                best_dist, best_id = dist, p.id
        return best_id

    for se in wall_edges:
        se.point_id_at_start = _id_near(se.start)
        se.point_id_at_end = _id_near(se.end)


def _mask_has_evidence_near(coord: tuple[float, float], mask: Optional[np.ndarray], radius_px: float) -> bool:
    if mask is None:
        return False
    nx, ny = int(round(coord[0])), int(round(coord[1]))
    r = int(round(radius_px))
    h, w = mask.shape
    x0, x1 = max(0, nx - r), min(w, nx + r + 1)
    y0, y1 = max(0, ny - r), min(h, ny + r + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    return bool(mask[y0:y1, x0:x1].any())


def _wall_evidence_near(coord: tuple[float, float], wall_components: list[ComponentRecord], radius_px: float) -> bool:
    return any(_mask_has_evidence_near(coord, comp.mask, radius_px) for comp in wall_components)


def build_door_candidate_records(
    door_arc_components: list[ComponentRecord],
    points: list[GraphPoint],
    rejected_evidence: list[RejectedEvidence],
    masks: dict[str, np.ndarray],
    wall_components: list[ComponentRecord],
    scale_info,
    support_probe_px: float = 14.0,
) -> list[DoorCandidateRecord]:
    """Reconcile the final hinge/end points and the rejected-evidence log
    back onto each red ``door_arc`` cluster (task13 "Metrics Requirements").

    Runs independently of ``detect_points()``'s own control flow - by
    construction every accepted door_arc component already produced exactly
    one hinge/end pair or was logged as rejected with that component_id, so
    this just reads the result back out instead of changing what
    ``detect_points`` returns.
    """
    hinge_by_arc: dict[int, GraphPoint] = {}
    end_by_arc: dict[int, GraphPoint] = {}
    for p in points:
        if not p.source_component_ids:
            continue
        if p.point_type == "wall_door_hinge_point":
            hinge_by_arc[p.source_component_ids[0]] = p
        elif p.point_type == "wall_door_end_point":
            end_by_arc[p.source_component_ids[0]] = p

    rejected_by_arc: dict[int, RejectedEvidence] = {}
    for r in rejected_evidence:
        if r.class_name == "door_arc" and r.component_id is not None and r.component_id not in rejected_by_arc:
            rejected_by_arc[r.component_id] = r

    px_to_mm = scale_info.px_to_mm if scale_info is not None else None
    wall_mask = masks.get("wall")
    records: list[DoorCandidateRecord] = []

    for arc in door_arc_components:
        long_edge = float(max(arc.bbox[2] - arc.bbox[0], arc.bbox[3] - arc.bbox[1]))
        hinge = hinge_by_arc.get(arc.component_id)
        end = end_by_arc.get(arc.component_id)

        # task17 "Required Metrics": report the bbox-vertex selection itself
        # (independent of whether a door was ultimately created from it) -
        # select_door_hinge_end_from_bbox is pure, so recomputing it here is
        # cheap and keeps detect_points()'s own return contract unchanged.
        selection = select_door_hinge_end_from_bbox(
            arc.bbox, masks.get("door_origin"), wall_mask, masks.get("door_leaf"), support_probe_px
        )

        if hinge is None or end is None:
            rej = rejected_by_arc.get(arc.component_id)
            records.append(
                DoorCandidateRecord(
                    red_component_id=arc.component_id, red_bbox=arc.bbox, red_bbox_long_edge_px=long_edge,
                    created_door_candidate=False,
                    door_inference_notes=rej.reason if rej else "no hinge/end pair produced",
                    all_four_bbox_vertices=selection.all_vertices,
                    selected_hinge_vertex=selection.hinge,
                    selected_end_vertex=selection.end,
                    hinge_vertex_score=selection.edge_score,
                    end_vertex_score=selection.edge_score,
                    selected_bbox_edge=selection.edge_name,
                    host_wall_alignment_score=selection.host_wall_alignment_score,
                )
            )
            continue

        hinge_support = ["red"]
        if _mask_has_evidence_near(hinge.coordinate, masks.get("door_leaf"), support_probe_px):
            hinge_support.append("orange")
        if _mask_has_evidence_near(hinge.coordinate, masks.get("door_origin"), support_probe_px):
            hinge_support.append("purple")
        if _wall_evidence_near(hinge.coordinate, wall_components, support_probe_px):
            hinge_support.append("black")

        end_support = ["red"]
        if _mask_has_evidence_near(end.coordinate, masks.get("door_origin"), support_probe_px):
            end_support.append("purple")
        if _mask_has_evidence_near(end.coordinate, masks.get("door_leaf"), support_probe_px):
            end_support.append("orange")

        confidence = (len(hinge_support) / 4.0 + len(end_support) / 3.0) / 2.0
        width_px = math.hypot(end.coordinate[0] - hinge.coordinate[0], end.coordinate[1] - hinge.coordinate[1])
        records.append(
            DoorCandidateRecord(
                red_component_id=arc.component_id, red_bbox=arc.bbox, red_bbox_long_edge_px=long_edge,
                created_door_candidate=True,
                scale_candidate_px_to_mm=px_to_mm,
                hinge_candidate_support_classes=hinge_support,
                end_candidate_support_classes=end_support,
                hinge_distance_to_red_bbox_mm=_point_to_bbox_distance(hinge.coordinate, arc.bbox) * (px_to_mm or 0.0),
                end_distance_to_red_bbox_mm=_point_to_bbox_distance(end.coordinate, arc.bbox) * (px_to_mm or 0.0),
                door_confidence=confidence,
                door_inference_notes=(
                    "hinge/end inferred from orange/purple/black evidence near the red cluster"
                    if len(hinge_support) > 2 and len(end_support) > 1
                    else "forced inference - weak or missing orange/purple evidence near the red cluster"
                ),
                all_four_bbox_vertices=selection.all_vertices,
                selected_hinge_vertex=selection.hinge,
                selected_end_vertex=selection.end,
                hinge_vertex_score=selection.edge_score,
                end_vertex_score=selection.edge_score,
                selected_bbox_edge=selection.edge_name,
                host_wall_alignment_score=selection.host_wall_alignment_score,
                door_width_mm=(width_px * px_to_mm) if px_to_mm is not None else None,
            )
        )

    return records


def validate_points(
    points: list[GraphPoint], accepted_door_arc_count: Optional[int] = None
) -> list[ValidationIssue]:
    """Enforce spec_v008 SS10's point-search invariants."""
    issues: list[ValidationIssue] = []

    for p in points:
        if p.point_type not in ALL_POINT_TYPES:
            issues.append(ValidationIssue("unresolved_point_type", f"{p.id} has unresolved point_type {p.point_type}", [p.id]))
        for a in p.attachments:
            if a.direction not in ("left", "right", "up", "down"):
                issues.append(ValidationIssue("non_cardinal_attachment", f"{p.id} direction {a.direction} is not cardinal", [p.id]))

    window_count = sum(1 for p in points if p.point_type == "wall_window_point")
    if window_count % 2 != 0:
        issues.append(ValidationIssue("odd_window_point_count", f"wall_window_point count {window_count} is odd", []))

    hinge_count = sum(1 for p in points if p.point_type == "wall_door_hinge_point")
    end_count = sum(1 for p in points if p.point_type == "wall_door_end_point")
    if hinge_count != end_count:
        issues.append(ValidationIssue("door_hinge_end_mismatch", f"hinge count {hinge_count} != end count {end_count}", []))
    if accepted_door_arc_count is not None and hinge_count != accepted_door_arc_count:
        issues.append(
            ValidationIssue(
                "door_count_mismatch",
                f"hinge count {hinge_count} != accepted door_arc components {accepted_door_arc_count}",
                [],
            )
        )

    return issues
