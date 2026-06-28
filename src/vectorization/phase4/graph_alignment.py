"""Orthogonal graph normalization for the Phase 4 wall centerline graph.

Implements spec_v008 §4 (Wall Graph Normalization) and §5 (Orthogonal Alignment).

Pipeline:
    1. Remove zero-length edges
    2. Remove exact duplicate edges
    3. Cluster near-equal X axes (for vertical edges)
    4. Cluster near-equal Y axes (for horizontal edges)
    5. Snap all edges to horizontal or vertical
    6. Merge collinear overlapping/touching edges (on the same axis)
    7. Split H/V intersections → add new intersection nodes
    8. Return aligned graph: {"nodes": [...], "edges": [...]}
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

# Edges more than this many degrees from H or V are rejected (spec §4).
_REJECT_ANGLE_DEG = 10.0
# Two axis values within this many pixels are considered the same axis.
_AXIS_CLUSTER_TOL_PX = 6.0
# Two endpoints within this many pixels are considered the same node.
_NODE_SNAP_TOL_PX = 4.0
# Intersection detection tolerance (px): how close V-x must be to H-extent etc.
_INTERSECT_TOL_PX = 1.0
# Collinear merge: touching segments (a <= prev_b + tol) are merged.
_MERGE_TOUCH_TOL_PX = 0.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _edge_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))


def _cluster_values(values: list[float], tol: float) -> dict[float, float]:
    """Map each raw value to its cluster representative (median of group)."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    groups: list[list[float]] = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    mapping: dict[float, float] = {}
    for group in groups:
        rep = sum(group) / len(group)
        for v in group:
            mapping[v] = rep
    return mapping


def _snap_node(pt: tuple[float, float], nodes: list[tuple[float, float]], tol: float) -> int:
    """Return index of existing node within tol, or append new node."""
    for i, n in enumerate(nodes):
        if math.hypot(pt[0] - n[0], pt[1] - n[1]) <= tol:
            return i
    nodes.append(pt)
    return len(nodes) - 1


def _merge_intervals(
    intervals: list[tuple[float, float]],
    touch_tol: float = _MERGE_TOUCH_TOL_PX,
) -> list[tuple[float, float]]:
    """Merge overlapping or near-touching intervals.

    Two intervals are merged when the next start is within touch_tol of the
    previous end.  This handles collinear edges that share an endpoint or
    nearly share one.
    """
    if not intervals:
        return []
    sorted_ivs = sorted(intervals)
    merged = [sorted_ivs[0]]
    for a, b in sorted_ivs[1:]:
        prev_a, prev_b = merged[-1]
        if a <= prev_b + touch_tol:
            merged[-1] = (prev_a, max(prev_b, b))
        else:
            merged.append((a, b))
    return merged


def _merge_collinear(
    segments: list[tuple[float, float, float, float]],
    touch_tol: float = _MERGE_TOUCH_TOL_PX,
) -> list[tuple[float, float, float, float]]:
    """Merge touching/overlapping collinear axis-aligned segments."""
    h_by_y: dict[float, list[tuple[float, float]]] = defaultdict(list)
    v_by_x: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for x1, y1, x2, y2 in segments:
        if y1 == y2:
            h_by_y[y1].append((min(x1, x2), max(x1, x2)))
        else:
            v_by_x[x1].append((min(y1, y2), max(y1, y2)))

    result: list[tuple[float, float, float, float]] = []
    for y, intervals in h_by_y.items():
        for a, b in _merge_intervals(intervals, touch_tol):
            result.append((a, y, b, y))
    for x, intervals in v_by_x.items():
        for a, b in _merge_intervals(intervals, touch_tol):
            result.append((x, a, x, b))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_graph(
    graph: dict[str, Any],
    axis_cluster_tol_px: float = _AXIS_CLUSTER_TOL_PX,
    node_snap_tol_px: float = _NODE_SNAP_TOL_PX,
    reject_angle_deg: float = _REJECT_ANGLE_DEG,
) -> dict[str, Any]:
    """Orthogonally normalize a Raster-to-Graph wall centerline graph.

    Args:
        graph: {"nodes": [[x,y],...], "edges": [[x1,y1,x2,y2],...]}

    Returns:
        Aligned graph in same format.  All edges are exactly H or V.
        Also carries "aligned_nodes"/"aligned_edges" as synonym keys.
    """
    raw_edges: list[tuple[float, float, float, float]] = [
        (float(e[0]), float(e[1]), float(e[2]), float(e[3]))
        for e in graph.get("edges", [])
    ]

    # --- 1. Remove zero-length and reject non-orthogonal edges ---
    cleaned: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in raw_edges:
        if math.hypot(x2 - x1, y2 - y1) < 1e-6:
            continue
        angle = _edge_angle_deg(x1, y1, x2, y2)
        # angle from atan2(|dy|,|dx|): 0°=H, 90°=V
        dist_to_h = min(angle, abs(180 - angle))
        dist_to_v = abs(90 - angle)
        if min(dist_to_h, dist_to_v) > reject_angle_deg:
            continue
        cleaned.append((x1, y1, x2, y2))

    # --- 2. Remove exact duplicate edges ---
    seen: set[tuple[float, float, float, float]] = set()
    deduped: list[tuple[float, float, float, float]] = []
    for e in cleaned:
        key = (min(e[0], e[2]), min(e[1], e[3]), max(e[0], e[2]), max(e[1], e[3]))
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    if not deduped:
        return {"nodes": [], "edges": [], "aligned_nodes": [], "aligned_edges": []}

    # --- 3 & 4. Classify each edge and collect axis candidates ---
    h_edges: list[tuple[float, float, float, float]] = []
    v_edges: list[tuple[float, float, float, float]] = []
    h_y_values: list[float] = []
    v_x_values: list[float] = []

    for x1, y1, x2, y2 in deduped:
        angle = _edge_angle_deg(x1, y1, x2, y2)
        if abs(90 - angle) > abs(angle):
            # near-horizontal: cluster on y
            h_edges.append((x1, y1, x2, y2))
            h_y_values.extend([y1, y2])
        else:
            # near-vertical: cluster on x
            v_edges.append((x1, y1, x2, y2))
            v_x_values.extend([x1, x2])

    y_map = _cluster_values(h_y_values, axis_cluster_tol_px)
    x_map = _cluster_values(v_x_values, axis_cluster_tol_px)

    # --- 5. Snap edges to exact H or V using clustered axes ---
    snapped: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in h_edges:
        sy1 = y_map.get(y1, y1)
        sy2 = y_map.get(y2, y2)
        sy = (sy1 + sy2) / 2.0
        sx1, sx2 = min(x1, x2), max(x1, x2)
        if abs(sx2 - sx1) > 1e-6:
            snapped.append((sx1, sy, sx2, sy))

    for x1, y1, x2, y2 in v_edges:
        sx1 = x_map.get(x1, x1)
        sx2 = x_map.get(x2, x2)
        sx = (sx1 + sx2) / 2.0
        sy1, sy2 = min(y1, y2), max(y1, y2)
        if abs(sy2 - sy1) > 1e-6:
            snapped.append((sx, sy1, sx, sy2))

    # --- 6. Merge collinear overlapping/touching edges (before splitting) ---
    # This handles multiple segments on the same wall line that share endpoints.
    merged = _merge_collinear(snapped)

    # --- 7. Find H/V intersections and split edges at crossing points ---
    h_indexed = [(i, e) for i, e in enumerate(merged) if e[1] == e[3]]
    v_indexed = [(i, e) for i, e in enumerate(merged) if e[0] == e[2]]

    split_pts: dict[int, list[float]] = defaultdict(list)
    for hi, (hx1, hy, hx2, _hy2) in h_indexed:
        xmin, xmax = min(hx1, hx2), max(hx1, hx2)
        for vi, (vx, vy1, _vx2, vy2) in v_indexed:
            ymin, ymax = min(vy1, vy2), max(vy1, vy2)
            if (xmin + _INTERSECT_TOL_PX < vx < xmax - _INTERSECT_TOL_PX and
                    ymin + _INTERSECT_TOL_PX < hy < ymax - _INTERSECT_TOL_PX):
                # Interior crossing (T or + junction inside edge extents)
                split_pts[hi].append(vx)
                split_pts[vi].append(hy)
            elif (xmin - _INTERSECT_TOL_PX <= vx <= xmax + _INTERSECT_TOL_PX and
                  ymin - _INTERSECT_TOL_PX <= hy <= ymax + _INTERSECT_TOL_PX):
                # T-junction: vx at endpoint of H or hy at endpoint of V
                # Only split the edge that does NOT already end at the junction
                if abs(vx - hx1) > _INTERSECT_TOL_PX and abs(vx - hx2) > _INTERSECT_TOL_PX:
                    split_pts[hi].append(vx)
                if abs(hy - vy1) > _INTERSECT_TOL_PX and abs(hy - vy2) > _INTERSECT_TOL_PX:
                    split_pts[vi].append(hy)

    result_segments: list[tuple[float, float, float, float]] = []
    for i, e in enumerate(merged):
        x1, y1, x2, y2 = e
        pts = split_pts.get(i, [])
        is_h = (y1 == y2)
        if not pts:
            result_segments.append(e)
            continue
        if is_h:
            xs = sorted({min(x1, x2), max(x1, x2)} | set(
                max(min(x1, x2), min(max(x1, x2), p)) for p in pts
            ))
            for a, b in zip(xs, xs[1:]):
                if abs(b - a) > 1e-6:
                    result_segments.append((a, y1, b, y1))
        else:
            ys = sorted({min(y1, y2), max(y1, y2)} | set(
                max(min(y1, y2), min(max(y1, y2), p)) for p in pts
            ))
            for a, b in zip(ys, ys[1:]):
                if abs(b - a) > 1e-6:
                    result_segments.append((x1, a, x1, b))

    # --- Build final node/edge lists ---
    final_nodes: list[tuple[float, float]] = []
    final_edges: list[tuple[int, int]] = []
    final_edges_raw: list[list[float]] = []

    for x1, y1, x2, y2 in result_segments:
        i1 = _snap_node((x1, y1), final_nodes, node_snap_tol_px)
        i2 = _snap_node((x2, y2), final_nodes, node_snap_tol_px)
        if i1 != i2:
            final_edges.append((i1, i2))
            final_edges_raw.append([x1, y1, x2, y2])

    return {
        "nodes": [[n[0], n[1]] for n in final_nodes],
        "edges": final_edges_raw,
        "aligned_nodes": [[n[0], n[1]] for n in final_nodes],
        "aligned_edges": final_edges_raw,
        "_node_index": final_nodes,
        "_edge_index": final_edges,
    }
