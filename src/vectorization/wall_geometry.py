"""Polygon geometry helpers for rendering walls/windows as closed filled
shapes instead of stroked centerlines (task08), with wall centerlines
chain-merged into connected polylines before buffering (task09).

Buffering each wall centerline segment independently and only unioning the
results gives every segment its own flat end caps - at a shared-endpoint
corner or junction, two independently-capped rectangles overlap instead of
forming one continuous body with a proper mitred corner. `merge_connected_chains`
joins segments that share an endpoint into longer polylines first (stopping
at true free ends and 3+-way junctions, which a single LineString cannot
represent), so `segments_to_polygon` only puts a flat cap where the wall
evidence actually ends.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import linemerge

if TYPE_CHECKING:
    # Deferred to avoid a circular import: primitives/door.py and
    # primitives/window.py import this module's buffer helpers at runtime.
    from .primitives import WallPrimitive
    from .primitives.scale import ScaleInfo

Segment = tuple[tuple[float, float], tuple[float, float]]


def _snap_to_grid(pt: tuple[float, float], tol: float) -> tuple[float, float]:
    return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)


def merge_connected_chains(segments: list[Segment], tol: float = 1.0) -> list[LineString]:
    """Merge centerline segments that share an endpoint into connected polylines.

    Endpoints are snapped to a `tol`-px grid first so that two segments which
    are meant to touch (e.g. after splitting or endpoint-snapping elsewhere
    in the pipeline) compare exactly equal despite minor floating-point
    drift - genuine T-junctions (one wall's endpoint landing mid-span on
    another wall) are untouched by this, since that point is still not an
    endpoint of the through-wall. `shapely.ops.linemerge` then joins any
    segments sharing an endpoint into one LineString, preserving vertex
    order, while correctly leaving 3+-way junctions and disconnected
    fragments as separate chains (a LineString cannot represent a branch).
    """
    lines = []
    for start, end in segments:
        s, e = _snap_to_grid(start, tol), _snap_to_grid(end, tol)
        if math.hypot(e[0] - s[0], e[1] - s[1]) < 1e-9:
            continue
        lines.append(LineString([s, e]))
    if not lines:
        return []
    merged = linemerge(lines)
    if merged.is_empty:
        return []
    return list(merged.geoms) if hasattr(merged, "geoms") else [merged]


def segments_to_polygon(
    segments: list[Segment], half_width_px: float, merge_tolerance_px: float = 1.0
) -> Polygon | MultiPolygon | None:
    """Chain-merge centerline segments, then buffer and union into one wall polygon system."""
    chains = merge_connected_chains(segments, tol=merge_tolerance_px)
    if not chains:
        return None
    buffers = [chain.buffer(half_width_px, cap_style="flat", join_style="mitre") for chain in chains]
    return shapely.union_all(buffers)


def _ring_to_subpath(coords) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    return f"M {pts} Z"


def polygon_to_svg_path(geom, fill: str, extra_attrs: str = "") -> str:
    """Convert a shapely Polygon/MultiPolygon (possibly with holes) into one
    SVG <path>, using fill-rule="evenodd" so interior rings render as holes."""
    if geom is None or geom.is_empty:
        return ""
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    subpaths = []
    for poly in polys:
        if poly.is_empty:
            continue
        subpaths.append(_ring_to_subpath(poly.exterior.coords))
        for interior in poly.interiors:
            subpaths.append(_ring_to_subpath(interior.coords))
    if not subpaths:
        return ""
    d = " ".join(subpaths)
    attrs = f" {extra_attrs}" if extra_attrs else ""
    return f'<path d="{d}" fill="{fill}" fill-rule="evenodd" stroke="none"{attrs} />'


def buffer_segment_polygon_svg(
    start: tuple[float, float],
    end: tuple[float, float],
    half_width_px: float,
    fill: str,
    extra_attrs: str = "",
) -> str:
    """Single-segment buffer -> filled polygon SVG path (windows/door-origin/door-leaf)."""
    if math.hypot(end[0] - start[0], end[1] - start[1]) < 1e-6:
        return ""
    line = LineString([start, end])
    poly = line.buffer(half_width_px, cap_style="flat", join_style="mitre")
    return polygon_to_svg_path(poly, fill, extra_attrs)


def _project_point_onto_line(
    point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> tuple[tuple[float, float], float, float]:
    """Project `point` onto the infinite line through a->b.

    Returns (projected_point, distance_to_line, t), where t is the
    parametric position along a->b (0=a, 1=b; t can fall outside [0, 1] if
    the projection lands beyond the actual segment).
    """
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


def connect_dangling_wall_endpoints(
    walls: list[WallPrimitive],
    max_connect_dist_px: float = 20.0,
    candidates: list[WallPrimitive] | None = None,
) -> None:
    """Snap dangling endpoints of `walls` onto nearby wall lines, in place.

    For every endpoint of every wall in `walls`, check every *other* wall in
    `candidates` (defaults to `walls` itself) for its infinite line; if the
    projection lands within `max_connect_dist_px` of the endpoint, and within
    (or only slightly beyond) that wall's actual span, snap the endpoint onto
    it. This closes small real gaps between an inner wall and the wall
    network it should be touching (outer loop or another inner wall) without
    requiring the two original detections to already be almost coincident.

    Pass `candidates` explicitly (e.g. outer+inner walls) when `walls` is a
    subset (e.g. only inner walls) so the already-closed outer loop is used
    as a snap target but never itself perturbed.
    """
    pool = candidates if candidates is not None else walls
    for wall in walls:
        for attr in ("start", "end"):
            pt = getattr(wall, attr)
            best_point = None
            best_dist = max_connect_dist_px
            for other in pool:
                if other is wall:
                    continue
                seg_len = math.hypot(
                    other.end[0] - other.start[0], other.end[1] - other.start[1]
                )
                if seg_len < 1e-6:
                    continue
                proj, dist, t = _project_point_onto_line(pt, other.start, other.end)
                overhang = max_connect_dist_px / seg_len
                if dist < best_dist and -overhang <= t <= 1.0 + overhang:
                    best_dist = dist
                    best_point = proj
            if best_point is not None:
                setattr(wall, attr, best_point)


def snap_inner_endpoints_to_outer_wall_mm(
    inner_walls: list[WallPrimitive],
    outer_walls: list[WallPrimitive],
    scale_info: ScaleInfo,
    threshold_mm: float = 500.0,
) -> dict[str, list[str]]:
    """Project inner-wall endpoints within `threshold_mm` of the outer wall
    loop onto the nearest outer wall segment, in place (task10).

    Requires `scale_info.px_to_mm` to be resolved/estimated - this rule is
    explicitly real-world-scale-only (task10: "do not add a pixel fallback
    path for this rule"), so callers must check scale_info themselves before
    calling and must not invoke this with an unresolved scale. Never moves
    the outer wall loop itself, only inner wall endpoints.

    Returns {wall.primitive_id: [endpoint names snapped]} for metrics/debug
    bookkeeping, e.g. {"wall_inner_0003": ["start"]}.
    """
    if scale_info.px_to_mm is None or scale_info.scale_status not in ("resolved", "estimated"):
        raise ValueError("scale must be resolved/estimated before mm-based outer-wall snapping")

    threshold_px = threshold_mm / scale_info.px_to_mm
    snapped: dict[str, list[str]] = {}
    for wall in inner_walls:
        for attr in ("start", "end"):
            pt = getattr(wall, attr)
            best_point = None
            best_dist = threshold_px
            for outer in outer_walls:
                seg_len = math.hypot(outer.end[0] - outer.start[0], outer.end[1] - outer.start[1])
                if seg_len < 1e-6:
                    continue
                proj, dist, t = _project_point_onto_line(pt, outer.start, outer.end)
                # Tight overhang tolerance (not max_connect_dist_px-scaled like
                # connect_dangling_wall_endpoints) - threshold_mm is already a
                # generous real-world distance, so a wide overhang window here
                # risks snapping onto the wrong outer edge near a corner.
                overhang = 0.02
                if dist < best_dist and -overhang <= t <= 1.0 + overhang:
                    best_dist = dist
                    best_point = proj
            if best_point is not None:
                setattr(wall, attr, best_point)
                snapped.setdefault(wall.primitive_id, []).append(attr)
    return snapped
