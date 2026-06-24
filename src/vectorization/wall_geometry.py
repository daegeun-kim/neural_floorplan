"""Turn final wall/window graph edges into primitives and SVG polygons
(spec_v008 SS14 / SS7 steps 11-12).

The buffer/merge helpers below are unchanged from before the v008 restart:
chain-merging centerline segments that share an endpoint before buffering
gives one continuous mitred-join polygon instead of overlapping
independently-capped rectangles at every corner/junction - exactly what
spec_v008 SS14 requires for "closed filled polygon generated from connected
wall graph edges." What changed is where the segments come from: they are
now the final wall ``GraphEdge`` list point_connection.py built from the
point graph, not contour-traced wall regions.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import linemerge

from .primitives.scale import WALL_MODULES_MM, snap_to_module_mm

if TYPE_CHECKING:
    from .graph_types import GraphEdge
    from .primitives import WallPrimitive, WindowPrimitive

Segment = tuple[tuple[float, float], tuple[float, float]]

DEFAULT_WALL_THICKNESS_PX = 8.0
DEFAULT_WINDOW_HOST_THICKNESS_PX = 16.0
WINDOW_THICKNESS_MM = 100.0


def _snap_to_grid(pt: tuple[float, float], tol: float) -> tuple[float, float]:
    return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)


def merge_connected_chains(segments: list[Segment], tol: float = 1.0) -> list[LineString]:
    """Merge centerline segments that share an endpoint into connected polylines."""
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
    """Single-segment buffer -> filled polygon SVG path (windows)."""
    if math.hypot(end[0] - start[0], end[1] - start[1]) < 1e-6:
        return ""
    line = LineString([start, end])
    poly = line.buffer(half_width_px, cap_style="flat", join_style="mitre")
    return polygon_to_svg_path(poly, fill, extra_attrs)


# ---------------------------------------------------------------------------
# GraphEdge -> primitive conversion (spec_v008 SS14)
# ---------------------------------------------------------------------------


def wall_edges_to_primitives(wall_edges: list["GraphEdge"], scale_info=None) -> list[WallPrimitive]:
    """Final wall GraphEdges -> WallPrimitives, thickness normalized to the
    100mm/200mm modules when scale is known (spec_v008 SS14)."""
    from .primitives import WallPrimitive  # local import: primitives/window.py imports this module

    walls = []
    for edge in wall_edges:
        thickness = edge.thickness_px if edge.thickness_px else DEFAULT_WALL_THICKNESS_PX
        thickness_mm = None
        if scale_info is not None and scale_info.px_to_mm is not None and scale_info.scale_status in ("resolved", "estimated"):
            thickness_mm, _ = snap_to_module_mm(thickness, scale_info, WALL_MODULES_MM)
        walls.append(
            WallPrimitive(
                primitive_id=edge.id,
                start=edge.start,
                end=edge.end,
                thickness=thickness,
                thickness_mm=thickness_mm,
                scale_info=scale_info,
                source_class_ids=[2],
            )
        )
    return walls


def window_edges_to_primitives(window_edges: list["GraphEdge"], scale_info=None) -> list[WindowPrimitive]:
    """Final window GraphEdges -> WindowPrimitives. Window total thickness is
    a fixed 100mm (rule 15) once scale is known, independent of the host
    wall's own thickness module (100mm or 200mm, rule 13) - falls back to
    half the host wall's pixel thickness only while scale is unresolved."""
    from .primitives import WindowPrimitive  # local import: see wall_edges_to_primitives

    windows = []
    for edge in window_edges:
        host_thickness = edge.thickness_px if edge.thickness_px else DEFAULT_WINDOW_HOST_THICKNESS_PX
        if scale_info is not None and scale_info.px_to_mm is not None and scale_info.scale_status in ("resolved", "estimated"):
            thickness = WINDOW_THICKNESS_MM / scale_info.px_to_mm
        else:
            thickness = host_thickness / 2.0
        center = ((edge.start[0] + edge.end[0]) / 2.0, (edge.start[1] + edge.end[1]) / 2.0)
        dx, dy = edge.end[0] - edge.start[0], edge.end[1] - edge.start[1]
        width = math.hypot(dx, dy)
        orientation_angle = math.degrees(math.atan2(dy, dx))
        windows.append(
            WindowPrimitive(
                primitive_id=edge.id,
                center=center,
                width=width,
                orientation_angle=orientation_angle,
                thickness=thickness,
                width_mm=edge.length_mm,
                scale_info=scale_info,
                source_class_ids=[3],
            )
        )
    return windows
