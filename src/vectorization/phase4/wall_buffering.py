"""Buffer connected wall chains into final wall polygon geometry.

Implements spec_v008 §10 (Wall Polygon Generation).

Must-rules:
    - Connect wall edges into chains BEFORE buffering (spec §10, §14.5)
    - Buffer 100mm on each side of centerline (200mm total, spec §10)
    - Use shapely linemerge + buffer(cap_style='flat', join_style='mitre')

task34 (Part C):
    Before calling shapely.linemerge, topology-snap near-identical endpoints
    within a configurable tolerance so that linemerge can actually chain segments
    whose endpoints are nearly (but not exactly) equal due to floating-point drift
    from opening insertion and trimming.

When scale is resolved:
    half_width_px = 100 / px_to_mm

When scale is unknown:
    use a configured preview fallback in pixels; mark as scale_blocked
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import linemerge, unary_union

from ..primitives.scale import ScaleInfo

_WALL_HALF_WIDTH_MM = 100.0
_DEFAULT_PREVIEW_HALF_WIDTH_PX = 8.0
_DEFAULT_SNAP_TOL_PX = 1.5  # snap near-identical endpoints within this many pixels


@dataclass
class WallGeometry:
    """Final wall polygon system."""
    polygon: Optional[Polygon | MultiPolygon]
    half_width_px: float
    wall_thickness_mm: Optional[float]
    scale_blocked: bool
    chain_count: int
    edge_count: int
    # Topology-snap metrics (task34 Part C)
    pre_buffer_node_count: int = 0
    post_snap_node_count: int = 0
    pre_buffer_edge_count: int = 0
    disconnected_endpoint_count: int = 0


def _topology_snap_edges(
    edges: list[list[float]],
    tol: float = _DEFAULT_SNAP_TOL_PX,
) -> tuple[list[list[float]], dict]:
    """Snap near-identical wall edge endpoints to force topological connectivity.

    shapely.linemerge() requires exact float equality to chain two segments.
    After opening insertion and trimming, endpoints that were geometrically
    coincident may differ by sub-pixel floating-point amounts.  This function
    clusters endpoints within `tol` pixels and replaces them with their
    cluster centroid, enabling linemerge to correctly form connected chains.

    Returns:
        (snapped_edges, metrics_dict)
    where metrics_dict contains:
        pre_buffer_node_count, post_snap_node_count, pre_buffer_edge_count,
        disconnected_endpoint_count
    """
    if not edges:
        return [], {
            "pre_buffer_node_count": 0,
            "post_snap_node_count": 0,
            "pre_buffer_edge_count": 0,
            "disconnected_endpoint_count": 0,
        }

    # Collect all endpoints with their source-edge index
    all_pts: list[tuple[float, float]] = []
    for e in edges:
        all_pts.append((e[0], e[1]))
        all_pts.append((e[2], e[3]))

    n = len(all_pts)

    # Union-Find clustering
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    tol_sq = tol * tol
    for i in range(n):
        for j in range(i + 1, n):
            dx = all_pts[i][0] - all_pts[j][0]
            dy = all_pts[i][1] - all_pts[j][1]
            if dx * dx + dy * dy <= tol_sq:
                _union(i, j)

    # Compute cluster centroids
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(i)
        clusters.setdefault(root, []).append(i)

    snapped: list[tuple[float, float]] = [(0.0, 0.0)] * n
    for root, members in clusters.items():
        cx = sum(all_pts[m][0] for m in members) / len(members)
        cy = sum(all_pts[m][1] for m in members) / len(members)
        for m in members:
            snapped[m] = (cx, cy)

    # Rebuild edges with snapped coordinates; drop zero-length and duplicates
    new_edges: list[list[float]] = []
    seen: set[tuple] = set()
    for i, e in enumerate(edges):
        p0 = snapped[2 * i]
        p1 = snapped[2 * i + 1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        if dx * dx + dy * dy < 1e-9:
            continue  # collapsed to zero-length after snap
        k = (round(p0[0], 4), round(p0[1], 4), round(p1[0], 4), round(p1[1], 4))
        kr = (round(p1[0], 4), round(p1[1], 4), round(p0[0], 4), round(p0[1], 4))
        if k in seen or kr in seen:
            continue  # duplicate edge
        seen.add(k)
        new_edges.append([p0[0], p0[1], p1[0], p1[1]])

    # Count unique pre-snap endpoints (by exact float equality)
    pre_node_count = len(set(all_pts))
    post_node_count = len(clusters)
    disconnected = sum(1 for members in clusters.values() if len(members) == 1)

    metrics = {
        "pre_buffer_node_count": pre_node_count,
        "post_snap_node_count": post_node_count,
        "pre_buffer_edge_count": len(edges),
        "disconnected_endpoint_count": disconnected,
    }
    return new_edges, metrics


def buffer_wall_chains(
    wall_edges: list[list[float]],
    scale_info: ScaleInfo,
    preview_half_width_px: float = _DEFAULT_PREVIEW_HALF_WIDTH_PX,
    snap_tol_px: float = _DEFAULT_SNAP_TOL_PX,
) -> WallGeometry:
    """Chain-merge wall centerlines and buffer into a wall polygon system.

    Args:
        wall_edges: List of [x1,y1,x2,y2] remaining after opening trimming
        scale_info: Scale metadata from infer_scale_from_components
        preview_half_width_px: Fallback buffer when scale is unknown
        snap_tol_px: Tolerance for topology-snap pre-processing (task34 Part C)

    Returns:
        WallGeometry with the final wall polygon and topology metrics
    """
    scale_blocked = False
    if scale_info.px_to_mm is not None and scale_info.scale_status in ("resolved", "estimated"):
        half_width_px = _WALL_HALF_WIDTH_MM / scale_info.px_to_mm
        wall_thickness_mm = _WALL_HALF_WIDTH_MM * 2.0
    else:
        half_width_px = preview_half_width_px
        wall_thickness_mm = None
        scale_blocked = True

    if not wall_edges:
        return WallGeometry(
            polygon=None,
            half_width_px=half_width_px,
            wall_thickness_mm=wall_thickness_mm,
            scale_blocked=scale_blocked,
            chain_count=0,
            edge_count=0,
        )

    # Topology snap: bring near-identical endpoints into exact equality (task34 Part C)
    snapped_edges, topo_metrics = _topology_snap_edges(wall_edges, tol=snap_tol_px)

    if not snapped_edges:
        return WallGeometry(
            polygon=None,
            half_width_px=half_width_px,
            wall_thickness_mm=wall_thickness_mm,
            scale_blocked=scale_blocked,
            chain_count=0,
            edge_count=len(wall_edges),
            **topo_metrics,
        )

    # Build shapely LineStrings; merge touching segments into chains
    lines: list[LineString] = []
    for e in snapped_edges:
        x1, y1, x2, y2 = e[0], e[1], e[2], e[3]
        if abs(x2 - x1) < 1e-6 and abs(y2 - y1) < 1e-6:
            continue
        lines.append(LineString([(x1, y1), (x2, y2)]))

    if not lines:
        return WallGeometry(
            polygon=None,
            half_width_px=half_width_px,
            wall_thickness_mm=wall_thickness_mm,
            scale_blocked=scale_blocked,
            chain_count=0,
            edge_count=len(wall_edges),
            **topo_metrics,
        )

    merged = linemerge(lines)
    chains = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    chains = [c for c in chains if not c.is_empty]

    if not chains:
        return WallGeometry(
            polygon=None,
            half_width_px=half_width_px,
            wall_thickness_mm=wall_thickness_mm,
            scale_blocked=scale_blocked,
            chain_count=0,
            edge_count=len(wall_edges),
            **topo_metrics,
        )

    # Buffer each chain, union all into one wall system
    buffered = [
        chain.buffer(half_width_px, cap_style="flat", join_style="mitre")
        for chain in chains
    ]
    wall_system = unary_union(buffered)

    return WallGeometry(
        polygon=wall_system,
        half_width_px=half_width_px,
        wall_thickness_mm=wall_thickness_mm,
        scale_blocked=scale_blocked,
        chain_count=len(chains),
        edge_count=len(wall_edges),
        **topo_metrics,
    )


def wall_polygon_to_svg_paths(
    wall_geom: WallGeometry,
    fill: str = "#1a1a1a",
    stroke: str = "none",
) -> list[str]:
    """Convert wall polygon geometry to SVG path strings."""
    if wall_geom.polygon is None or wall_geom.polygon.is_empty:
        return []

    def _ring_to_d(coords) -> str:
        pts = list(coords)
        if not pts:
            return ""
        path = f"M {pts[0][0]:.2f},{pts[0][1]:.2f}"
        for x, y in pts[1:]:
            path += f" L {x:.2f},{y:.2f}"
        path += " Z"
        return path

    def _poly_to_path(poly: Polygon) -> str:
        exterior_d = _ring_to_d(poly.exterior.coords)
        interior_ds = " ".join(_ring_to_d(ring.coords) for ring in poly.interiors)
        d = exterior_d
        if interior_ds:
            d += " " + interior_ds
        return (
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
            f'fill-rule="evenodd" />'
        )

    poly = wall_geom.polygon
    if isinstance(poly, Polygon):
        return [_poly_to_path(poly)]
    elif hasattr(poly, "geoms"):
        return [_poly_to_path(p) for p in poly.geoms if isinstance(p, Polygon)]
    return []
