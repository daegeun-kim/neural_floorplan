"""Generate orthogonal wall graph labels from CubiCasa5K model.svg files (spec_v003-1).

Pipeline: render/reuse the existing wall+window masks, bridge the wall mask
across door/window openings using their exact source SVG geometry (so the
skeleton passes straight through every opening as one continuous chain),
then reuse the v008 mask-to-vector pipeline's skeleton-walk/orthogonalization
machinery (src/vectorization/{components,point_detection,point_alignment}.py)
to build wall nodes/edges. Door/window centers are added afterward as
annotation-only nodes projected onto their nearest final wall edge - they
never split a wall edge or act as an edge endpoint (spec_v003-1 SS3/SS9),
which is why src/vectorization/point_connection.py's edge builder (designed
to split wall edges at openings for the prediction pipeline) is not reused
here; a simpler dedicated edge builder is used instead.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from lxml import etree
from PIL import Image, ImageDraw

from src import generate_semantic_masks
from src.vectorization.components import extract_components
from src.vectorization.graph_types import GraphPoint
from src.vectorization.point_detection import (
    WallSkeletonEdge,
    _classify_wall_nodes,
    _finalize_free_ends,
    _link_skeleton_edges_to_points,
    _point_to_wall_distance,
    build_wall_skeleton_graph,
)

logger = logging.getLogger(__name__)

GRAPH_FILENAME = "wall_graph.json"
METRICS_FILENAME = "wall_graph_metrics.json"
DEBUG_PNG_FILENAME = "wall_graph_debug.png"
DEBUG_SVG_FILENAME = "wall_graph_debug.svg"

DEFAULTS: dict = {
    "door_swing_strip_dilate_px": 5,
    "min_area_px": 16.0,
    "cardinal_tolerance_deg": 25.0,
    "free_end_merge_tol_px": 8.0,
    "node_merge_tol_px": 6.0,
    "min_edge_length_px": 6.0,
    "min_component_nodes": 3,
    "opening_host_distance_factor": 3.0,
    "opening_host_min_distance_px": 30.0,
    "min_wall_node_count": 2,
    "max_wall_node_count": 2000,
    # Skeletonize routinely leaves a harmless 1-2px non-cardinal "spur" chain
    # right at ordinary corners/T-junctions (most often where two wall bands
    # of equal thickness meet) - that noise scales with junction count, not
    # with how clean the wall evidence actually is. So the diagonal-chain
    # budget below scales with node count instead of a flat ratio of edges,
    # to avoid flagging ordinary floor plans while still catching samples
    # whose wall evidence is genuinely, mostly non-orthogonal.
    "diagonal_chains_per_node_budget": 1.5,
    "diagonal_chains_absolute_floor": 5,
    "max_dropped_components": 5,
    "min_mask_coverage_ratio": 0.4,
}

_DEBUG_COLORS_RGB = {
    "wall_node": (220, 40, 40),
    "door_center": (240, 150, 30),
    "window_center": (40, 170, 60),
}
_DEBUG_COLORS_SVG = {
    "wall_node": "red",
    "door_center": "orange",
    "window_center": "green",
}
_EDGE_COLOR_RGB = (40, 80, 220)


# ---------------------------------------------------------------------------
# SVG coordinate transform
# ---------------------------------------------------------------------------


def svg_to_pixel_transform(
    svg_root: etree._Element, width: int, height: int
) -> tuple[float, float, float, float]:
    """Return (vb_x, vb_y, scale_x, scale_y) mapping SVG-space points to the
    raster pixel space used by the generated masks."""
    vb = svg_root.get("viewBox", "")
    if vb:
        parts = vb.split()
        if len(parts) == 4:
            try:
                vb_x, vb_y, vb_w, vb_h = (float(p) for p in parts)
                if vb_w > 0 and vb_h > 0:
                    return vb_x, vb_y, width / vb_w, height / vb_h
            except ValueError:
                pass
    return 0.0, 0.0, 1.0, 1.0


def _apply_transform(
    transform: tuple[float, float, float, float], point: tuple[float, float]
) -> tuple[float, float]:
    vb_x, vb_y, sx, sy = transform
    x, y = point
    return (x - vb_x) * sx, (y - vb_y) * sy


# ---------------------------------------------------------------------------
# Opening evidence (door/window openings, in pixel space)
# ---------------------------------------------------------------------------


@dataclass
class OpeningEvidence:
    kind: str  # "door" | "window"
    center: tuple[float, float]
    width_px: float
    orientation: str  # "horizontal" | "vertical"
    polygon_px: list[tuple[float, float]]
    source_bbox: tuple[float, float, float, float]


def _window_opening_polygon(window_el: etree._Element) -> etree._Element | None:
    """The Window group's own direct-child polygon - the opening rectangle
    (same wall-thickness span as the parent Wall polygon), not the nested
    Glass/Panel descendants."""
    for child in window_el:
        if child.tag is etree.Comment:
            continue
        if generate_semantic_masks._local_name(child) == "polygon":
            return child
    return None


def _opening_from_points(
    points: list[tuple[float, float]],
    transform: tuple[float, float, float, float],
    kind: str,
) -> OpeningEvidence | None:
    if len(points) < 3:
        return None
    px_points = [_apply_transform(transform, p) for p in points]
    c1, c2 = generate_semantic_masks._bbox_centerline(px_points)
    center = ((c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0)
    width = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
    orientation = "horizontal" if abs(c2[0] - c1[0]) >= abs(c2[1] - c1[1]) else "vertical"
    xs = [p[0] for p in px_points]
    ys = [p[1] for p in px_points]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return OpeningEvidence(
        kind=kind, center=center, width_px=width, orientation=orientation,
        polygon_px=px_points, source_bbox=bbox,
    )


def collect_openings(
    svg_root: etree._Element, transform: tuple[float, float, float, float]
) -> list[OpeningEvidence]:
    """Walk every Wall's Window/Door children and return their opening
    evidence (center, width, orientation, polygon) in pixel space."""
    openings: list[OpeningEvidence] = []
    containers = generate_semantic_masks._find_floor_containers(svg_root) or [svg_root]

    for container in containers:
        for child in container:
            if generate_semantic_masks._classify_floor_child(child) != "wall":
                continue
            for window_el in generate_semantic_masks._get_window_children(child):
                poly = _window_opening_polygon(window_el)
                if poly is None:
                    continue
                points = generate_semantic_masks._parse_points(poly.get("points", ""))
                ev = _opening_from_points(points, transform, "window")
                if ev:
                    openings.append(ev)
            for door_el in generate_semantic_masks._get_door_children(child):
                for poly in generate_semantic_masks._find_threshold_polygons(door_el):
                    points = generate_semantic_masks._parse_points(poly.get("points", ""))
                    ev = _opening_from_points(points, transform, "door")
                    if ev:
                        openings.append(ev)
    return openings


def strip_door_swing_evidence(wall_mask: np.ndarray, masks_dir: Path, dilate_px: int = 5) -> np.ndarray:
    """Remove the door swing-arc/leaf stroke from the wall mask.

    The original SVG's Door > Panel > path (the swing-arc/leaf visual) has no
    fill/stroke of its own - it inherits stroke="#000000" from the Wall/Door
    group ancestors - so generate_semantic_masks' "wall" category render
    (which keeps the whole Wall subtree as-is) picks up that curved stroke as
    if it were wall evidence. The skeleton walk then orthogonalizes that
    curve into a spurious little stair-step loop near every door.
    door_arc_mask.png/door_leaf_mask.png are synthetic renders of that exact
    same swing geometry (spec_v005 run3), so subtracting their (dilated, to
    fully cover the stroke width) footprint cleanly removes it without
    touching the real wall band, which sits in a spatially distinct region.
    """
    stripped = wall_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    for name in ("door_arc_mask.png", "door_leaf_mask.png"):
        path = masks_dir / name
        if not path.exists():
            continue
        evidence = np.array(Image.open(path).convert("L"))
        evidence = np.where(evidence > 0, 255, 0).astype(np.uint8)
        evidence = cv2.dilate(evidence, kernel)
        stripped[evidence > 0] = 0
    return stripped


def bridge_wall_mask(wall_mask: np.ndarray, openings: list[OpeningEvidence]) -> np.ndarray:
    """Fill each opening's exact source rectangle back into the wall mask, so
    the wall evidence is continuous straight through every door/window
    (spec_v003-1 SS6/SS9: the host wall edge must remain unsplit)."""
    bridged = wall_mask.copy()
    for ev in openings:
        pts = np.array([[round(x), round(y)] for x, y in ev.polygon_px], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(bridged, [pts], 255)
    return bridged


# ---------------------------------------------------------------------------
# Near-duplicate node merging (spec_v003-1 SS9 step 1)
# ---------------------------------------------------------------------------


def merge_near_duplicate_points(
    points: list[GraphPoint], tol_px: float
) -> tuple[list[GraphPoint], dict[str, str]]:
    """Collapse clusters of points within tol_px of each other into one node
    (centroid coordinate). Skeletonize commonly leaves a handful of separate
    junction/free-end pixels within a couple of pixels of each other right at
    a real architectural corner (e.g. where a partition wall's band meets an
    exterior wall's band of the same width) - without this, those become
    spurious near-duplicate nodes joined only by short, non-cardinal
    "spur" chains that get rejected as diagonal, fragmenting the graph.
    Returns (merged_points, old_id -> merged_id)."""
    n = len(points)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i].coordinate[0] - points[j].coordinate[0]
            dy = points[i].coordinate[1] - points[j].coordinate[1]
            if math.hypot(dx, dy) <= tol_px:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[GraphPoint] = []
    id_remap: dict[str, str] = {}
    for counter, indices in enumerate(groups.values(), start=1):
        avg_x = sum(points[i].coordinate[0] for i in indices) / len(indices)
        avg_y = sum(points[i].coordinate[1] for i in indices) / len(indices)
        new_id = f"mergedpt_{counter}"
        merged.append(GraphPoint(id=new_id, point_type="wall_point", coordinate=(avg_x, avg_y)))
        for i in indices:
            id_remap[points[i].id] = new_id
    return merged, id_remap


def snap_shared_axes(points: list[GraphPoint], wall_edges: list[WallSkeletonEdge]) -> None:
    """Mutate points in place so every edge is exactly orthogonal (spec_v003-1
    SS9: "each edge must be x1==x2 or y1==y2"). Every point transitively
    connected through horizontal skeleton edges is forced to one shared y;
    every point transitively connected through vertical edges is forced to
    one shared x. This is a transitive union-find, unlike
    src.vectorization.point_alignment._assert_wall_edge_axes's plain pairwise
    averaging - a corridor of 3+ collinear points (or any point of degree>=3,
    e.g. a T-junction) needs the transitive version to land on one single
    consistent value instead of a slightly different average per edge."""
    n = len(points)
    index_by_id = {p.id: i for i, p in enumerate(points)}
    parent_x = list(range(n))  # grouped by vertical edges -> must share x
    parent_y = list(range(n))  # grouped by horizontal edges -> must share y

    def find(parent: list[int], i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(parent: list[int], i: int, j: int) -> None:
        ri, rj = find(parent, i), find(parent, j)
        if ri != rj:
            parent[ri] = rj

    for se in wall_edges:
        i = index_by_id.get(se.point_id_at_start) if se.point_id_at_start else None
        j = index_by_id.get(se.point_id_at_end) if se.point_id_at_end else None
        if i is None or j is None or i == j:
            continue
        if se.dir_from_start in ("left", "right"):
            union(parent_y, i, j)
        else:
            union(parent_x, i, j)

    def apply_axis(parent: list[int], axis_index: int) -> None:
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(parent, i), []).append(i)
        for indices in groups.values():
            if len(indices) < 2:
                continue
            avg = sum(points[i].coordinate[axis_index] for i in indices) / len(indices)
            for i in indices:
                coord = list(points[i].coordinate)
                coord[axis_index] = avg
                points[i].coordinate = (coord[0], coord[1])

    apply_axis(parent_y, 1)
    apply_axis(parent_x, 0)


# ---------------------------------------------------------------------------
# Final wall edges (no opening-splitting - see module docstring)
# ---------------------------------------------------------------------------


@dataclass
class _FinalEdge:
    id: int
    start_node: int
    end_node: int
    start: tuple[float, float]
    end: tuple[float, float]


def _build_raw_edges(
    points: list, wall_edges: list[WallSkeletonEdge], min_edge_length_px: float
) -> tuple[list[tuple[str, str, tuple[float, float], tuple[float, float]]], int]:
    """One edge per skeleton chain, connecting its two endpoint point ids
    directly - unlike point_connection.build_wall_edges, no window/door host
    points are inserted along the way."""
    points_by_id = {p.id: p for p in points}
    seen_pairs: set[frozenset] = set()
    edges: list[tuple[str, str, tuple[float, float], tuple[float, float]]] = []
    dropped_short = 0

    for se in wall_edges:
        pa_id, pb_id = se.point_id_at_start, se.point_id_at_end
        if pa_id is None or pb_id is None or pa_id == pb_id:
            continue
        pa, pb = points_by_id.get(pa_id), points_by_id.get(pb_id)
        if pa is None or pb is None:
            continue
        length = math.hypot(
            pa.coordinate[0] - pb.coordinate[0], pa.coordinate[1] - pb.coordinate[1]
        )
        if length < min_edge_length_px:
            dropped_short += 1
            continue
        key = frozenset((pa_id, pb_id))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        edges.append((pa_id, pb_id, pa.coordinate, pb.coordinate))

    return edges, dropped_short


def _filter_small_components(
    points: list,
    edges: list[tuple[str, str, tuple[float, float], tuple[float, float]]],
    min_component_nodes: int,
) -> tuple[set[str], int, int, int]:
    """Drop connected components smaller than min_component_nodes as noise.
    Returns (kept_point_ids, dropped_component_count, dropped_node_count,
    kept_component_count)."""
    g = nx.Graph()
    for p in points:
        g.add_node(p.id)
    for pa_id, pb_id, _, _ in edges:
        g.add_edge(pa_id, pb_id)

    keep_ids: set[str] = set()
    dropped_components = 0
    dropped_nodes = 0
    kept_components = 0
    for comp in nx.connected_components(g):
        if len(comp) >= min_component_nodes:
            keep_ids |= comp
            kept_components += 1
        else:
            dropped_components += 1
            dropped_nodes += len(comp)
    return keep_ids, dropped_components, dropped_nodes, kept_components


def _renumber_nodes(
    points: list,
    edges: list[tuple[str, str, tuple[float, float], tuple[float, float]]],
    keep_ids: set[str],
) -> tuple[list[dict], list[_FinalEdge]]:
    kept_points = [p for p in points if p.id in keep_ids]
    id_map = {p.id: i for i, p in enumerate(kept_points)}
    nodes = [
        {"id": id_map[p.id], "type": "wall_node", "x": round(p.coordinate[0], 2), "y": round(p.coordinate[1], 2)}
        for p in kept_points
    ]
    final_edges: list[_FinalEdge] = []
    for pa_id, pb_id, pa_xy, pb_xy in edges:
        if pa_id not in keep_ids or pb_id not in keep_ids:
            continue
        final_edges.append(
            _FinalEdge(
                id=len(final_edges), start_node=id_map[pa_id], end_node=id_map[pb_id],
                start=pa_xy, end=pb_xy,
            )
        )
    return nodes, final_edges


def _host_opening(
    center: tuple[float, float], final_edges: list[_FinalEdge], max_dist_px: float
):
    best, best_dist = None, max_dist_px
    for fe in final_edges:
        dist = _point_to_wall_distance(center, fe)
        if dist <= best_dist:
            best_dist, best = dist, fe
    return best


def _mask_coverage_ratio(
    final_edges: list[_FinalEdge], bridged_mask: np.ndarray, line_thickness_px: float
) -> float:
    h, w = bridged_mask.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    thickness = max(1, int(round(line_thickness_px)))
    for fe in final_edges:
        cv2.line(
            canvas,
            (int(round(fe.start[0])), int(round(fe.start[1]))),
            (int(round(fe.end[0])), int(round(fe.end[1]))),
            255, thickness=thickness,
        )
    wall_fg = bridged_mask > 0
    graph_fg = canvas > 0
    union = int(np.logical_or(wall_fg, graph_fg).sum())
    if union == 0:
        return 1.0
    inter = int(np.logical_and(wall_fg, graph_fg).sum())
    return inter / union


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------


def _backdrop_image(bridged_mask: np.ndarray, width: int, height: int) -> Image.Image:
    base = np.full((height, width, 3), 255, dtype=np.uint8)
    base[bridged_mask > 0] = (210, 210, 210)
    return Image.fromarray(base)


def _draw_overlay(img: Image.Image, nodes: list[dict], edges: list[dict]) -> None:
    draw = ImageDraw.Draw(img)
    node_xy = {n["id"]: (n["x"], n["y"]) for n in nodes}
    for e in edges:
        x1, y1 = node_xy[e["start"]]
        x2, y2 = node_xy[e["end"]]
        draw.line([(x1, y1), (x2, y2)], fill=_EDGE_COLOR_RGB, width=3)
    r = 5
    for n in nodes:
        x, y = n["x"], n["y"]
        color = _DEBUG_COLORS_RGB[n["type"]]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def write_debug_outputs(
    masks_dir: Path,
    bridged_mask: np.ndarray,
    nodes: list[dict],
    edges: list[dict],
    width: int,
    height: int,
) -> None:
    backdrop = _backdrop_image(bridged_mask, width, height)

    png_img = backdrop.copy()
    _draw_overlay(png_img, nodes, edges)
    png_img.save(masks_dir / DEBUG_PNG_FILENAME)

    buf = io.BytesIO()
    backdrop.save(buf, format="PNG")
    b64 = b64encode(buf.getvalue()).decode("ascii")

    node_xy = {n["id"]: (n["x"], n["y"]) for n in nodes}
    lines = [
        f'<line x1="{node_xy[e["start"]][0]:.2f}" y1="{node_xy[e["start"]][1]:.2f}" '
        f'x2="{node_xy[e["end"]][0]:.2f}" y2="{node_xy[e["end"]][1]:.2f}" '
        f'stroke="blue" stroke-width="3" />'
        for e in edges
    ]
    circles = [
        f'<circle cx="{n["x"]:.2f}" cy="{n["y"]:.2f}" r="5" fill="{_DEBUG_COLORS_SVG[n["type"]]}" />'
        for n in nodes
    ]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<image href="data:image/png;base64,{b64}" x="0" y="0" width="{width}" height="{height}" />'
        + "".join(lines) + "".join(circles) +
        "</svg>"
    )
    (masks_dir / DEBUG_SVG_FILENAME).write_text(svg, encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-sample pipeline
# ---------------------------------------------------------------------------


def generate_wall_graph(
    sample_dir: Path, overwrite: bool = False, verbose: bool = False, config: dict | None = None,
) -> dict:
    """Process one sample directory. Returns a status dict."""
    cfg = {**DEFAULTS, **(config or {})}

    svg_path = sample_dir / generate_semantic_masks.SVG_NAME
    if not svg_path.exists():
        logger.warning("No model.svg — skipping: %s", sample_dir)
        return {"status": "missing_svg"}

    masks_dir = sample_dir / generate_semantic_masks.MASKS_DIR
    graph_path = masks_dir / GRAPH_FILENAME
    if graph_path.exists() and not overwrite:
        logger.debug("Already exists, skipping: %s", sample_dir)
        return {"status": "skipped"}

    masks_dir.mkdir(exist_ok=True)

    wall_mask_path = masks_dir / "wall_mask.png"
    window_mask_path = masks_dir / "window_mask.png"
    if not wall_mask_path.exists() or not window_mask_path.exists():
        generate_semantic_masks.generate_masks(sample_dir)
    if not wall_mask_path.exists():
        return {"status": "failed", "reason": "no_wall_mask"}

    try:
        svg_root = generate_semantic_masks._parse_svg(svg_path)
    except Exception:
        logger.exception("SVG parse error: %s", svg_path)
        return {"status": "failed", "reason": "svg_parse_error"}

    reference = sample_dir / "model_clean.png"
    width, height = generate_semantic_masks._get_dimensions(svg_root, reference)
    transform = svg_to_pixel_transform(svg_root, width, height)
    openings = collect_openings(svg_root, transform)

    wall_mask_raw = np.array(Image.open(wall_mask_path).convert("L"))
    wall_mask = np.where(wall_mask_raw > 0, 255, 0).astype(np.uint8)
    wall_mask = strip_door_swing_evidence(wall_mask, masks_dir, cfg["door_swing_strip_dilate_px"])
    bridged = bridge_wall_mask(wall_mask, openings)

    components, _rejected_small = extract_components(bridged, "wall", min_area_px=cfg["min_area_px"])
    node_edges, diagonal_rejected = build_wall_skeleton_graph(components, cfg["cardinal_tolerance_deg"])
    junction_points, free_ends = _classify_wall_nodes(node_edges)
    free_points, _free_rejected = _finalize_free_ends(
        free_ends, junction_points, cfg["free_end_merge_tol_px"], {}, 0.0
    )
    points = junction_points + free_points

    all_se: dict[str, WallSkeletonEdge] = {}
    for se_list in node_edges.values():
        for se in se_list:
            all_se[se.id] = se
    wall_edge_list = list(all_se.values())

    _link_skeleton_edges_to_points(wall_edge_list, points)

    merged_points, id_remap = merge_near_duplicate_points(points, cfg["node_merge_tol_px"])
    for se in wall_edge_list:
        if se.point_id_at_start is not None:
            se.point_id_at_start = id_remap.get(se.point_id_at_start)
        if se.point_id_at_end is not None:
            se.point_id_at_end = id_remap.get(se.point_id_at_end)

    snap_shared_axes(merged_points, wall_edge_list)

    raw_edges, dropped_short = _build_raw_edges(merged_points, wall_edge_list, cfg["min_edge_length_px"])
    keep_ids, dropped_components, dropped_nodes, kept_components = _filter_small_components(
        merged_points, raw_edges, cfg["min_component_nodes"]
    )
    nodes, final_edges = _renumber_nodes(merged_points, raw_edges, keep_ids)
    edges_json = [{"id": fe.id, "start": fe.start_node, "end": fe.end_node} for fe in final_edges]

    thicknesses = [se.thickness for se in wall_edge_list if se.thickness]
    median_thickness = float(np.median(thicknesses)) if thicknesses else 16.0
    max_host_dist_px = max(
        cfg["opening_host_distance_factor"] * median_thickness, cfg["opening_host_min_distance_px"]
    )

    opening_nodes: list[dict] = []
    unhosted = 0
    next_id = len(nodes)
    for ev in openings:
        host = _host_opening(ev.center, final_edges, max_host_dist_px)
        if host is None:
            unhosted += 1
            continue
        opening_nodes.append(
            {
                "id": next_id,
                "type": "door_center" if ev.kind == "door" else "window_center",
                "x": round(ev.center[0], 2),
                "y": round(ev.center[1], 2),
                "host_edge": host.id,
                "opening_width_px": round(ev.width_px, 2),
                "orientation": ev.orientation,
                "source_bbox": [round(v, 2) for v in ev.source_bbox],
            }
        )
        next_id += 1

    all_nodes = nodes + opening_nodes
    coverage_ratio = _mask_coverage_ratio(final_edges, bridged, median_thickness)

    reasons: list[str] = []
    if len(nodes) < cfg["min_wall_node_count"]:
        reasons.append("too_few_nodes")
    if len(nodes) > cfg["max_wall_node_count"]:
        reasons.append("too_many_nodes")
    if not edges_json:
        reasons.append("no_edges")
    diagonal_budget = (
        cfg["diagonal_chains_per_node_budget"] * max(len(nodes), 1) + cfg["diagonal_chains_absolute_floor"]
    )
    if len(diagonal_rejected) > diagonal_budget:
        reasons.append("too_many_diagonal_chains")
    if dropped_components > cfg["max_dropped_components"]:
        reasons.append("too_many_tiny_components")
    if coverage_ratio < cfg["min_mask_coverage_ratio"]:
        reasons.append("graph_far_from_wall_mask")
    if edges_json and (len(nodes) - len(edges_json)) > kept_components:
        reasons.append("edge_node_count_inconsistent")

    status = "unusable" if reasons else "ok"

    graph = {
        "sample_id": sample_dir.name,
        "source_svg": generate_semantic_masks.SVG_NAME,
        "coordinate_space": "raster",
        "image_width": width,
        "image_height": height,
        "nodes": all_nodes,
        "edges": edges_json,
    }
    with open(graph_path, "w") as f:
        json.dump(graph, f, indent=2)

    metrics = {
        "status": status,
        "reasons": reasons,
        "wall_node_count": len(nodes),
        "edge_count": len(edges_json),
        "door_center_count": sum(1 for n in opening_nodes if n["type"] == "door_center"),
        "window_center_count": sum(1 for n in opening_nodes if n["type"] == "window_center"),
        "unhosted_openings": unhosted,
        "diagonal_rejected_chains": len(diagonal_rejected),
        "dropped_short_stub_edges": dropped_short,
        "dropped_tiny_components": dropped_components,
        "dropped_tiny_component_nodes": dropped_nodes,
        "mask_coverage_ratio": round(coverage_ratio, 4),
    }
    with open(masks_dir / METRICS_FILENAME, "w") as f:
        json.dump(metrics, f, indent=2)

    write_debug_outputs(masks_dir, bridged, all_nodes, edges_json, width, height)

    if verbose:
        logger.info(
            "%s -> %s (%d wall nodes, %d edges, %d openings hosted, %d unhosted)",
            sample_dir.name, status, len(nodes), len(edges_json), len(opening_nodes), unhosted,
        )

    return {"status": status, "reasons": reasons}


# ---------------------------------------------------------------------------
# Dataset-level processing
# ---------------------------------------------------------------------------


def process_dataset(
    root_dir: Path,
    overwrite: bool = False,
    verbose: bool = False,
    limit: int | None = None,
    config: dict | None = None,
) -> dict:
    """Process all immediate subdirectories of root_dir."""
    sample_dirs = sorted(d for d in root_dir.iterdir() if d.is_dir())
    if limit is not None:
        sample_dirs = sample_dirs[:limit]
    total = len(sample_dirs)
    counts = {
        "processed": 0, "ok": 0, "unusable": 0,
        "skipped_existing": 0, "missing_svg": 0, "failed": 0,
    }

    for i, sample_dir in enumerate(sample_dirs, 1):
        logger.info("[%d/%d] %s", i, total, sample_dir.name)
        result = generate_wall_graph(sample_dir, overwrite, verbose, config)
        status = result.get("status", "failed")

        if status == "skipped":
            counts["skipped_existing"] += 1
        elif status == "missing_svg":
            counts["missing_svg"] += 1
        elif status == "failed":
            counts["failed"] += 1
        elif status == "unusable":
            counts["processed"] += 1
            counts["unusable"] += 1
        else:
            counts["processed"] += 1
            counts["ok"] += 1

    summary_path = root_dir / "wall_graph_generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(counts, f, indent=2)

    logger.info(
        "Done. processed=%d ok=%d unusable=%d skipped=%d missing_svg=%d failed=%d",
        counts["processed"], counts["ok"], counts["unusable"],
        counts["skipped_existing"], counts["missing_svg"], counts["failed"],
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate orthogonal wall graph labels from CubiCasa model.svg files."
    )
    parser.add_argument("root_dir", type=Path, help="Dataset root with per-sample subdirectories.")
    parser.add_argument("--overwrite", action="store_true", help="Re-generate existing wall graphs.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N samples.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    process_dataset(args.root_dir, overwrite=args.overwrite, verbose=args.verbose, limit=args.limit)


if __name__ == "__main__":
    main()
