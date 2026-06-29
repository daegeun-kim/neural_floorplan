"""Build the Phase 4 debug overlay image (spec_v008 §13).

Renders onto the preprocessed 512x512 input image:
    - Aligned graph edges (green)
    - Accepted door components (red bbox outline)
    - Accepted window components (blue bbox outline)
    - Raw opening endpoint candidates (yellow circles)
    - Snapped opening endpoints (cyan circles)
    - Host edge highlighted (magenta)
    - Rejected openings with reason text (red X + label)
    - Scale evidence component bbox (orange outline)

A legend panel is appended to the right of the image, labeling every
color/shape with a plain-text description plus scale metadata.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .door_geometry import DoorGeometry
from .opening_detection import DoorCandidate, WindowCandidate
from .opening_hosting import HostedOpening, RejectedOpening
from .wall_interval_editing import TrimmedGraph
from ..graph_types import ComponentRecord
from ..primitives.scale import ScaleInfo

_GRAPH_EDGE_COLOR    = (0, 200, 0)       # green
_GRAPH_NODE_COLOR    = (0, 200, 0)
_DOOR_BBOX_COLOR     = (220, 60, 60)     # red
_WIN_BBOX_COLOR      = (60, 100, 220)    # blue
_RAW_PT_COLOR        = (255, 220, 0)     # yellow
_SNAP_PT_COLOR       = (0, 220, 220)     # cyan
_HOST_EDGE_COLOR     = (220, 0, 220)     # magenta
_REJECT_COLOR        = (220, 40, 40)     # red
_SCALE_COLOR         = (255, 140, 0)     # orange
_ORIG_INTERVAL_COLOR = (255, 100, 0)     # orange-red: original interval (pre-adjustment)
_ADJ_INTERVAL_COLOR  = (0, 255, 160)     # teal-green: adjusted interval
_HINGE_COLOR         = (255, 255, 0)     # yellow: door hinge point
_FINAL_PT_COLOR      = (255, 255, 255)   # white: final primitive endpoints
_RED_EVIDENCE_COLOR  = (180, 0, 80)      # dark-red: red-pixel evidence region

_LEGEND_BG        = (28, 28, 28)      # near-black panel background
_LEGEND_TITLE     = (230, 230, 230)
_LEGEND_TEXT      = (195, 195, 195)
_LEGEND_DIM       = (130, 130, 130)
_LEGEND_W         = 200               # px wide panel

# (label, color, shape)  where shape ∈ "line" | "circle" | "rect" | "cross"
_LEGEND_ITEMS = [
    ("wall graph edge",    _GRAPH_EDGE_COLOR,    "line"),
    ("wall graph node",    _GRAPH_NODE_COLOR,    "circle"),
    ("scale evidence bbox",_SCALE_COLOR,         "rect"),
    ("door candidate bbox",_DOOR_BBOX_COLOR,     "rect"),
    ("window cand. bbox",  _WIN_BBOX_COLOR,      "rect"),
    ("raw endpoint",       _RAW_PT_COLOR,        "circle"),
    ("snapped endpoint",   _SNAP_PT_COLOR,       "circle"),
    ("host wall edge",     _HOST_EDGE_COLOR,     "line"),
    ("orig. interval",     _ORIG_INTERVAL_COLOR, "line"),
    ("adj. interval",      _ADJ_INTERVAL_COLOR,  "line"),
    ("door hinge",         _HINGE_COLOR,         "circle"),
    ("final endpoints",    _FINAL_PT_COLOR,       "circle"),
    ("rejected opening",   _REJECT_COLOR,         "cross"),
]


def _draw_graph(img: np.ndarray, aligned_edges: list[list[float]]) -> None:
    for e in aligned_edges:
        x1, y1, x2, y2 = int(e[0]), int(e[1]), int(e[2]), int(e[3])
        cv2.line(img, (x1, y1), (x2, y2), _GRAPH_EDGE_COLOR, 2)
    seen: set[tuple[int, int]] = set()
    for e in aligned_edges:
        for x, y in [(int(e[0]), int(e[1])), (int(e[2]), int(e[3]))]:
            if (x, y) not in seen:
                cv2.circle(img, (x, y), 4, _GRAPH_NODE_COLOR, -1)
                seen.add((x, y))


def _draw_bbox(img: np.ndarray, bbox: tuple, color: tuple, thickness: int = 2) -> None:
    x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)


def _draw_point(img: np.ndarray, pt: tuple[float, float], color: tuple, r: int = 5) -> None:
    cv2.circle(img, (int(pt[0]), int(pt[1])), r, color, -1)


def _draw_cross(img: np.ndarray, cx: float, cy: float, color: tuple, size: int = 8) -> None:
    x, y = int(cx), int(cy)
    cv2.line(img, (x - size, y - size), (x + size, y + size), color, 2)
    cv2.line(img, (x + size, y - size), (x - size, y + size), color, 2)


def _draw_legend_symbol(
    panel: np.ndarray, x: int, y: int, color: tuple, shape: str
) -> None:
    """Draw the small glyph for one legend row."""
    if shape == "line":
        cv2.line(panel, (x - 10, y), (x + 10, y), color, 2)
    elif shape == "circle":
        cv2.circle(panel, (x, y), 6, color, -1)
    elif shape == "rect":
        cv2.rectangle(panel, (x - 9, y - 6), (x + 9, y + 6), color, 2)
    elif shape == "cross":
        cv2.line(panel, (x - 7, y - 7), (x + 7, y + 7), color, 2)
        cv2.line(panel, (x + 7, y - 7), (x - 7, y + 7), color, 2)


def _build_legend_panel(height: int, scale_info: ScaleInfo) -> np.ndarray:
    """Return a (height × _LEGEND_W × 3) uint8 legend strip."""
    panel = np.full((height, _LEGEND_W, 3), _LEGEND_BG, dtype=np.uint8)

    # --- Title ---
    cv2.putText(panel, "DEBUG LEGEND", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _LEGEND_TITLE, 1, cv2.LINE_AA)
    # Thin separator
    cv2.line(panel, (6, 24), (_LEGEND_W - 6, 24), (70, 70, 70), 1)

    # --- Legend items ---
    row_h = 22
    y0 = 42
    sym_x = 18
    text_x = 34
    for i, (label, color, shape) in enumerate(_LEGEND_ITEMS):
        y = y0 + i * row_h
        _draw_legend_symbol(panel, sym_x, y, color, shape)
        cv2.putText(panel, label, (text_x, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, _LEGEND_TEXT, 1, cv2.LINE_AA)

    # --- Scale metadata ---
    sep_y = y0 + len(_LEGEND_ITEMS) * row_h + 8
    cv2.line(panel, (6, sep_y), (_LEGEND_W - 6, sep_y), (70, 70, 70), 1)
    scale_y = sep_y + 18
    cv2.putText(panel, "SCALE", (8, scale_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _LEGEND_TITLE, 1, cv2.LINE_AA)
    scale_y += 18
    status_color = (80, 200, 80) if scale_info.scale_status == "resolved" else (200, 180, 80)
    cv2.putText(panel, f"status: {scale_info.scale_status}", (8, scale_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, status_color, 1, cv2.LINE_AA)
    scale_y += 16
    if scale_info.px_to_mm is not None:
        px_mm_txt = f"px/mm: {scale_info.px_to_mm:.3f}"
        mm_px_txt = f"mm/px: {1.0 / scale_info.px_to_mm:.2f}"
    else:
        px_mm_txt = "px/mm: unknown"
        mm_px_txt = ""
    cv2.putText(panel, px_mm_txt, (8, scale_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, _LEGEND_DIM, 1, cv2.LINE_AA)
    if mm_px_txt:
        scale_y += 16
        cv2.putText(panel, mm_px_txt, (8, scale_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, _LEGEND_DIM, 1, cv2.LINE_AA)
    scale_y += 16
    src = scale_info.scale_source or "-"
    cv2.putText(panel, f"source: {src}", (8, scale_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, _LEGEND_DIM, 1, cv2.LINE_AA)

    return panel


def _draw_interval_on_edge(
    img: np.ndarray,
    edge: list[float],
    t_start: float,
    t_end: float,
    color: tuple,
    thickness: int = 3,
    offset_px: int = 0,
) -> None:
    """Draw a segment between t_start and t_end on an edge, with optional perpendicular offset."""
    x1, y1, x2, y2 = edge[0], edge[1], edge[2], edge[3]
    ax = x1 + t_start * (x2 - x1)
    ay = y1 + t_start * (y2 - y1)
    bx = x1 + t_end * (x2 - x1)
    by_ = y1 + t_end * (y2 - y1)
    if offset_px != 0:
        edge_len = math.hypot(x2 - x1, y2 - y1)
        if edge_len > 1e-6:
            nx = -(y2 - y1) / edge_len * offset_px
            ny = (x2 - x1) / edge_len * offset_px
            ax += nx; ay += ny
            bx += nx; by_ += ny
    cv2.line(img, (int(ax), int(ay)), (int(bx), int(by_)), color, thickness)


def build_debug_overlay(
    input_pil: Image.Image,
    aligned_edges: list[list[float]],
    door_candidates: list[DoorCandidate],
    window_candidates: list[WindowCandidate],
    hosted_doors: list[HostedOpening],
    hosted_windows: list[HostedOpening],
    rejected_openings: list[RejectedOpening],
    door_arc_components: list[ComponentRecord],
    scale_info: ScaleInfo,
    trimmed_graph: Optional[TrimmedGraph] = None,
    door_geometries: Optional[list[DoorGeometry]] = None,
) -> Image.Image:
    """Build the debug overlay image with legend (spec_v008 §13 + task33).

    Shows original and adjusted opening intervals when interval adjustment occurred.
    Returns a 512×(512 + _LEGEND_W) PIL image.
    """
    img = np.array(input_pil.convert("RGB"), dtype=np.uint8).copy()

    # Aligned graph
    _draw_graph(img, aligned_edges)

    # Scale evidence: door_arc component bboxes used for scale
    for comp in door_arc_components:
        _draw_bbox(img, comp.bbox, _SCALE_COLOR, thickness=1)

    # Accepted door candidates: show bbox and raw points
    for dc in door_candidates:
        _draw_bbox(img, dc.bbox, _DOOR_BBOX_COLOR)
        for pt in dc.raw_points:
            _draw_point(img, pt, _RAW_PT_COLOR, r=4)

    # Accepted window candidates: show bbox and raw points
    for wc in window_candidates:
        _draw_bbox(img, wc.bbox, _WIN_BBOX_COLOR)
        for pt in wc.raw_points:
            _draw_point(img, pt, _RAW_PT_COLOR, r=4)

    # Hosted doors: snapped points + host edge + evidence-based hinge/swing
    for i, hosted in enumerate(hosted_doors):
        edge = hosted.host_edge_raw
        cv2.line(img, (int(edge[0]), int(edge[1])), (int(edge[2]), int(edge[3])),
                 _HOST_EDGE_COLOR, 3)
        # Final primitive endpoints (white)
        for pt in hosted.snapped_points:
            _draw_point(img, pt, _FINAL_PT_COLOR, r=4)
            _draw_point(img, pt, _SNAP_PT_COLOR, r=6)  # cyan ring
        # Evidence-based hinge point (yellow) + swing / side / fallback / type label
        if door_geometries and i < len(door_geometries):
            dg = door_geometries[i]
            _draw_point(img, dg.hinge_point, _HINGE_COLOR, r=5)
            lx, ly = int(dg.hinge_point[0]) + 6, int(dg.hinge_point[1]) - 6
            swing_str = dg.swing_side.replace("fallback_", "fb_")
            side_str = dg.red_side_selected[:3] if dg.red_side_selected else "?"
            fb_str = "fb" if dg.fallback_used else side_str
            type_tag = "DS" if dg.door_type == "double_swing_shared_origin" else ""
            label = f"{type_tag}{swing_str}|{fb_str}" if type_tag else f"{swing_str}|{fb_str}"
            cv2.putText(img, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, _HINGE_COLOR, 1, cv2.LINE_AA)

    # Hosted windows: final snapped points + host edge
    for hosted in hosted_windows:
        edge = hosted.host_edge_raw
        cv2.line(img, (int(edge[0]), int(edge[1])), (int(edge[2]), int(edge[3])),
                 _HOST_EDGE_COLOR, 3)
        for pt in hosted.snapped_points:
            _draw_point(img, pt, _FINAL_PT_COLOR, r=4)
            _draw_point(img, pt, _SNAP_PT_COLOR, r=6)

    # Interval adjustment overlay from trimmed_graph opening_gaps (task33)
    if trimmed_graph is not None:
        for gap in trimmed_graph.opening_gaps:
            if not gap.get("was_adjusted", False):
                continue
            ei = gap.get("host_edge_idx", -1)
            if ei < 0 or ei >= len(aligned_edges):
                continue
            edge = aligned_edges[ei]
            orig = gap.get("original_interval", [])
            adj = gap.get("adjusted_interval", [])
            if len(orig) == 2:
                # Draw original interval slightly offset above edge
                _draw_interval_on_edge(img, edge, orig[0], orig[1],
                                       _ORIG_INTERVAL_COLOR, thickness=2, offset_px=-6)
            if len(adj) == 2:
                # Draw adjusted interval slightly offset below edge
                _draw_interval_on_edge(img, edge, adj[0], adj[1],
                                       _ADJ_INTERVAL_COLOR, thickness=2, offset_px=6)
            # Label the adjustment
            x1, y1, x2, y2 = edge[0], edge[1], edge[2], edge[3]
            if len(adj) == 2:
                mid_t = (adj[0] + adj[1]) / 2.0
            elif len(orig) == 2:
                mid_t = (orig[0] + orig[1]) / 2.0
            else:
                mid_t = 0.5
            lx = int(x1 + mid_t * (x2 - x1)) + 4
            ly = int(y1 + mid_t * (y2 - y1)) - 10
            opening_type = gap.get("opening_type", "?")
            cv2.putText(img, f"adj:{opening_type}", (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, _ADJ_INTERVAL_COLOR, 1, cv2.LINE_AA)

        # Show last-resort rejected from conflict resolution
        for rej in trimmed_graph.last_resort_rejected:
            ei = rej.get("host_edge_idx", -1)
            if 0 <= ei < len(aligned_edges):
                edge = aligned_edges[ei]
                t_mid = (rej.get("original_t_start", 0.0) + rej.get("original_t_end", 1.0)) / 2.0
                x1, y1, x2, y2 = edge[0], edge[1], edge[2], edge[3]
                cx = x1 + t_mid * (x2 - x1)
                cy = y1 + t_mid * (y2 - y1)
                _draw_cross(img, cx, cy, _REJECT_COLOR, size=10)
                cv2.putText(img, f"no-fit:{rej.get('opening_type', '?')}",
                            (int(cx) + 8, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, _REJECT_COLOR, 1, cv2.LINE_AA)

    # Rejected openings from hosting: draw X at centroid of raw points
    for rej in rejected_openings:
        pts = rej.raw_points
        if pts:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            _draw_cross(img, cx, cy, _REJECT_COLOR)
            label = rej.rejection_reason[:28]
            cv2.putText(img, label, (int(cx) + 10, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, _REJECT_COLOR, 1, cv2.LINE_AA)

    # Append legend panel
    legend = _build_legend_panel(img.shape[0], scale_info)
    combined = np.concatenate([img, legend], axis=1)

    return Image.fromarray(combined)


def write_debug_overlay(overlay: Image.Image, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(str(output_path))
