"""Export final architectural SVG (spec_v008 §12).

The final SVG contains ONLY final architectural geometry — no debug content.

Visible groups (back to front):
    walls        — filled polygon system
    windows      — colored line segments
    doors        — door_origin + door_leaf + door_arc per door

Door primitives use the Phase 3 DoorOriginPrimitive / DoorLeafPrimitive /
DoorArcPrimitive geometry contract (task32 fix).

task34: door geometry now uses pre-computed DoorGeometry objects (with evidence-
based hinge/swing) instead of calling compute_door_geometry() inline.
If door_geometries is not passed, the old fallback path is used.

task34 Part A: exported window/door endpoints come from adjusted HostedOpenings
(snapped_points already updated by apply_adjusted_intervals_to_hosted_openings).

Debug geometry belongs exclusively in image_debug_overlay.png.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from .door_geometry import DoorGeometry, compute_door_geometry
from .opening_hosting import HostedOpening
from .wall_buffering import WallGeometry, wall_polygon_to_svg_paths
from ..primitives.door import DoorArcPrimitive, DoorLeafPrimitive, DoorOriginPrimitive
from ..primitives.scale import ScaleInfo

WALL_FILL = "#1a1a1a"
WINDOW_STROKE = "#3a78dc"


def _svg_header(scale_info: ScaleInfo) -> str:
    px_to_mm = scale_info.px_to_mm if scale_info.px_to_mm is not None else ""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="512" height="512" viewBox="0 0 512 512" '
        f'data-unit="{scale_info.unit}" '
        f'data-scale-status="{scale_info.scale_status}" '
        f'data-px-to-mm="{px_to_mm}">\n'
    )


def _window_to_svg(win: HostedOpening) -> str:
    """Window line from adjusted snapped_points (Part A)."""
    if len(win.snapped_points) < 2:
        return ""
    x1, y1 = win.snapped_points[0]
    x2, y2 = win.snapped_points[1]
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
        f'x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{WINDOW_STROKE}" stroke-width="4" '
        f'stroke-linecap="square" />'
    )


def _door_to_svg(door: HostedOpening, idx: int, geom: Optional[DoorGeometry] = None) -> str:
    """Generate door_origin + door_leaf + door_arc SVG elements.

    Correct primitive contract (task32):
        door_origin  — purple line along the hosted wall opening (p0 → p1)
        door_leaf    — orange line perpendicular to origin from hinge_point
        door_arc     — red 90° arc from origin_far_point to leaf_end,
                       centered on hinge_point

    Args:
        door: adjusted HostedOpening (snapped_points from adjusted interval).
        idx:  door index for element IDs.
        geom: pre-computed DoorGeometry with evidence-based direction (task34);
              if None, fallback compute_door_geometry() is called.
    """
    if len(door.snapped_points) < 2:
        return ""

    if geom is None:
        geom = compute_door_geometry(door)
    if geom.width_px < 1e-3:
        return ""

    swing_base = geom.swing_side.replace("fallback_", "")  # "left" or "right"
    mid = (
        (geom.hinge_point[0] + geom.origin_far_point[0]) / 2.0,
        (geom.hinge_point[1] + geom.origin_far_point[1]) / 2.0,
    )

    origin_prim = DoorOriginPrimitive(
        primitive_id=f"door_{idx}_origin",
        center=mid,
        width=geom.width_px,
        orientation_angle=geom.orientation_angle_deg,
    )
    leaf_prim = DoorLeafPrimitive(
        primitive_id=f"door_{idx}_leaf",
        hinge_point=geom.hinge_point,
        width=geom.width_px,
        orientation_angle=geom.orientation_angle_deg,
        swing_direction=swing_base,
    )
    arc_prim = DoorArcPrimitive(
        primitive_id=f"door_{idx}_arc",
        hinge_point=geom.hinge_point,
        origin_far_point=geom.origin_far_point,
        width=geom.width_px,
        orientation_angle=geom.orientation_angle_deg,
        swing_direction=swing_base,
    )

    return (
        f'<g id="door_{idx}" data-type="door">'
        f'{origin_prim.to_svg()}'
        f'{leaf_prim.to_svg()}'
        f'{arc_prim.to_svg()}'
        f'</g>'
    )


def build_final_svg(
    scale_info: ScaleInfo,
    wall_geometry: Optional[WallGeometry],
    hosted_doors: list[HostedOpening],
    hosted_windows: list[HostedOpening],
    door_geometries: Optional[list[DoorGeometry]] = None,
) -> str:
    """Build the final SVG string (spec_v008 §12).

    Args:
        scale_info:      scale metadata
        wall_geometry:   buffered wall polygon system
        hosted_doors:    final adjusted door openings (Part A)
        hosted_windows:  final adjusted window openings (Part A)
        door_geometries: pre-computed evidence-based door geometries (Part B);
                         if provided, must be the same length as hosted_doors.
    """
    parts = [_svg_header(scale_info)]

    # --- Wall group ---
    if wall_geometry is not None and wall_geometry.polygon is not None:
        wall_paths = wall_polygon_to_svg_paths(wall_geometry, fill=WALL_FILL)
        if wall_paths:
            parts.append(f'  <g id="walls">\n    {"    ".join(wall_paths)}\n  </g>\n')

    # --- Window group (adjusted endpoints) ---
    if hosted_windows:
        window_svgs = [_window_to_svg(w) for w in hosted_windows]
        window_svgs = [s for s in window_svgs if s]
        if window_svgs:
            parts.append(f'  <g id="windows">\n    {"    ".join(window_svgs)}\n  </g>\n')

    # --- Door group (evidence-based geometry) ---
    if hosted_doors:
        door_svgs = []
        for i, d in enumerate(hosted_doors):
            geom = door_geometries[i] if door_geometries and i < len(door_geometries) else None
            s = _door_to_svg(d, i, geom)
            if s:
                door_svgs.append(s)
        if door_svgs:
            parts.append(f'  <g id="doors">\n    {"    ".join(door_svgs)}\n  </g>\n')

    parts.append("</svg>")
    return "\n".join(parts)


def write_final_svg(svg_content: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content, encoding="utf-8")
