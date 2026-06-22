"""Generate SVG output from vectorized primitives (active 7-class run3 scheme).

Final SVG group order (back to front): floor -> wall -> window -> door. Per
task08, the final SVG contains only these four component types - no debug
group, no unresolved/unidentified markers. `wall` is rendered as one (or a
few, if genuinely disconnected) black filled polygon built by buffering and
unioning every outer+inner wall centerline (wall_geometry.segments_to_polygon)
rather than per-segment stroked lines. `door` holds one `<g>` per door
containing its origin/leaf/arc primitives.

Debug-only visualization of unresolved/unhosted evidence lives exclusively in
run_mask_to_vector's debug_overlay.png raster and metrics.json - never in
this SVG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .primitives import FloorPrimitive, ScaleInfo, WallPrimitive, WindowPrimitive
from .primitives.door import DoorArcPrimitive, DoorLeafPrimitive, DoorOriginPrimitive
from .wall_geometry import polygon_to_svg_path, segments_to_polygon

WALL_COLOR = "#000000"


def _svg_header(width: int, height: int, scale_info: ScaleInfo) -> str:
    px_to_mm = scale_info.px_to_mm if scale_info.px_to_mm is not None else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'data-unit="{scale_info.unit}" '
        f'data-scale-status="{scale_info.scale_status}" '
        f'data-px-to-mm="{px_to_mm}" '
        f'data-scale-source="{scale_info.scale_source}">\n'
    )


def _group(group_id: str, content: str) -> str:
    if not content.strip():
        return ""
    return f'  <g id="{group_id}">\n    {content}\n  </g>\n'


def build_svg(
    image_width: int,
    image_height: int,
    walls: list[WallPrimitive],
    windows: list[WindowPrimitive],
    door_origins: list[DoorOriginPrimitive],
    door_leaves: list[DoorLeafPrimitive],
    door_arcs: list[DoorArcPrimitive],
    floor: FloorPrimitive | None = None,
    scale_info: ScaleInfo | None = None,
    svg_config: dict[str, Any] | None = None,
) -> str:
    scale_info = scale_info or ScaleInfo()
    cfg = svg_config or {}

    header = _svg_header(image_width, image_height, scale_info)

    floor_svg = floor.to_svg() if floor is not None and cfg.get("draw_floor", True) else ""

    wall_svg = ""
    if cfg.get("draw_wall", True) and walls:
        half_width = walls[0].thickness / 2.0
        wall_geom = segments_to_polygon([(w.start, w.end) for w in walls], half_width)
        wall_svg = polygon_to_svg_path(wall_geom, WALL_COLOR, extra_attrs='id="wall_polygon"')

    window_svg = "\n    ".join(w.to_svg() for w in windows) if cfg.get("draw_window", True) else ""

    door_svg = ""
    if cfg.get("draw_door", True):
        doors_by_id = {}
        for origin in door_origins:
            idx = origin.primitive_id.rsplit("_", 1)[-1]
            doors_by_id.setdefault(idx, {})["origin"] = origin
        for leaf in door_leaves:
            idx = leaf.primitive_id.rsplit("_", 1)[-1]
            doors_by_id.setdefault(idx, {})["leaf"] = leaf
        for arc in door_arcs:
            idx = arc.primitive_id.rsplit("_", 1)[-1]
            doors_by_id.setdefault(idx, {})["arc"] = arc

        door_groups = []
        for idx, parts in sorted(doors_by_id.items()):
            inner = "".join(
                parts[key].to_svg() for key in ("origin", "leaf", "arc") if key in parts
            )
            door_groups.append(f'<g id="door_{idx}" data-type="door">{inner}</g>')
        door_svg = "\n    ".join(door_groups)

    body = (
        _group("floor", floor_svg)
        + _group("wall", wall_svg)
        + _group("window", window_svg)
        + _group("door", door_svg)
    )

    return header + body + "</svg>\n"


def save_svg(svg_content: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content, encoding="utf-8")
