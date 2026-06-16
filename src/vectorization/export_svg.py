"""Generate SVG output from vectorized primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .primitives import (
    DoorPrimitive,
    OpeningPrimitive,
    RoomPrimitive,
    ScaleInfo,
    WallPrimitive,
    WindowPrimitive,
)


def _svg_header(width: int, height: int, scale_info: ScaleInfo) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'data-unit="{scale_info.unit}" '
        f'data-scale-status="{scale_info.scale_status}">\n'
    )


def _group(group_id: str, content: str) -> str:
    if not content.strip():
        return ""
    return f'  <g id="{group_id}">\n    {content}\n  </g>\n'


def build_svg(
    image_width: int,
    image_height: int,
    walls: list[WallPrimitive],
    openings: list[OpeningPrimitive],
    doors: list[DoorPrimitive],
    windows: list[WindowPrimitive],
    rooms: list[RoomPrimitive],
    unresolved_openings: list[OpeningPrimitive] | None = None,
    scale_info: ScaleInfo | None = None,
    svg_config: dict[str, Any] | None = None,
) -> str:
    scale_info = scale_info or ScaleInfo()
    cfg = svg_config or {}

    header = _svg_header(image_width, image_height, scale_info)

    room_svg = "\n    ".join(r.to_svg() for r in rooms) if cfg.get("draw_rooms", True) else ""
    wall_svg = "\n    ".join(w.to_svg() for w in walls) if cfg.get("draw_walls", True) else ""
    opening_svg = "\n    ".join(o.to_svg() for o in openings) if cfg.get("draw_openings", True) else ""
    door_svg = "\n    ".join(d.to_svg() for d in doors) if cfg.get("draw_doors", True) else ""
    window_svg = "\n    ".join(w.to_svg() for w in windows) if cfg.get("draw_windows", True) else ""

    debug_parts = []
    if cfg.get("include_debug_layer", True) and unresolved_openings:
        for op in unresolved_openings:
            s, e = op.start, op.end
            debug_parts.append(
                f'<line x1="{s[0]:.1f}" y1="{s[1]:.1f}" '
                f'x2="{e[0]:.1f}" y2="{e[1]:.1f}" '
                f'stroke="#ff8800" stroke-width="2" stroke-dasharray="6 3" '
                f'data-id="{op.primitive_id}" data-type="unresolved" />'
            )
    debug_svg = "\n    ".join(debug_parts)

    body = (
        _group("rooms", room_svg)
        + _group("walls", wall_svg)
        + _group("openings", opening_svg)
        + _group("doors", door_svg)
        + _group("windows", window_svg)
        + _group("debug", debug_svg)
    )

    return header + body + "</svg>\n"


def save_svg(svg_content: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content, encoding="utf-8")
