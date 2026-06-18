"""Generate SVG output from vectorized primitives.

Final SVG group order (back to front): floor -> wall -> opening -> icon.
`opening` holds both DoorPrimitive and WindowPrimitive elements - doors and
windows are not separate top-level classes. Unresolved/floating openings are
debug-only and rendered in a separate, clearly non-final `debug` group.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .primitives import (
    DoorPrimitive,
    FloorPrimitive,
    IconPrimitive,
    OpeningPrimitive,
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
    doors: list[DoorPrimitive],
    windows: list[WindowPrimitive],
    icons: list[IconPrimitive],
    floor: FloorPrimitive | None = None,
    unresolved_openings: list[OpeningPrimitive] | None = None,
    scale_info: ScaleInfo | None = None,
    svg_config: dict[str, Any] | None = None,
) -> str:
    scale_info = scale_info or ScaleInfo()
    cfg = svg_config or {}

    header = _svg_header(image_width, image_height, scale_info)

    floor_svg = floor.to_svg() if floor is not None and cfg.get("draw_floor", True) else ""
    wall_svg = "\n    ".join(w.to_svg() for w in walls) if cfg.get("draw_wall", True) else ""
    icon_svg = "\n    ".join(i.to_svg() for i in icons) if cfg.get("draw_icon", True) else ""

    opening_parts: list[str] = []
    if cfg.get("draw_opening", True):
        opening_parts.extend(w.to_svg() for w in windows)
        opening_parts.extend(d.to_svg() for d in doors)
    opening_svg = "\n    ".join(opening_parts)

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
        _group("floor", floor_svg)
        + _group("wall", wall_svg)
        + _group("opening", opening_svg)
        + _group("icon", icon_svg)
        + _group("debug", debug_svg)
    )

    return header + body + "</svg>\n"


def save_svg(svg_content: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content, encoding="utf-8")
