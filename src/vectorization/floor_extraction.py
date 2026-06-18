"""Build the floor footprint as a direct translation of the outer wall loop."""

from __future__ import annotations

from .primitives import ScaleInfo
from .primitives.floor import FloorPrimitive


def extract_floor(
    outer_polygon: list[tuple[float, float]],
    scale_info: ScaleInfo | None = None,
) -> FloorPrimitive | None:
    """Wrap the outer wall loop polygon as a filled FloorPrimitive.

    The floor boundary is the outer wall loop itself - no separate mask
    processing. Returns None if the outer wall loop could not be resolved.
    """
    if not outer_polygon or len(outer_polygon) < 3:
        return None

    return FloorPrimitive(
        primitive_id="floor_0",
        polygon=list(outer_polygon),
        confidence=1.0,
        scale_info=scale_info or ScaleInfo(),
    )
