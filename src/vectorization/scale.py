"""Top-level scale module for the v008 point-graph pipeline (spec_v008 SS5/SS6).

The actual estimation math (clustering red door_arc bbox long edges as the
primary metric source, with door_origin width and wall thickness as
secondary cross-check/debug evidence only - task12 SS1) already lives in
``primitives.scale`` and is shared with the v007 primitive contract -
this module re-exports it under the top-level name spec_v008 SS6 expects, and
adds the v008-specific entry point that pulls evidence lengths directly out
of ``ComponentRecord``s produced by ``components.py``.
"""

from __future__ import annotations

from .graph_types import ComponentRecord
from .primitives.scale import (
    DOOR_MODULES_MM,
    WALL_MODULES_MM,
    ScaleInfo,
    resolve_scale,
    resolve_scale_from_door_arc_bboxes,
    resolve_scale_with_door_arc_priority,
    snap_to_module_mm,
)

__all__ = [
    "DOOR_MODULES_MM",
    "WALL_MODULES_MM",
    "ScaleInfo",
    "resolve_scale",
    "resolve_scale_from_door_arc_bboxes",
    "resolve_scale_with_door_arc_priority",
    "snap_to_module_mm",
    "resolve_scale_from_components",
]


def resolve_scale_from_components(
    door_arc_components: list[ComponentRecord],
    door_origin_components: list[ComponentRecord],
    wall_components: list[ComponentRecord],
    explicit_px_to_mm: float | None = None,
    door_modules_mm: tuple[float, ...] = DOOR_MODULES_MM,
    wall_modules_mm: tuple[float, ...] = WALL_MODULES_MM,
    min_confidence: float = 0.70,
) -> ScaleInfo:
    """Resolve px-to-mm scale from accepted door_arc/door_origin/wall component evidence.

    Red ``door_arc`` bounding-box long edge (``max(x1-x0, y1-y0)``) is the
    primary metric source; door-origin length (``rect_size[0]``) and wall
    thickness (``rect_size[1]``) - both from ``cv2.minAreaRect`` in
    ``components.py`` - are secondary cross-check/debug evidence only.
    Delegates the priority order and clustering to
    ``primitives.scale.resolve_scale_with_door_arc_priority``.
    """
    door_arc_long_edges_px = [
        float(max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1])) for c in door_arc_components
    ]
    door_lengths_px = [c.rect_size[0] for c in door_origin_components if c.rect_size]
    wall_thickness_px = [c.rect_size[1] for c in wall_components if c.rect_size]
    return resolve_scale_with_door_arc_priority(
        door_arc_bbox_long_edges_px=door_arc_long_edges_px,
        door_origin_lengths_px=door_lengths_px,
        wall_thickness_px=wall_thickness_px,
        explicit_px_to_mm=explicit_px_to_mm,
        door_modules_mm=door_modules_mm,
        wall_modules_mm=wall_modules_mm,
        min_confidence=min_confidence,
    )
