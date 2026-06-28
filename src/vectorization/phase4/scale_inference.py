"""Scale inference from segmentation component evidence (spec_v008 §6).

Thin wrapper over the existing vectorization.scale module, adapted for
Phase 4 usage where segmentation components are passed directly.
"""

from __future__ import annotations

from ..components import extract_all_components
from ..graph_types import ComponentRecord
from ..scale import ScaleInfo, resolve_scale_from_components


def infer_scale_from_masks(
    masks: dict[str, any],  # per-class binary masks
    explicit_px_to_mm: float | None = None,
) -> ScaleInfo:
    """Extract components from masks and resolve scale.

    Priority (spec §6): explicit metadata > red door_arc bbox long edge
    clustering > unknown.
    """
    relevant = {k: v for k, v in masks.items() if k in ("door_arc", "door_origin", "wall")}
    components, _ = extract_all_components(relevant)
    return resolve_scale_from_components(
        door_arc_components=components.get("door_arc", []),
        door_origin_components=components.get("door_origin", []),
        wall_components=components.get("wall", []),
        explicit_px_to_mm=explicit_px_to_mm,
    )


def infer_scale_from_components(
    door_arc_components: list[ComponentRecord],
    door_origin_components: list[ComponentRecord],
    wall_components: list[ComponentRecord],
    explicit_px_to_mm: float | None = None,
) -> ScaleInfo:
    """Resolve scale directly from pre-extracted component lists."""
    return resolve_scale_from_components(
        door_arc_components=door_arc_components,
        door_origin_components=door_origin_components,
        wall_components=wall_components,
        explicit_px_to_mm=explicit_px_to_mm,
    )
