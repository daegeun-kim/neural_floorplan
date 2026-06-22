"""Scale metadata and metric scale resolution for vectorized primitives.

Per spec_v007 SS5 / spec_v008 SS5: a raster prediction alone does not guarantee
absolute building scale. ``resolve_scale`` estimates millimeters-per-pixel from
clustered door_origin widths (primary evidence) cross-checked against clustered
wall thickness (secondary evidence), falling back to pixel units when the
evidence is insufficient or the two sources disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

DOOR_MODULES_MM: tuple[float, ...] = (600.0, 800.0, 900.0)
WALL_MODULES_MM: tuple[float, ...] = (100.0, 200.0)
WINDOW_MODULES_MM: tuple[float, ...] = (600.0, 900.0, 1200.0, 1500.0)

# How close a measurement must be to a module (relative error) to "vote" for it.
_MODULE_TOLERANCE = 0.15
# How close two independent px_to_mm estimates must be to agree (relative error).
_CROSS_CHECK_TOLERANCE = 0.25


@dataclass
class ScaleInfo:
    """Scale metadata carried by every primitive.

    ``unit`` is "px" or "mm". ``px_to_mm`` is the millimeters represented by
    one pixel (``None`` when unit is "px" / scale is unknown). ``scale_status``
    is "resolved" (explicit metadata confirmed it), "estimated" (clustering
    produced a plausible value), or "unknown" (fell back to pixel units).
    """

    unit: str = "px"
    px_to_mm: float | None = None
    scale_status: str = "unknown"
    scale_source: str = "none"
    confidence: float = 0.0


def _nearest_module(value_mm: float, modules_mm: tuple[float, ...]) -> tuple[float, float]:
    """Return (nearest_module, relative_error) for a measured mm value."""
    best_module = modules_mm[0]
    best_err = abs(value_mm - best_module) / best_module
    for module in modules_mm[1:]:
        err = abs(value_mm - module) / module
        if err < best_err:
            best_err = err
            best_module = module
    return best_module, best_err


def _best_px_to_mm_from_lengths(
    lengths_px: list[float], modules_mm: tuple[float, ...]
) -> tuple[float | None, float]:
    """Vote for the px_to_mm candidate (from each length/module pair) that the
    most *other* measurements agree with. Returns (px_to_mm, confidence in [0,1]).
    """
    candidates = [length for length in lengths_px if length > 1e-3]
    if not candidates:
        return None, 0.0

    best_candidate_px_to_mm = None
    best_votes = 0
    for length_px in candidates:
        for module_mm in modules_mm:
            candidate_px_to_mm = module_mm / length_px
            votes = 0
            for other_px in candidates:
                other_mm = other_px * candidate_px_to_mm
                _, err = _nearest_module(other_mm, modules_mm)
                if err <= _MODULE_TOLERANCE:
                    votes += 1
            if votes > best_votes:
                best_votes = votes
                best_candidate_px_to_mm = candidate_px_to_mm

    if best_candidate_px_to_mm is None:
        return None, 0.0
    confidence = min(1.0, best_votes / len(candidates))
    return best_candidate_px_to_mm, confidence


def resolve_scale(
    door_origin_lengths_px: list[float] | None = None,
    wall_thickness_px: list[float] | None = None,
    explicit_px_to_mm: float | None = None,
    door_modules_mm: tuple[float, ...] = DOOR_MODULES_MM,
    wall_modules_mm: tuple[float, ...] = WALL_MODULES_MM,
    min_confidence: float = 0.70,
) -> ScaleInfo:
    """Resolve a px-to-mm scale factor using the priority order from the spec.

    1. ``explicit_px_to_mm`` (dataset/SVG metadata) - always accepted as resolved.
    2. Clustered door_origin widths against common door modules.
    3. Cross-checked / cross-validated against clustered wall thickness.
    4. Fallback to pixel units with scale_status="unknown".
    """
    if explicit_px_to_mm is not None and explicit_px_to_mm > 0:
        return ScaleInfo(
            unit="mm",
            px_to_mm=explicit_px_to_mm,
            scale_status="resolved",
            scale_source="explicit_metadata",
            confidence=1.0,
        )

    door_px_to_mm, door_conf = _best_px_to_mm_from_lengths(
        door_origin_lengths_px or [], door_modules_mm
    )
    wall_px_to_mm, wall_conf = _best_px_to_mm_from_lengths(
        wall_thickness_px or [], wall_modules_mm
    )

    if door_px_to_mm is not None and wall_px_to_mm is not None:
        rel_diff = abs(door_px_to_mm - wall_px_to_mm) / max(door_px_to_mm, wall_px_to_mm)
        if rel_diff <= _CROSS_CHECK_TOLERANCE:
            # Doors and walls agree - blend, weighted toward door evidence
            # (door evidence is the spec's first practical fallback).
            px_to_mm = 0.7 * door_px_to_mm + 0.3 * wall_px_to_mm
            confidence = min(1.0, max(door_conf, wall_conf) + 0.15)
            if confidence >= min_confidence:
                return ScaleInfo(
                    unit="mm",
                    px_to_mm=px_to_mm,
                    scale_status="estimated",
                    scale_source="door_origin_and_wall_thickness_clustering",
                    confidence=confidence,
                )
        # Doors and walls disagree strongly - do not guess, report the conflict.
        return ScaleInfo(
            unit="px",
            px_to_mm=None,
            scale_status="unknown",
            scale_source="scale_conflict_door_vs_wall",
            confidence=0.0,
        )

    if door_px_to_mm is not None and door_conf >= min_confidence:
        return ScaleInfo(
            unit="mm",
            px_to_mm=door_px_to_mm,
            scale_status="estimated",
            scale_source="door_origin_width_clustering",
            confidence=door_conf,
        )

    if wall_px_to_mm is not None and wall_conf >= min_confidence:
        return ScaleInfo(
            unit="mm",
            px_to_mm=wall_px_to_mm,
            scale_status="estimated",
            scale_source="wall_thickness_clustering",
            confidence=wall_conf,
        )

    return ScaleInfo(
        unit="px",
        px_to_mm=None,
        scale_status="unknown",
        scale_source="insufficient_evidence",
        confidence=0.0,
    )


def snap_to_module_mm(
    value_px: float,
    scale_info: ScaleInfo,
    modules_mm: tuple[float, ...],
    min_confidence_for_metric: float = 0.70,
) -> tuple[float | None, float]:
    """Convert a pixel measurement to mm and snap to the nearest module.

    Returns (snapped_mm_or_None, value_px_unchanged). Returns None for the mm
    value when scale is not trustworthy enough (per ``min_confidence_for_metric``).
    """
    if (
        scale_info.px_to_mm is None
        or scale_info.scale_status not in ("resolved", "estimated")
        or scale_info.confidence < min_confidence_for_metric
    ):
        return None, value_px
    value_mm = value_px * scale_info.px_to_mm
    module, err = _nearest_module(value_mm, modules_mm)
    if err <= _MODULE_TOLERANCE:
        return module, value_px
    return value_mm, value_px
