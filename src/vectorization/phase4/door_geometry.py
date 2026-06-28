"""Compute door primitive geometry from a HostedOpening (spec_v008 task32/task34).

A door has three semantic elements:
    door_origin  — purple line along the wall opening (the threshold)
    door_leaf    — orange line perpendicular to origin from hinge_point
    door_arc     — red 90-degree arc from origin_far_point to leaf_end,
                   centered on hinge_point

task32: fixed primitive order (origin → leaf → arc).
task34: hinge and swing now inferred from local red/orange/purple raster evidence
        before falling back to the deterministic defaults.

Scoring (spec §7 Part B):
    Four hypotheses: hinge=p0|p1  ×  swing=left|right
    Each is scored by counting red door_arc mask pixels along the arc trajectory.
    Secondary score: orange door_leaf mask pixels along the leaf line.
    Best-scoring hypothesis wins; fallback is used only when total evidence is zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .opening_hosting import HostedOpening


@dataclass
class DoorGeometry:
    """Resolved geometry for one door's three SVG primitives."""
    hinge_point: tuple[float, float]
    origin_far_point: tuple[float, float]
    leaf_end: tuple[float, float]
    swing_side: str          # "left" | "right" | "fallback_left" | "fallback_right"
    width_px: float
    orientation_angle_deg: float  # angle from hinge toward origin_far_point, degrees
    hinge_source: str        # "red_orange_purple_evidence" | "fallback_pt0"
    swing_source: str        # "red_door_arc_side" | "fallback"


def _perp_end(
    hinge: tuple[float, float],
    width: float,
    orientation_angle_deg: float,
    swing_side: str,
) -> tuple[float, float]:
    """Return leaf endpoint: `width` from hinge, perpendicular to origin direction."""
    base = swing_side.replace("fallback_", "")
    sign = 1.0 if base == "left" else -1.0
    angle_rad = math.radians(orientation_angle_deg)
    perp = angle_rad + sign * math.pi / 2.0
    return (
        hinge[0] + width * math.cos(perp),
        hinge[1] + width * math.sin(perp),
    )


def _score_arc_pixels(
    hinge: tuple[float, float],
    far: tuple[float, float],
    swing: str,
    mask: np.ndarray,
    n_samples: int = 16,
) -> float:
    """Count mask pixels along the 90-degree arc sweep for one hypothesis.

    The arc starts at `far` (= origin_far_point) and sweeps 90° in the
    direction determined by `swing`, centered at `hinge`.
    """
    r = math.hypot(far[0] - hinge[0], far[1] - hinge[1])
    if r < 1e-3:
        return 0.0
    start_ang = math.atan2(far[1] - hinge[1], far[0] - hinge[0])
    sign = 1.0 if swing == "left" else -1.0
    h, w = mask.shape[:2]
    hit = 0
    for i in range(n_samples + 1):
        ang = start_ang + sign * (math.pi / 2.0) * (i / n_samples)
        sx = int(round(hinge[0] + r * math.cos(ang)))
        sy = int(round(hinge[1] + r * math.sin(ang)))
        if 0 <= sx < w and 0 <= sy < h and mask[sy, sx] > 0:
            hit += 1
    return hit / (n_samples + 1)


def _score_line_pixels(
    pt_a: tuple[float, float],
    pt_b: tuple[float, float],
    mask: np.ndarray,
    n_samples: int = 16,
) -> float:
    """Count mask pixels along a straight line segment."""
    h, w = mask.shape[:2]
    hit = 0
    for i in range(n_samples + 1):
        t = i / n_samples
        sx = int(round(pt_a[0] + t * (pt_b[0] - pt_a[0])))
        sy = int(round(pt_a[1] + t * (pt_b[1] - pt_a[1])))
        if 0 <= sx < w and 0 <= sy < h and mask[sy, sx] > 0:
            hit += 1
    return hit / (n_samples + 1)


def infer_door_direction_from_evidence(
    p0: tuple[float, float],
    p1: tuple[float, float],
    door_arc_mask: Optional[np.ndarray],
    door_leaf_mask: Optional[np.ndarray] = None,
    n_arc_samples: int = 16,
    min_score_threshold: float = 0.05,
) -> tuple[str, str, str, str]:
    """Score 4 hinge/swing hypotheses against local raster evidence.

    Tests hinge ∈ {p0, p1} × swing ∈ {left, right} and picks the
    hypothesis whose arc trajectory overlaps the most red door_arc pixels.

    Args:
        p0, p1:             adjusted door origin endpoints
        door_arc_mask:      binary (uint8) mask of red door_arc class pixels
        door_leaf_mask:     binary mask of orange door_leaf class pixels (secondary)
        n_arc_samples:      sample count along the arc for scoring
        min_score_threshold: minimum arc pixel ratio to accept evidence result

    Returns:
        (hinge_pt, swing_side, hinge_source, swing_source)
        where hinge_pt ∈ {"p0","p1"}, swing_side ∈ {"left","right","fallback_left","fallback_right"}
    """
    if door_arc_mask is None:
        return "p0", "fallback_left", "fallback_pt0", "fallback"

    width = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if width < 1e-3:
        return "p0", "fallback_left", "fallback_pt0", "fallback"

    candidates: list[tuple[float, str, str]] = []
    for hinge_name, hinge, far in [("p0", p0, p1), ("p1", p1, p0)]:
        for swing in ["left", "right"]:
            # Primary: arc pixel overlap with red door_arc mask
            score = _score_arc_pixels(hinge, far, swing, door_arc_mask, n_arc_samples)

            # Secondary: leaf line overlap with orange door_leaf mask
            if door_leaf_mask is not None:
                orientation_deg = math.degrees(math.atan2(far[1] - hinge[1], far[0] - hinge[0]))
                leaf_end = _perp_end(hinge, width, orientation_deg, swing)
                leaf_score = _score_line_pixels(hinge, leaf_end, door_leaf_mask, n_arc_samples)
                score += 0.3 * leaf_score

            candidates.append((score, hinge_name, swing))

    best_score, best_hinge, best_swing = max(candidates, key=lambda x: x[0])

    if best_score < min_score_threshold:
        return "p0", "fallback_left", "fallback_pt0", "fallback"

    return best_hinge, best_swing, "red_orange_purple_evidence", "red_door_arc_side"


def compute_door_geometry(
    hosted_door: HostedOpening,
    swing_side: Optional[str] = None,
    door_arc_mask: Optional[np.ndarray] = None,
    door_leaf_mask: Optional[np.ndarray] = None,
) -> DoorGeometry:
    """Derive hinge, far-point, leaf-end, and arc geometry from a hosted door.

    Args:
        hosted_door:    door snapped to a wall edge (snapped_points must be the
                        adjusted final endpoints from apply_adjusted_intervals_...).
        swing_side:     override swing direction ("left"/"right") — bypasses evidence.
        door_arc_mask:  binary mask of red door_arc pixels for evidence-based scoring.
        door_leaf_mask: binary mask of orange door_leaf pixels (secondary evidence).

    Returns:
        DoorGeometry with all three primitive endpoints + provenance fields.
    """
    p0 = tuple(hosted_door.snapped_points[0])
    p1 = tuple(hosted_door.snapped_points[1])

    width = math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    if swing_side is not None:
        # Explicit override — keep hinge at p0 (fallback) but record evidence source
        hinge = p0
        far = p1
        hinge_source = "fallback_pt0"
        swing_source = "evidence"
    elif door_arc_mask is not None:
        # Evidence-based inference
        hinge_pt, swing_side, hinge_source, swing_source = infer_door_direction_from_evidence(
            p0, p1, door_arc_mask, door_leaf_mask
        )
        hinge = p0 if hinge_pt == "p0" else p1
        far   = p1 if hinge_pt == "p0" else p0
    else:
        hinge = p0
        far = p1
        hinge_source = "fallback_pt0"
        swing_side = "fallback_left"
        swing_source = "fallback"

    orientation_deg = math.degrees(math.atan2(far[1] - hinge[1], far[0] - hinge[0]))
    leaf = _perp_end(hinge, width, orientation_deg, swing_side)

    return DoorGeometry(
        hinge_point=hinge,
        origin_far_point=far,
        leaf_end=leaf,
        swing_side=swing_side,
        width_px=width,
        orientation_angle_deg=orientation_deg,
        hinge_source=hinge_source,
        swing_source=swing_source,
    )


def door_geometry_to_dict(geom: DoorGeometry, width_mm: Optional[float] = None) -> dict:
    """Serialize DoorGeometry to the final_vector.json door_geometry sub-dict."""
    return {
        "hinge_point": [round(geom.hinge_point[0], 2), round(geom.hinge_point[1], 2)],
        "origin_far_point": [round(geom.origin_far_point[0], 2), round(geom.origin_far_point[1], 2)],
        "leaf_end": [round(geom.leaf_end[0], 2), round(geom.leaf_end[1], 2)],
        "swing_side": geom.swing_side,
        "width_px": round(geom.width_px, 2),
        "width_mm": round(width_mm, 1) if width_mm is not None else None,
        "orientation_angle_deg": round(geom.orientation_angle_deg, 2),
        "hinge_source": geom.hinge_source,
        "swing_source": geom.swing_source,
        "primitive_contract": "door_origin_leaf_arc",
    }
