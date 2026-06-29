"""Compute door primitive geometry from a HostedOpening (spec_v008 task32/task34/task35).

A door has three semantic elements:
    door_origin  — purple line along the wall opening (the threshold)
    door_leaf    — orange line perpendicular to origin from hinge_point
    door_arc     — red 90-degree arc from origin_far_point to leaf_end,
                   centered on hinge_point

task32: fixed primitive order (origin → leaf → arc).
task34: hinge and swing now inferred from local red/orange/purple raster evidence.
task35: evidence hierarchy updated:
    Primary   swing  — red-pixel cross-product side count (absolute side of p0→p1 line)
    Primary   hinge  — orange door_leaf corridor + near-endpoint proximity
    Secondary hinge  — arc-sampling with fixed swing (2 hypotheses, not 4)
    Fallback         — p0 hinge / fallback_left swing when evidence is absent/ambiguous

Absolute side → per-hinge swing mapping (in image-y↓ coords):
    positive cross + hinge=p0 → swing="left"   (arc below line when going east)
    positive cross + hinge=p1 → swing="right"
    negative cross + hinge=p0 → swing="right"
    negative cross + hinge=p1 → swing="left"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace as _dc_replace
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
    # task35 evidence debug fields (default-valued; existing construction is unaffected)
    red_side_positive_count: int = 0
    red_side_negative_count: int = 0
    red_side_selected: str = ""       # "positive" | "negative" | "fallback"
    orange_hinge_p0_score: float = 0.0
    orange_hinge_p1_score: float = 0.0
    hinge_selected: str = ""          # "p0" | "p1"
    fallback_used: bool = False
    # task36 double-swing fields (default-valued)
    door_type: str = "single_swing"               # "single_swing" | "double_swing_shared_origin"
    secondary_leaf_end: Optional[tuple] = None    # for double_swing: leaf endpoint on the other side
    secondary_swing_side: str = ""
    classification_reason: str = ""
    double_swing_ratio: Optional[float] = None
    source_door_component_ids: list = field(default_factory=list)


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
    """Count mask pixels along the 90-degree arc sweep for one hypothesis."""
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


def _score_side_by_red_pixels(
    p0: tuple[float, float],
    p1: tuple[float, float],
    red_mask: np.ndarray,
    min_line_dist: float = 2.0,
) -> tuple[int, int, str]:
    """Count red pixels on each signed side of the directed segment p0→p1.

    Cross product: dx*(py - p0y) - dy*(px - p0x)
        > 0  →  "positive" side
        < 0  →  "negative" side

    Returns (positive_count, negative_count, selected_side).
    Pixels within min_line_dist of the line are excluded (threshold noise).
    """
    ys, xs = np.where(red_mask > 0)
    if len(xs) == 0:
        return 0, 0, "fallback"

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    line_len = math.hypot(dx, dy)

    cross = dx * (ys.astype(float) - p0[1]) - dy * (xs.astype(float) - p0[0])

    if line_len > 1e-3 and min_line_dist > 0:
        dist = np.abs(cross) / line_len
        cross = cross[dist > min_line_dist]

    if len(cross) == 0:
        return 0, 0, "fallback"

    pos_count = int(np.sum(cross > 0))
    neg_count = int(np.sum(cross < 0))

    if pos_count > neg_count:
        return pos_count, neg_count, "positive"
    elif neg_count > pos_count:
        return pos_count, neg_count, "negative"
    else:
        return pos_count, neg_count, "fallback"


def _score_hinge_by_orange_pixels(
    endpoint: tuple[float, float],
    swing_if_hinge: str,
    p0: tuple[float, float],
    p1: tuple[float, float],
    orange_mask: np.ndarray,
    door_width_px: float,
    near_radius: float = 8.0,
    corridor_half_width: float = 6.0,
) -> float:
    """Score one endpoint as the door hinge using orange leaf-pixel evidence.

    Score = near-endpoint orange pixels (within near_radius)
           + orange pixels inside the candidate leaf corridor.

    The corridor extends door_width_px in the perpendicular (swing) direction
    from endpoint, with corridor_half_width tolerance across the origin axis.
    """
    ys, xs = np.where(orange_mask > 0)
    if len(xs) == 0:
        return 0.0

    px_arr = xs.astype(float)
    py_arr = ys.astype(float)

    near_score = float(np.sum(np.hypot(px_arr - endpoint[0], py_arr - endpoint[1]) <= near_radius))

    # Unit vector along door origin from endpoint toward far end
    far = p1 if (abs(endpoint[0] - p0[0]) < 1.0 and abs(endpoint[1] - p0[1]) < 1.0) else p0
    along_dx = far[0] - endpoint[0]
    along_dy = far[1] - endpoint[1]
    along_len = math.hypot(along_dx, along_dy)
    if along_len < 1e-3:
        return near_score

    ux = along_dx / along_len
    uy = along_dy / along_len
    # Perpendicular direction toward swing side (mirrors _perp_end math)
    sign = 1.0 if swing_if_hinge.replace("fallback_", "") == "left" else -1.0
    perp_x = sign * (-uy)
    perp_y = sign * ux

    rel_x = px_arr - endpoint[0]
    rel_y = py_arr - endpoint[1]
    along_proj = rel_x * perp_x + rel_y * perp_y   # leaf axis: 0..door_width_px
    across_proj = rel_x * ux + rel_y * uy           # origin axis: ±corridor_half_width

    corridor_mask_arr = (
        (along_proj > 0) &
        (along_proj <= door_width_px) &
        (np.abs(across_proj) <= corridor_half_width)
    )
    return near_score + float(np.sum(corridor_mask_arr))


def infer_door_direction_from_evidence(
    p0: tuple[float, float],
    p1: tuple[float, float],
    door_arc_mask: Optional[np.ndarray],
    door_leaf_mask: Optional[np.ndarray] = None,
    n_arc_samples: int = 16,
    min_score_threshold: float = 0.05,
) -> tuple[str, str, str, str]:
    """Infer door hinge and swing from local raster evidence (task35 hierarchy).

    Returns:
        (hinge_pt, swing_side, hinge_source, swing_source)
        where hinge_pt ∈ {"p0","p1"}, swing_side ∈ {"left","right","fallback_*"}
    """
    four_tuple, _ = _infer_with_evidence_fields(
        p0, p1, door_arc_mask, door_leaf_mask, n_arc_samples, min_score_threshold
    )
    return four_tuple


def _infer_with_evidence_fields(
    p0: tuple[float, float],
    p1: tuple[float, float],
    door_arc_mask: Optional[np.ndarray],
    door_leaf_mask: Optional[np.ndarray],
    n_arc_samples: int = 16,
    min_score_threshold: float = 0.05,
) -> tuple[tuple[str, str, str, str], dict]:
    """Like infer_door_direction_from_evidence() but also returns an evidence metrics dict."""
    ev: dict = {
        "red_side_positive_count": 0,
        "red_side_negative_count": 0,
        "red_side_selected": "",
        "orange_hinge_p0_score": 0.0,
        "orange_hinge_p1_score": 0.0,
        "hinge_selected": "",
        "fallback_used": False,
    }

    if door_arc_mask is None:
        ev["fallback_used"] = True
        return ("p0", "fallback_left", "fallback_pt0", "fallback"), ev

    width = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if width < 1e-3:
        ev["fallback_used"] = True
        return ("p0", "fallback_left", "fallback_pt0", "fallback"), ev

    # --- Primary swing: red-pixel side count ---
    pos_count, neg_count, abs_side = _score_side_by_red_pixels(p0, p1, door_arc_mask)
    ev["red_side_positive_count"] = pos_count
    ev["red_side_negative_count"] = neg_count
    ev["red_side_selected"] = abs_side

    if abs_side == "fallback" or (pos_count + neg_count) == 0:
        # No clear side → full 4-hypothesis arc scoring
        candidates: list[tuple[float, str, str]] = []
        for hinge_name, hinge, far in [("p0", p0, p1), ("p1", p1, p0)]:
            for swing in ["left", "right"]:
                score = _score_arc_pixels(hinge, far, swing, door_arc_mask, n_arc_samples)
                if door_leaf_mask is not None:
                    orientation_deg = math.degrees(math.atan2(far[1] - hinge[1], far[0] - hinge[0]))
                    leaf_end = _perp_end(hinge, width, orientation_deg, swing)
                    score += 0.3 * _score_line_pixels(hinge, leaf_end, door_leaf_mask, n_arc_samples)
                candidates.append((score, hinge_name, swing))
        best_score, best_hinge, best_swing = max(candidates, key=lambda x: x[0])
        if best_score < min_score_threshold:
            ev["fallback_used"] = True
            ev["hinge_selected"] = "p0"
            return ("p0", "fallback_left", "fallback_pt0", "fallback"), ev
        ev["hinge_selected"] = best_hinge
        return (best_hinge, best_swing, "red_orange_purple_evidence", "red_door_arc_side"), ev

    # Side decided: map to per-hinge swing candidates
    if abs_side == "positive":
        swing_if_p0, swing_if_p1 = "left", "right"
    else:
        swing_if_p0, swing_if_p1 = "right", "left"

    # --- Primary hinge: orange leaf corridor ---
    if door_leaf_mask is not None:
        h_p0 = _score_hinge_by_orange_pixels(p0, swing_if_p0, p0, p1, door_leaf_mask, width)
        h_p1 = _score_hinge_by_orange_pixels(p1, swing_if_p1, p0, p1, door_leaf_mask, width)
        ev["orange_hinge_p0_score"] = round(h_p0, 2)
        ev["orange_hinge_p1_score"] = round(h_p1, 2)
        if h_p0 > 0 or h_p1 > 0:
            if h_p0 >= h_p1:
                ev["hinge_selected"] = "p0"
                return ("p0", swing_if_p0, "red_orange_purple_evidence", "red_door_arc_side"), ev
            else:
                ev["hinge_selected"] = "p1"
                return ("p1", swing_if_p1, "red_orange_purple_evidence", "red_door_arc_side"), ev

    # --- Secondary hinge: arc-sampling with fixed swing (2 hypotheses) ---
    score_p0 = _score_arc_pixels(p0, p1, swing_if_p0, door_arc_mask, n_arc_samples)
    score_p1 = _score_arc_pixels(p1, p0, swing_if_p1, door_arc_mask, n_arc_samples)

    if max(score_p0, score_p1) < min_score_threshold:
        ev["fallback_used"] = True
        ev["hinge_selected"] = "p0"
        return ("p0", "fallback_left", "fallback_pt0", "fallback"), ev

    if score_p0 >= score_p1:
        ev["hinge_selected"] = "p0"
        return ("p0", swing_if_p0, "red_orange_purple_evidence", "red_door_arc_side"), ev
    else:
        ev["hinge_selected"] = "p1"
        return ("p1", swing_if_p1, "red_orange_purple_evidence", "red_door_arc_side"), ev


def compute_door_geometry(
    hosted_door: HostedOpening,
    swing_side: Optional[str] = None,
    door_arc_mask: Optional[np.ndarray] = None,
    door_leaf_mask: Optional[np.ndarray] = None,
) -> DoorGeometry:
    """Derive hinge, far-point, leaf-end, and arc geometry from a hosted door."""
    p0 = tuple(hosted_door.snapped_points[0])
    p1 = tuple(hosted_door.snapped_points[1])
    width = math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    _empty_ev: dict = {
        "red_side_positive_count": 0, "red_side_negative_count": 0,
        "red_side_selected": "", "orange_hinge_p0_score": 0.0,
        "orange_hinge_p1_score": 0.0, "hinge_selected": "p0", "fallback_used": False,
    }

    if swing_side is not None:
        hinge = p0
        far = p1
        hinge_source = "fallback_pt0"
        swing_source = "evidence"
        ev = _empty_ev
    elif door_arc_mask is not None:
        (hinge_pt, swing_side, hinge_source, swing_source), ev = _infer_with_evidence_fields(
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
        ev = {**_empty_ev, "fallback_used": True}

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
        red_side_positive_count=ev["red_side_positive_count"],
        red_side_negative_count=ev["red_side_negative_count"],
        red_side_selected=ev["red_side_selected"],
        orange_hinge_p0_score=ev["orange_hinge_p0_score"],
        orange_hinge_p1_score=ev["orange_hinge_p1_score"],
        hinge_selected=ev["hinge_selected"],
        fallback_used=ev["fallback_used"],
    )


def compute_door_geometry_double_swing(geom: DoorGeometry) -> DoorGeometry:
    """Return a copy of geom extended with a secondary swing on the opposite side.

    Used for double_swing_shared_origin doors: the hinge and origin are shared;
    a second leaf and arc are added on the side opposite to the primary swing.
    """
    primary_base = geom.swing_side.replace("fallback_", "")
    secondary_swing = "right" if primary_base == "left" else "left"
    secondary_leaf = _perp_end(geom.hinge_point, geom.width_px, geom.orientation_angle_deg, secondary_swing)
    return _dc_replace(
        geom,
        door_type="double_swing_shared_origin",
        secondary_leaf_end=secondary_leaf,
        secondary_swing_side=secondary_swing,
    )


def door_geometry_to_dict(geom: DoorGeometry, width_mm: Optional[float] = None) -> dict:
    """Serialize DoorGeometry to the final_vector.json door_geometry sub-dict."""
    d: dict = {
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
        # task35 evidence debug fields
        "red_side_positive_count": geom.red_side_positive_count,
        "red_side_negative_count": geom.red_side_negative_count,
        "red_side_selected": geom.red_side_selected,
        "orange_hinge_p0_score": geom.orange_hinge_p0_score,
        "orange_hinge_p1_score": geom.orange_hinge_p1_score,
        "hinge_selected": geom.hinge_selected,
        "fallback_used": geom.fallback_used,
        # task36 classification fields
        "door_type": geom.door_type,
        "classification_reason": geom.classification_reason,
        "double_swing_ratio": geom.double_swing_ratio,
        "source_door_component_ids": list(geom.source_door_component_ids),
    }
    if geom.secondary_leaf_end is not None:
        d["secondary_leaf_end"] = [round(geom.secondary_leaf_end[0], 2), round(geom.secondary_leaf_end[1], 2)]
        d["secondary_swing_side"] = geom.secondary_swing_side
    return d
