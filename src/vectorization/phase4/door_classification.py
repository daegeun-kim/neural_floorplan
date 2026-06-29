"""Door classification: single-swing, double-swing, ignored duplicates (task36).

After host_openings(), this module runs before wall trimming to:
  Part A — detect two-sided red evidence for each hosted door
  Part B — find pairs of doors that share the same wall origin segment
  Part C — classify each door/pair as single_swing, double_swing_shared_origin,
            ignored_duplicate, or separate_single_swing_doors
  Part F — merge paired doors into one HostedOpening for the trimmer

Constants are module-level for easy tuning; they are not hidden in expressions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .door_geometry import _score_side_by_red_pixels
from .opening_hosting import HostedOpening, RejectedOpening

# ─── Thresholds ────────────────────────────────────────────────────────────────
MIN_SIDE_PIXELS: int = 10          # minimum pixels on a side to consider it "supported"
MIN_DOUBLE_SWING_RATIO: float = 0.30   # weaker / stronger pixel count ratio for double-swing
SAME_ORIGIN_OVERLAP_RATIO: float = 0.75   # min fraction of shorter interval that must overlap
SAME_ORIGIN_ENDPOINT_TOL_PX: float = 10.0  # endpoint proximity for "same origin" check


@dataclass
class DoorClassification:
    """Classification decision for one final door entry (parallel to final_doors list)."""
    door_type: str                  # "single_swing" | "double_swing_shared_origin"
    classification: str             # same, or "ignored_duplicate"
    source_component_ids: list      # all component IDs that contributed to this door
    decision_reason: str
    red_positive_count: int = 0
    red_negative_count: int = 0
    double_swing_ratio: Optional[float] = None
    merged_from_component_ids: list = field(default_factory=list)
    ignored_as_duplicate_of: Optional[int] = None  # component_id of the surviving door


@dataclass
class ClassificationResult:
    """Return value of classify_door_openings()."""
    final_doors: list              # HostedOpening list to pass to trim_wall_intervals()
    classifications: list          # DoorClassification, one per entry in final_doors
    ignored_doors: list            # HostedOpening objects classified as ignored_duplicate
    double_swing_count: int = 0
    ignored_duplicate_count: int = 0


def _snapped_to_t(snapped_points: list, edge: list) -> tuple[float, float]:
    """Project snapped_points onto the host edge and return (t_start, t_end) in [0,1]."""
    x1, y1, x2, y2 = edge[0], edge[1], edge[2], edge[3]
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < 1e-6:
        return 0.0, 1.0
    t0 = ((snapped_points[0][0] - x1) * dx + (snapped_points[0][1] - y1) * dy) / L2
    t1 = ((snapped_points[1][0] - x1) * dx + (snapped_points[1][1] - y1) * dy) / L2
    return (min(t0, t1), max(t0, t1))


def _interval_overlap_ratio(
    t_a: tuple[float, float],
    t_b: tuple[float, float],
) -> float:
    """Overlap length / shorter interval length (result in [0,1])."""
    a0, a1 = t_a
    b0, b1 = t_b
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    shorter = min(a1 - a0, b1 - b0)
    return overlap / shorter if shorter > 1e-6 else 0.0


def _local_mask_for_door(
    door: HostedOpening,
    door_arc_mask: Optional[np.ndarray],
    comp,
    pad: int = 8,
) -> Optional[np.ndarray]:
    """Crop door_arc_mask to the component bbox + pad, zeroing the rest."""
    if door_arc_mask is None:
        return None
    if comp is None:
        return door_arc_mask
    x0, y0, x1, y1 = comp.bbox
    h, w = door_arc_mask.shape[:2]
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
    local = np.zeros_like(door_arc_mask)
    local[cy0:cy1, cx0:cx1] = door_arc_mask[cy0:cy1, cx0:cx1]
    return local


def _is_two_sided(pos_count: int, neg_count: int) -> tuple[bool, float]:
    """Return (two_sided, ratio). two_sided=True when both sides have enough evidence."""
    if pos_count < MIN_SIDE_PIXELS or neg_count < MIN_SIDE_PIXELS:
        return False, 0.0
    stronger = max(pos_count, neg_count)
    weaker = min(pos_count, neg_count)
    ratio = weaker / stronger
    return ratio >= MIN_DOUBLE_SWING_RATIO, ratio


def _build_merged_hosted_opening(
    primary: HostedOpening,
    secondary: HostedOpening,
    t_primary: tuple[float, float],
    t_secondary: tuple[float, float],
) -> HostedOpening:
    """Create a merged HostedOpening spanning the union of both intervals."""
    edge = primary.host_edge_raw
    x1, y1, x2, y2 = edge[0], edge[1], edge[2], edge[3]
    t_start = min(t_primary[0], t_secondary[0])
    t_end = max(t_primary[1], t_secondary[1])
    p0 = (x1 + t_start * (x2 - x1), y1 + t_start * (y2 - y1))
    p1 = (x1 + t_end   * (x2 - x1), y1 + t_end   * (y2 - y1))
    return HostedOpening(
        opening_type="door",
        source_component_id=primary.source_component_id,
        host_edge_idx=primary.host_edge_idx,
        host_edge_raw=primary.host_edge_raw,
        raw_points=primary.raw_points,
        snapped_points=[p0, p1],
        width_px=max(primary.width_px, secondary.width_px),
        width_mm=primary.width_mm,
        confidence=max(primary.confidence, secondary.confidence),
        snapped_module_mm=primary.snapped_module_mm,
    )


def classify_door_openings(
    hosted_doors: list[HostedOpening],
    door_arc_mask: Optional[np.ndarray],
    door_arc_comps: dict,
) -> ClassificationResult:
    """Classify hosted doors and merge shared-origin pairs.

    Args:
        hosted_doors:   output of host_openings(); snapped_points are pre-adjustment.
        door_arc_mask:  global door_arc segmentation mask; will be cropped per door.
        door_arc_comps: {component_id: ComponentRecord} for bbox cropping.

    Returns:
        ClassificationResult with final_doors for trimming + per-door classifications.
    """
    n = len(hosted_doors)
    if n == 0:
        return ClassificationResult(
            final_doors=[], classifications=[], ignored_doors=[],
            double_swing_count=0, ignored_duplicate_count=0,
        )

    # ─── Step 1: per-door evidence ───────────────────────────────────────────
    evidences: list[dict] = []
    for door in hosted_doors:
        comp = door_arc_comps.get(door.source_component_id)
        local_mask = _local_mask_for_door(door, door_arc_mask, comp)
        p0 = tuple(door.snapped_points[0])
        p1 = tuple(door.snapped_points[1])
        if local_mask is not None and np.any(local_mask > 0):
            pos, neg, side = _score_side_by_red_pixels(p0, p1, local_mask)
        else:
            pos, neg, side = 0, 0, "fallback"
        two_sided, ratio = _is_two_sided(pos, neg)
        evidences.append({"pos": pos, "neg": neg, "side": side,
                          "two_sided": two_sided, "ratio": ratio})

    # ─── Step 2: find pairs on same edge with overlapping intervals ──────────
    t_values = [_snapped_to_t(d.snapped_points, d.host_edge_raw) for d in hosted_doors]

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if hosted_doors[i].host_edge_idx != hosted_doors[j].host_edge_idx:
                continue
            if _interval_overlap_ratio(t_values[i], t_values[j]) >= SAME_ORIGIN_OVERLAP_RATIO:
                pairs.append((i, j))

    # ─── Step 3: classify pairs (greedy — each door is assigned to at most one pair) ──
    assigned: set[int] = set()

    final_doors: list[HostedOpening] = []
    classifications: list[DoorClassification] = []
    ignored_doors: list[HostedOpening] = []
    double_swing_count = 0
    ignored_duplicate_count = 0

    for (i, j) in pairs:
        if i in assigned or j in assigned:
            continue  # already consumed by an earlier pair
        assigned.add(i)
        assigned.add(j)

        ev_i, ev_j = evidences[i], evidences[j]
        di, dj = hosted_doors[i], hosted_doors[j]

        opposite = (
            ev_i["side"] in ("positive", "negative") and
            ev_j["side"] in ("positive", "negative") and
            ev_i["side"] != ev_j["side"]
        )

        if opposite:
            # Merge into one double-swing door
            primary, secondary = (di, dj) if di.confidence >= dj.confidence else (dj, di)
            pi, si = (i, j) if di.confidence >= dj.confidence else (j, i)
            merged = _build_merged_hosted_opening(primary, secondary, t_values[pi], t_values[si])
            combined_pos = ev_i["pos"] + ev_j["pos"]
            combined_neg = ev_i["neg"] + ev_j["neg"]
            stronger = max(combined_pos, combined_neg)
            ratio = min(combined_pos, combined_neg) / stronger if stronger > 0 else 0.0
            final_doors.append(merged)
            classifications.append(DoorClassification(
                door_type="double_swing_shared_origin",
                classification="double_swing_shared_origin",
                source_component_ids=[primary.source_component_id, secondary.source_component_id],
                decision_reason="opposite_side_evidence_merged_from_pair",
                red_positive_count=combined_pos,
                red_negative_count=combined_neg,
                double_swing_ratio=ratio,
                merged_from_component_ids=[primary.source_component_id, secondary.source_component_id],
            ))
            double_swing_count += 1
        else:
            # Same side or no evidence — keep the stronger, ignore the weaker duplicate
            i_str = ev_i["pos"] + ev_i["neg"]
            j_str = ev_j["pos"] + ev_j["neg"]
            survivor_idx = i if i_str >= j_str else j
            dup_idx = j if i_str >= j_str else i
            survivor_ev = ev_i if survivor_idx == i else ev_j
            final_doors.append(hosted_doors[survivor_idx])
            classifications.append(DoorClassification(
                door_type="single_swing",
                classification="single_swing",
                source_component_ids=[hosted_doors[survivor_idx].source_component_id],
                decision_reason="same_origin_same_side_weaker_ignored",
                red_positive_count=survivor_ev["pos"],
                red_negative_count=survivor_ev["neg"],
            ))
            ignored_doors.append(hosted_doors[dup_idx])
            ignored_duplicate_count += 1

    # ─── Step 4: un-assigned doors (not consumed by any pair) ───────────────
    for i in range(n):
        if i in assigned:
            continue
        door = hosted_doors[i]
        ev = evidences[i]
        if ev["two_sided"]:
            final_doors.append(door)
            classifications.append(DoorClassification(
                door_type="double_swing_shared_origin",
                classification="double_swing_shared_origin",
                source_component_ids=[door.source_component_id],
                decision_reason="single_component_two_sided_red_evidence",
                red_positive_count=ev["pos"],
                red_negative_count=ev["neg"],
                double_swing_ratio=ev["ratio"],
                merged_from_component_ids=[],
            ))
            double_swing_count += 1
        else:
            final_doors.append(door)
            classifications.append(DoorClassification(
                door_type="single_swing",
                classification="single_swing",
                source_component_ids=[door.source_component_id],
                decision_reason="one_sided_red_evidence",
                red_positive_count=ev["pos"],
                red_negative_count=ev["neg"],
            ))

    return ClassificationResult(
        final_doors=final_doors,
        classifications=classifications,
        ignored_doors=ignored_doors,
        double_swing_count=double_swing_count,
        ignored_duplicate_count=ignored_duplicate_count,
    )
