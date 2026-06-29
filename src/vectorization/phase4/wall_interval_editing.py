"""Insert opening endpoints into the wall graph and trim wall intervals.

Implements spec_v008 §9 (Opening Interval Editing).

Order of operations (must-rule, spec §14.4):
    1. Resolve overlapping opening intervals by adjustment (not rejection)
    2. Insert the two snapped opening endpoints as new nodes on the host edge
    3. Split the host edge at those two nodes
    4. Mark the interval between as an opening gap (removed from wall)
    5. Wall edges outside opening gaps are preserved as trimmed wall edges

Conflict resolution (spec §9 conflict behavior + task33):
    - door vs window: keep door fixed, move/shrink window
    - door vs door: keep higher-confidence door fixed, move/shrink lower
    - window vs window: keep higher-confidence window fixed, move/shrink lower
    - Reject only as last resort when no feasible non-overlapping placement exists
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .opening_hosting import HostedOpening, RejectedOpening

# Configurable conflict-resolution parameters
_MIN_OPENING_SEPARATOR_MM: float = 50.0   # desired gap between adjacent openings
_MAX_OPENING_ADJUSTMENT_MM: float = 200.0 # flag large adjustments as suspicious
_MIN_SEPARATOR_FALLBACK_PX: float = 3.0   # used when scale is unknown
_MAX_ADJUSTMENT_FALLBACK_PX: float = 30.0 # used when scale is unknown


@dataclass
class AdjustedOpening:
    """Opening with interval adjustment metadata from conflict resolution."""
    opening: HostedOpening
    original_t_start: float
    original_t_end: float
    adjusted_t_start: float
    adjusted_t_end: float
    center_t: float
    was_adjusted: bool
    adjustment_reason: str
    adjustment_px: float
    adjustment_mm: Optional[float]
    overlap_resolution_priority: str  # "door_fixed"|"higher_confidence_fixed"|"not_needed"
    large_adjustment_flagged: bool = False


@dataclass
class TrimmedGraph:
    """Wall graph after opening intervals are removed."""
    wall_edges: list[list[float]]
    opening_gaps: list[dict]              # one per accepted opening (includes adjustment meta)
    inserted_nodes: list[list[float]]
    last_resort_rejected: list[dict] = field(default_factory=list)  # openings rejected during conflict resolution


def _project_t(
    pt: tuple[float, float], x1: float, y1: float, x2: float, y2: float
) -> float:
    """Parametric t in [0,1] of point projected onto segment."""
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-9:
        return 0.0
    return max(0.0, min(1.0, ((pt[0] - x1) * dx + (pt[1] - y1) * dy) / seg_len_sq))


def _opening_priority(op: HostedOpening) -> tuple[int, float]:
    """Return (type_rank, confidence) for conflict resolution ordering.

    Higher type_rank = fixed in a conflict; within same rank higher confidence = fixed.
    door > window in type_rank.
    """
    type_rank = 1 if op.opening_type == "door" else 0
    return (type_rank, op.confidence)


def _resolve_conflicts(
    openings: list[HostedOpening],
    aligned_edges: list[list[float]],
    px_to_mm: Optional[float] = None,
    min_sep_mm: float = _MIN_OPENING_SEPARATOR_MM,
    max_adj_mm: float = _MAX_OPENING_ADJUSTMENT_MM,
    min_sep_fallback_px: float = _MIN_SEPARATOR_FALLBACK_PX,
    max_adj_fallback_px: float = _MAX_ADJUSTMENT_FALLBACK_PX,
) -> tuple[list[AdjustedOpening], list[dict]]:
    """Resolve overlapping opening intervals, preferring adjustment over rejection.

    Returns:
        (adjusted_openings, last_resort_rejected)

    Conflict priority (spec §9 + task33):
        door > window  (type priority)
        higher confidence > lower confidence (within same type)

    The lower-priority interval is pushed away from the fixed interval until
    the two intervals no longer overlap and a min separator is preserved.
    Rejection only occurs when no feasible non-overlapping position exists.
    """
    if not openings:
        return [], []

    # Group by host_edge_idx
    by_edge: dict[int, list[HostedOpening]] = {}
    for op in openings:
        by_edge.setdefault(op.host_edge_idx, []).append(op)

    adjusted: list[AdjustedOpening] = []
    last_resort_rejected: list[dict] = []

    for edge_idx, group in by_edge.items():
        if edge_idx >= len(aligned_edges):
            # Edge index out of bounds — accept without adjustment
            for op in group:
                adjusted.append(_make_adjusted(op, op_t=(0.0, 1.0), edge_len=1.0,
                                               px_to_mm=px_to_mm, priority="not_needed",
                                               reason="", adjusted=False))
            continue

        edge = aligned_edges[edge_idx]
        x1, y1, x2, y2 = edge[0], edge[1], edge[2], edge[3]
        edge_len = math.hypot(x2 - x1, y2 - y1)

        # Separator and max-adjustment in t-units
        if px_to_mm is not None and edge_len > 1e-6:
            min_sep_t = (min_sep_mm / px_to_mm) / edge_len
            max_adj_t = (max_adj_mm / px_to_mm) / edge_len
        else:
            min_sep_t = min_sep_fallback_px / max(edge_len, 1.0)
            max_adj_t = max_adj_fallback_px / max(edge_len, 1.0)

        # Compute initial t-intervals
        intervals: list[tuple[float, float, HostedOpening]] = []
        for op in group:
            ta = _project_t(op.snapped_points[0], x1, y1, x2, y2)
            tb = _project_t(op.snapped_points[1], x1, y1, x2, y2)
            tmin, tmax = min(ta, tb), max(ta, tb)
            intervals.append((tmin, tmax, op))

        # Sort by position, then by decreasing priority (so equal-start fixed > moved)
        intervals.sort(key=lambda x: (x[0], -_opening_priority(x[2])[0], -_opening_priority(x[2])[1]))

        # Greedy sweep: fix higher-priority, move lower-priority
        # We process left-to-right; for each new interval check against all already-placed ones
        placed: list[tuple[float, float, HostedOpening, str]] = []  # (t_start, t_end, op, priority_label)

        for tmin, tmax, op in intervals:
            cur_t_start = tmin
            cur_t_end = tmax
            width_t = tmax - tmin
            was_adjusted = False
            adjustment_reason = ""
            priority_label = "not_needed"
            large_flag = False

            # Check overlap with each already-placed interval (left-to-right)
            for p_start, p_end, p_op, _ in placed:
                # Overlap condition: cur_t_start < p_end + min_sep and cur_t_end > p_start
                if cur_t_start < p_end + min_sep_t and cur_t_end > p_start:
                    # Determine priority
                    cur_pri = _opening_priority(op)
                    p_pri = _opening_priority(p_op)

                    if cur_pri >= p_pri:
                        # Current has higher (or equal) priority — it should be fixed.
                        # The already-placed one would need to move left.
                        # Since we process left-to-right and already placed it, we cannot
                        # retroactively move it. Instead, we try to move the current one
                        # to the left of the placed one.
                        new_end = p_start - min_sep_t
                        new_start = new_end - width_t
                        if new_start >= 0.0:
                            adj_px = abs(new_start - tmin) * edge_len
                            adj_mm = adj_px / px_to_mm if px_to_mm else None
                            large_flag = (adj_px > max_adj_fallback_px if px_to_mm is None
                                          else (adj_mm or 0) > max_adj_mm)
                            cur_t_start = new_start
                            cur_t_end = new_end
                            was_adjusted = True
                            adjustment_reason = (
                                f"higher-priority {op.opening_type} moved left of "
                                f"lower-priority {p_op.opening_type} (type/confidence)"
                            )
                            priority_label = (
                                "door_fixed" if op.opening_type == "door" and p_op.opening_type == "window"
                                else "higher_confidence_fixed"
                            )
                        else:
                            # Cannot move left — fall through to right-push logic below
                            pass
                    else:
                        # Current has lower priority — push it right of the placed interval
                        new_start = p_end + min_sep_t
                        new_end = new_start + width_t
                        adj_px = abs(new_start - tmin) * edge_len
                        adj_mm = adj_px / px_to_mm if px_to_mm else None
                        large_flag = (adj_px > max_adj_fallback_px if px_to_mm is None
                                      else (adj_mm or 0) > max_adj_mm)
                        cur_t_start = new_start
                        cur_t_end = new_end
                        was_adjusted = True
                        adjustment_reason = (
                            f"{op.opening_type} interval moved right of "
                            f"{p_op.opening_type} (lower priority)"
                        )
                        priority_label = (
                            "door_fixed" if p_op.opening_type == "door" and op.opening_type == "window"
                            else "higher_confidence_fixed"
                        )

            # After resolving against all placed intervals, check if placement is feasible
            if cur_t_end > 1.0:
                # Try to shrink to fit within chain bounds
                cur_t_end = 1.0
                cur_t_start = max(cur_t_start, 1.0 - width_t)
                # Check it still doesn't overlap with placed intervals
                feasible = True
                for p_start, p_end, _, _ in placed:
                    if cur_t_start < p_end + min_sep_t and cur_t_end > p_start:
                        feasible = False
                        break
                if not feasible or cur_t_start >= cur_t_end:
                    last_resort_rejected.append({
                        "opening_type": op.opening_type,
                        "source_component_id": op.source_component_id,
                        "host_edge_idx": edge_idx,
                        "original_t_start": tmin,
                        "original_t_end": tmax,
                        "rejection_reason": "no_feasible_non_overlapping_interval",
                        "confidence": op.confidence,
                    })
                    continue
                was_adjusted = True
                if not adjustment_reason:
                    adjustment_reason = "shrunken to fit within wall chain bounds"

            if cur_t_start < 0.0:
                # Clamp to left boundary and re-check
                cur_t_start = 0.0
                cur_t_end = min(cur_t_start + width_t, 1.0)
                feasible = True
                for p_start, p_end, _, _ in placed:
                    if cur_t_start < p_end + min_sep_t and cur_t_end > p_start:
                        feasible = False
                        break
                if not feasible or cur_t_start >= cur_t_end:
                    last_resort_rejected.append({
                        "opening_type": op.opening_type,
                        "source_component_id": op.source_component_id,
                        "host_edge_idx": edge_idx,
                        "original_t_start": tmin,
                        "original_t_end": tmax,
                        "rejection_reason": "no_feasible_non_overlapping_interval",
                        "confidence": op.confidence,
                    })
                    continue
                was_adjusted = True
                if not adjustment_reason:
                    adjustment_reason = "clamped to wall chain left boundary"

            # Compute final adjustment metrics
            final_adj_px = abs(cur_t_start - tmin) * edge_len
            final_adj_mm = final_adj_px / px_to_mm if px_to_mm else None

            adj_op = AdjustedOpening(
                opening=op,
                original_t_start=tmin,
                original_t_end=tmax,
                adjusted_t_start=cur_t_start,
                adjusted_t_end=cur_t_end,
                center_t=(cur_t_start + cur_t_end) / 2.0,
                was_adjusted=was_adjusted,
                adjustment_reason=adjustment_reason,
                adjustment_px=final_adj_px,
                adjustment_mm=final_adj_mm,
                overlap_resolution_priority=priority_label,
                large_adjustment_flagged=large_flag,
            )
            adjusted.append(adj_op)
            placed.append((cur_t_start, cur_t_end, op, priority_label))

    return adjusted, last_resort_rejected


def apply_adjusted_intervals_to_hosted_openings(
    trimmed_graph: TrimmedGraph,
    hosted_openings: list[HostedOpening],
) -> list[HostedOpening]:
    """Return new HostedOpening objects whose snapped_points match the adjusted trim intervals.

    After _resolve_conflicts() adjusts opening intervals, the trimmed_graph.opening_gaps
    store the final snapped_points computed from those adjusted t-values.  The original
    HostedOpening objects still carry the pre-adjustment snapped_points, which means
    SVG/JSON/debug exports would draw primitives at the WRONG location.

    This function creates one authoritative HostedOpening per accepted opening, with
    snapped_points == the wall trim endpoint pair, so every downstream consumer sees
    the same adjusted geometry.

    Openings whose gap was last-resort-rejected are dropped (they are already in
    trimmed_graph.last_resort_rejected).
    """
    # Index gaps: (source_component_id, opening_type, host_edge_idx) -> gap dict
    gap_index: dict[tuple, dict] = {}
    for gap in trimmed_graph.opening_gaps:
        key = (gap["source_component_id"], gap["opening_type"], gap["host_edge_idx"])
        gap_index[key] = gap

    result: list[HostedOpening] = []
    for op in hosted_openings:
        key = (op.source_component_id, op.opening_type, op.host_edge_idx)
        gap = gap_index.get(key)
        if gap is None:
            continue  # last-resort rejected or not found

        adj_pts = gap["snapped_points"]   # [[x,y],[x,y]] from adjusted t values
        snap_a = (float(adj_pts[0][0]), float(adj_pts[0][1]))
        snap_b = (float(adj_pts[1][0]), float(adj_pts[1][1]))
        width_px = math.hypot(snap_b[0] - snap_a[0], snap_b[1] - snap_a[1])

        result.append(HostedOpening(
            opening_type=op.opening_type,
            source_component_id=op.source_component_id,
            host_edge_idx=op.host_edge_idx,
            host_edge_raw=op.host_edge_raw,
            raw_points=op.raw_points,
            snapped_points=[snap_a, snap_b],
            width_px=width_px,
            width_mm=gap.get("width_mm"),
            confidence=op.confidence,
            snapped_module_mm=op.snapped_module_mm,
        ))

    return result


def _make_adjusted(
    op: HostedOpening,
    op_t: tuple[float, float],
    edge_len: float,
    px_to_mm: Optional[float],
    priority: str,
    reason: str,
    adjusted: bool,
) -> AdjustedOpening:
    """Build an AdjustedOpening for an opening that needed no conflict resolution."""
    return AdjustedOpening(
        opening=op,
        original_t_start=op_t[0],
        original_t_end=op_t[1],
        adjusted_t_start=op_t[0],
        adjusted_t_end=op_t[1],
        center_t=(op_t[0] + op_t[1]) / 2.0,
        was_adjusted=adjusted,
        adjustment_reason=reason,
        adjustment_px=0.0,
        adjustment_mm=None,
        overlap_resolution_priority=priority,
    )


def trim_wall_intervals(
    aligned_edges: list[list[float]],
    hosted_openings: list[HostedOpening],
    px_to_mm: Optional[float] = None,
    min_sep_mm: float = _MIN_OPENING_SEPARATOR_MM,
    max_adj_mm: float = _MAX_OPENING_ADJUSTMENT_MM,
) -> TrimmedGraph:
    """Trim wall intervals by inserting opening endpoints and removing opening spans.

    Overlapping intervals on the same edge are de-overlapped by adjustment before
    trimming; rejection is a last resort only (spec §9 + task33).

    Args:
        aligned_edges: List of [x1,y1,x2,y2] wall edges after alignment
        hosted_openings: Accepted openings to insert
        px_to_mm: Scale factor for separator/adjustment metric limits
        min_sep_mm: Minimum separator between adjacent openings in mm
        max_adj_mm: Flag adjustments exceeding this distance

    Returns:
        TrimmedGraph with remaining wall_edges, opening_gaps, inserted_nodes,
        and last_resort_rejected
    """
    adjusted_openings, last_resort_rejected = _resolve_conflicts(
        hosted_openings,
        aligned_edges,
        px_to_mm=px_to_mm,
        min_sep_mm=min_sep_mm,
        max_adj_mm=max_adj_mm,
    )

    # Map host_edge_idx -> list of AdjustedOpening on that edge (already de-overlapped)
    by_edge: dict[int, list[AdjustedOpening]] = {}
    for adj in adjusted_openings:
        by_edge.setdefault(adj.opening.host_edge_idx, []).append(adj)

    wall_edges: list[list[float]] = []
    opening_gaps: list[dict] = []
    inserted_nodes: list[list[float]] = []

    for ei, edge in enumerate(aligned_edges):
        x1, y1, x2, y2 = edge[0], edge[1], edge[2], edge[3]
        edge_len = math.hypot(x2 - x1, y2 - y1)

        if ei not in by_edge or edge_len < 1e-6:
            wall_edges.append(edge)
            continue

        adj_here = by_edge[ei]
        # Sort by adjusted t_start
        adj_here.sort(key=lambda a: a.adjusted_t_start)

        # Walk through the edge emitting wall sub-segments between opening gaps
        t_cursor = 0.0
        for adj in adj_here:
            tmin = adj.adjusted_t_start
            tmax = adj.adjusted_t_end

            # Wall segment before this opening
            if tmin - t_cursor > 1e-4:
                ax = x1 + t_cursor * (x2 - x1)
                ay = y1 + t_cursor * (y2 - y1)
                bx = x1 + tmin * (x2 - x1)
                by_ = y1 + tmin * (y2 - y1)
                wall_edges.append([ax, ay, bx, by_])

            # Snapped endpoints using adjusted t values
            snap_a = (x1 + tmin * (x2 - x1), y1 + tmin * (y2 - y1))
            snap_b = (x1 + tmax * (x2 - x1), y1 + tmax * (y2 - y1))
            inserted_nodes.extend([[snap_a[0], snap_a[1]], [snap_b[0], snap_b[1]]])

            op = adj.opening
            opening_gaps.append({
                "opening_type": op.opening_type,
                "source_component_id": op.source_component_id,
                "host_edge_idx": ei,
                "host_edge_raw": edge,
                "snapped_points": [list(snap_a), list(snap_b)],
                # Interval data
                "original_interval": [adj.original_t_start, adj.original_t_end],
                "adjusted_interval": [adj.adjusted_t_start, adj.adjusted_t_end],
                "t_start": tmin,
                "t_end": tmax,
                "width_px": op.width_px,
                "width_mm": op.width_mm,
                "confidence": op.confidence,
                # Adjustment metadata
                "was_adjusted": adj.was_adjusted,
                "adjustment_reason": adj.adjustment_reason,
                "adjustment_px": round(adj.adjustment_px, 2),
                "adjustment_mm": round(adj.adjustment_mm, 2) if adj.adjustment_mm is not None else None,
                "overlap_resolution_priority": adj.overlap_resolution_priority,
                "large_adjustment_flagged": adj.large_adjustment_flagged,
            })
            t_cursor = tmax

        # Remaining wall segment after last opening
        if 1.0 - t_cursor > 1e-4:
            ax = x1 + t_cursor * (x2 - x1)
            ay = y1 + t_cursor * (y2 - y1)
            wall_edges.append([ax, ay, x2, y2])

    return TrimmedGraph(
        wall_edges=wall_edges,
        opening_gaps=opening_gaps,
        inserted_nodes=inserted_nodes,
        last_resort_rejected=last_resort_rejected,
    )
