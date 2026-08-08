"""Wall junction and wall edge metrics (spec_v006 §4.1, §4.2). Primary.

These are the metrics that distinguish a genuinely recovered wall topology from
geometry that merely lands near the right pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.evaluation.matching import (
    MatchResult,
    match_points,
    prf,
    segment_overlap_ratio,
)
from src.evaluation.types import Graph

TOL_NODE = 8.0
TOL_NODE_RELAXED = 16.0
ORTHOGONAL_TOLERANCE_DEG = 10.0
MIN_EDGE_OVERLAP_RATIO = 0.5


@dataclass
class TopologyResult:
    """Per-sample topology outcome. Counts aggregate; ratios are per sample."""

    counts: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    node_match: MatchResult | None = None
    edge_match_pairs: list[tuple[int, int]] = field(default_factory=list)


def evaluate_topology(
    predicted: Graph,
    reference: Graph,
    tolerance: float = TOL_NODE,
    relaxed_tolerance: float = TOL_NODE_RELAXED,
) -> TopologyResult:
    """Score one predicted wall graph against its reference."""
    result = TopologyResult()

    node_match = match_points(predicted.nodes, reference.nodes, tolerance)
    result.node_match = node_match

    p, r, f1 = prf(node_match.tp, node_match.fp, node_match.fn)
    result.metrics.update(node_precision=p, node_recall=r, node_f1=f1)
    result.counts.update(node_tp=node_match.tp, node_fp=node_match.fp, node_fn=node_match.fn)

    relaxed = match_points(predicted.nodes, reference.nodes, relaxed_tolerance)
    rp, rr, rf1 = prf(relaxed.tp, relaxed.fp, relaxed.fn)
    result.metrics.update(node_precision_relaxed=rp, node_recall_relaxed=rr, node_f1_relaxed=rf1)

    _add_degree_aware(result, predicted, reference, node_match)
    _add_edge_metrics(result, predicted, reference, node_match)
    _add_structure_metrics(result, predicted, reference)
    _add_distance_diagnostics(result, predicted, reference, node_match)
    return result


def _add_degree_aware(
    result: TopologyResult,
    predicted: Graph,
    reference: Graph,
    node_match: MatchResult,
) -> None:
    """Degree-aware junction F1 (spec_v006 §4.1)."""
    pred_deg = predicted.degree()
    ref_deg = reference.degree()

    exact = 0
    within1 = 0
    for pi, ri in node_match.pairs:
        delta = abs(pred_deg[pi] - ref_deg[ri])
        if delta == 0:
            exact += 1
        if delta <= 1:
            within1 += 1

    total_pred = len(predicted.nodes)
    total_ref = len(reference.nodes)
    for label, tp in (("degree_exact", exact), ("degree_within_1", within1)):
        p, r, f1 = prf(tp, total_pred - tp, total_ref - tp)
        result.metrics[f"junction_{label}_precision"] = p
        result.metrics[f"junction_{label}_recall"] = r
        result.metrics[f"junction_{label}_f1"] = f1

    result.counts["broken_junctions"] = node_match.tp - exact


def _add_edge_metrics(
    result: TopologyResult,
    predicted: Graph,
    reference: Graph,
    node_match: MatchResult,
) -> None:
    """Topological edge matching plus the segment-overlap view (spec_v006 §4.2)."""
    pred_to_ref = dict(node_match.pairs)

    ref_edge_key = {}
    for idx, (a, b) in enumerate(reference.edges):
        ref_edge_key[(a, b) if a < b else (b, a)] = idx

    topo_pairs: list[tuple[int, int]] = []
    used_ref: set[int] = set()
    for p_idx, (pa, pb) in enumerate(predicted.edges):
        ra = pred_to_ref.get(pa)
        rb = pred_to_ref.get(pb)
        if ra is None or rb is None:
            continue
        key = (ra, rb) if ra < rb else (rb, ra)
        r_idx = ref_edge_key.get(key)
        if r_idx is None or r_idx in used_ref:
            continue
        used_ref.add(r_idx)
        topo_pairs.append((p_idx, r_idx))

    tp = len(topo_pairs)
    fp = len(predicted.edges) - tp
    fn = len(reference.edges) - tp
    p, r, f1 = prf(tp, fp, fn)
    result.metrics.update(edge_precision=p, edge_recall=r, edge_f1=f1)
    result.counts.update(edge_tp=tp, edge_fp=fp, edge_fn=fn)
    result.edge_match_pairs = topo_pairs

    # Second view: collinear-and-overlapping, independent of node assignment.
    pred_segs = predicted.edge_segments()
    ref_segs = reference.edge_segments()
    overlap_used: set[int] = set()
    overlap_tp = 0
    for pseg in pred_segs:
        best_idx = -1
        best_ratio = 0.0
        for j, rseg in enumerate(ref_segs):
            if j in overlap_used:
                continue
            perp, ratio = segment_overlap_ratio(pseg, rseg)
            if perp <= TOL_NODE and ratio >= MIN_EDGE_OVERLAP_RATIO and ratio > best_ratio:
                best_idx, best_ratio = j, ratio
        if best_idx >= 0:
            overlap_used.add(best_idx)
            overlap_tp += 1

    op, orr, of1 = prf(overlap_tp, len(pred_segs) - overlap_tp, len(ref_segs) - overlap_tp)
    result.metrics.update(edge_overlap_precision=op, edge_overlap_recall=orr, edge_overlap_f1=of1)
    result.counts["edge_overlap_tp"] = overlap_tp


def _add_structure_metrics(result: TopologyResult, predicted: Graph, reference: Graph) -> None:
    """Connected-component error and invalid/dangling topology rates (spec_v006 §4.2)."""
    pred_components = predicted.component_count()
    ref_components = reference.component_count()
    result.counts.update(predicted_components=pred_components, reference_components=ref_components)
    result.metrics["component_count_error"] = float(pred_components - ref_components)
    result.metrics["component_count_abs_error"] = float(abs(pred_components - ref_components))
    result.metrics["component_count_exact"] = float(pred_components == ref_components)

    for label, graph in (("predicted", predicted), ("reference", reference)):
        for key, value in _structure_ratios(graph).items():
            result.metrics[f"{label}_{key}"] = value


def _structure_ratios(graph: Graph) -> dict[str, float]:
    degrees = graph.degree()
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)

    dangling = sum(1 for d in degrees if d == 1)
    isolated = sum(1 for d in degrees if d == 0)

    non_orthogonal = 0
    zero_length = 0
    for (x1, y1), (x2, y2) in graph.edge_segments():
        dx, dy = x2 - x1, y2 - y1
        if math.hypot(dx, dy) < 1.0:
            zero_length += 1
            continue
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        if min(angle, abs(90.0 - angle)) > ORTHOGONAL_TOLERANCE_DEG:
            non_orthogonal += 1

    seen: set[tuple[int, int]] = set()
    duplicates = 0
    for a, b in graph.edges:
        key = (a, b) if a < b else (b, a)
        if key in seen:
            duplicates += 1
        seen.add(key)

    return {
        "dangling_node_ratio": dangling / n_nodes if n_nodes else 0.0,
        "isolated_node_ratio": isolated / n_nodes if n_nodes else 0.0,
        "non_orthogonal_ratio": non_orthogonal / n_edges if n_edges else 0.0,
        "duplicate_edge_count": float(duplicates),
        "zero_length_edge_count": float(zero_length),
    }


def _add_distance_diagnostics(
    result: TopologyResult,
    predicted: Graph,
    reference: Graph,
    node_match: MatchResult,
) -> None:
    """Secondary geometric diagnostics (spec_v006 §4.7). Never the headline."""
    if not node_match.pairs:
        result.metrics["diag_node_distance_mean_px"] = float("nan")
        result.metrics["diag_node_distance_median_px"] = float("nan")
        return

    distances = sorted(
        math.dist(predicted.nodes[pi], reference.nodes[ri]) for pi, ri in node_match.pairs
    )
    mid = len(distances) // 2
    median = distances[mid] if len(distances) % 2 else (distances[mid - 1] + distances[mid]) / 2
    result.metrics["diag_node_distance_mean_px"] = sum(distances) / len(distances)
    result.metrics["diag_node_distance_median_px"] = median
