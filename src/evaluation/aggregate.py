"""Micro and macro aggregation of per-sample topology results (spec_v006 §4).

Micro pools TP/FP/FN across samples then computes P/R/F1. Macro averages the
per-sample P/R/F1. Both are reported; neither is published alone, because they
disagree exactly where it matters — a plan with two openings and a plan with
thirty should not carry equal weight silently.
"""

from __future__ import annotations

import math

from src.evaluation.matching import prf
from src.evaluation.types import SampleRecord

_MICRO_FAMILIES = {
    "node": ("node_tp", "node_fp", "node_fn"),
    "edge": ("edge_tp", "edge_fp", "edge_fn"),
    "adjacency": ("adjacency_tp", "adjacency_fp", "adjacency_fn"),
}


def aggregate_topology(records: list[SampleRecord]) -> dict[str, float]:
    """Aggregate scored records into one flat metric dict."""
    scored = [r for r in records if r.scored]
    out: dict[str, float] = {"_denominator": float(len(scored))}
    if not scored:
        return out

    metric_keys: set[str] = set()
    for record in scored:
        metric_keys.update(record.metrics)

    for key in sorted(metric_keys):
        values = [
            r.metrics[key] for r in scored if key in r.metrics and not math.isnan(r.metrics[key])
        ]
        out[f"macro_{key}"] = sum(values) / len(values) if values else float("nan")
        out[f"n_{key}"] = float(len(values))

    for family, (tp_key, fp_key, fn_key) in _MICRO_FAMILIES.items():
        tp = sum(int(r.counts.get(tp_key, 0)) for r in scored)
        fp = sum(int(r.counts.get(fp_key, 0)) for r in scored)
        fn = sum(int(r.counts.get(fn_key, 0)) for r in scored)
        precision, recall, f1 = prf(tp, fp, fn)
        out[f"micro_{family}_precision"] = precision
        out[f"micro_{family}_recall"] = recall
        out[f"micro_{family}_f1"] = f1

    for opening_type in ("door", "window"):
        tp = sum(int(r.counts.get(f"{opening_type}_tp", 0)) for r in scored)
        fp = sum(int(r.counts.get(f"spurious_{opening_type}s", 0)) for r in scored)
        fn = sum(int(r.counts.get(f"missing_{opening_type}s", 0)) for r in scored)
        precision, recall, f1 = prf(tp, fp, fn)
        out[f"micro_{opening_type}_precision"] = precision
        out[f"micro_{opening_type}_recall"] = recall
        out[f"micro_{opening_type}_f1"] = f1

    matched = sum(int(r.counts.get("matched_openings", 0)) for r in scored)
    hosted = sum(int(r.counts.get("correctly_hosted_openings", 0)) for r in scored)
    gap_ok = sum(int(r.counts.get("gap_correct_openings", 0)) for r in scored)
    out["micro_host_accuracy"] = hosted / matched if matched else float("nan")
    out["micro_gap_accuracy"] = gap_ok / hosted if hosted else float("nan")
    out["_matched_openings"] = float(matched)

    for key in (
        "unhosted_openings",
        "missing_doors",
        "missing_windows",
        "spurious_doors",
        "spurious_windows",
        "mistyped_openings",
        "broken_junctions",
        "duplicate_gaps",
        "disconnected_circulation",
        "false_connections",
        "false_separations",
    ):
        out[f"total_{key}"] = float(sum(int(r.counts.get(key, 0) or 0) for r in scored))

    return out


def headline(aggregated: dict[str, float]) -> dict[str, float]:
    """The primary numbers a README may quote, spatial logic first."""
    keys = [
        "macro_junction_degree_exact_f1",
        "macro_junction_degree_within_1_f1",
        "macro_node_f1",
        "macro_edge_f1",
        "macro_edge_overlap_f1",
        "macro_door_f1",
        "macro_window_f1",
        "macro_host_accuracy",
        "macro_gap_accuracy",
        "macro_adjacency_f1",
        "macro_reachability_agreement",
        "macro_component_count_abs_error",
        "micro_node_f1",
        "micro_edge_f1",
        "micro_door_f1",
        "micro_window_f1",
        "micro_host_accuracy",
        "micro_adjacency_f1",
    ]
    return {k: aggregated[k] for k in keys if k in aggregated}
