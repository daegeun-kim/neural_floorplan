"""Stable output schemas for an evaluation run (spec_v006 §6).

Writes only compact, reviewer-useful artifacts. Raw predictions, dataset copies,
full checkpoints, and large debug dumps stay outside the committed result set.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from src.evaluation.types import SampleRecord

RESULT_FILES = (
    "provenance.json",
    "summary.json",
    "per_sample.csv",
    "segmentation.csv",
    "topology.csv",
    "baseline_comparison.csv",
    "failures.json",
    "selected_cases.json",
)

# spec_v006 §6.3 - the end-to-end score used to pick representative cases.
SELECTION_METRICS = (
    "junction_degree_exact_f1",
    "edge_f1",
    "door_f1",
    "window_f1",
    "host_accuracy",
    "reachability_agreement",
)

FAILURE_COUNTERS = (
    "unhosted_openings",
    "missing_doors",
    "missing_windows",
    "spurious_doors",
    "spurious_windows",
    "mistyped_openings",
    "broken_junctions",
    "duplicate_gaps",
    "disconnected_circulation",
)


def _json_safe(value: Any) -> Any:
    """NaN is not valid JSON; it becomes null so the files stay parseable."""
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _csv_value(row.get(c)) for c in columns})


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return value


def end_to_end_score(metrics: dict[str, float]) -> float:
    """Unweighted mean of the six primary metrics, skipping unavailable ones."""
    values = [
        metrics[name]
        for name in SELECTION_METRICS
        if name in metrics and not math.isnan(metrics[name])
    ]
    return sum(values) / len(values) if values else float("nan")


def select_cases(records: list[SampleRecord]) -> dict[str, Any]:
    """Deterministic success / mixed / failure selection (spec_v006 §6.3)."""
    scored = [
        (end_to_end_score(r.metrics), r.sample_id)
        for r in records
        if r.scored and not math.isnan(end_to_end_score(r.metrics))
    ]
    if not scored:
        return {
            "rule": _SELECTION_RULE,
            "metrics_used": list(SELECTION_METRICS),
            "cases": {},
            "note": "no scorable samples",
        }

    # Sort by score, then by sample_id, so ties resolve identically every run.
    scored.sort(key=lambda item: (item[0], item[1]))
    median_score = scored[len(scored) // 2][0]
    mixed = min(scored, key=lambda item: (abs(item[0] - median_score), item[1]))

    return {
        "rule": _SELECTION_RULE,
        "metrics_used": list(SELECTION_METRICS),
        "cases": {
            "failure": {"sample_id": scored[0][1], "score": scored[0][0]},
            "mixed": {"sample_id": mixed[1], "score": mixed[0]},
            "success": {"sample_id": scored[-1][1], "score": scored[-1][0]},
        },
        "scored_sample_count": len(scored),
    }


_SELECTION_RULE = (
    "end_to_end_score = unweighted mean of "
    + ", ".join(SELECTION_METRICS)
    + "; success = max, failure = min, mixed = closest to median; "
    "ties broken by ascending sample_id"
)


def collect_failures(records: list[SampleRecord]) -> dict[str, Any]:
    """Failure category totals plus the sample ids exhibiting each (spec_v006 §4.5)."""
    totals: dict[str, int] = dict.fromkeys(FAILURE_COUNTERS, 0)
    by_sample: dict[str, list[str]] = {name: [] for name in FAILURE_COUNTERS}

    for record in records:
        if not record.scored:
            continue
        for name in FAILURE_COUNTERS:
            count = int(record.counts.get(name, 0) or 0)
            if count:
                totals[name] += count
                by_sample[name].append(record.sample_id)

    excluded: dict[str, list[str]] = {}
    for record in records:
        if record.scored:
            continue
        excluded.setdefault(record.exclusion_reason or "unknown", []).append(record.sample_id)

    return {
        "scored_sample_count": sum(1 for r in records if r.scored),
        "totals": totals,
        "samples_affected": {k: sorted(v) for k, v in by_sample.items()},
        "excluded_by_reason": {
            k: {"count": len(v), "sample_ids": sorted(v)} for k, v in excluded.items()
        },
    }


def write_per_sample(path: Path, records: list[SampleRecord]) -> list[str]:
    metric_keys: set[str] = set()
    count_keys: set[str] = set()
    for record in records:
        metric_keys.update(record.metrics)
        count_keys.update(record.counts)

    columns = (
        ["sample_id", "base_plan_id", "input_type", "scored", "exclusion_reason"]
        + sorted(metric_keys)
        + sorted(count_keys)
    )
    rows = [
        {
            "sample_id": r.sample_id,
            "base_plan_id": r.base_plan_id,
            "input_type": r.input_type,
            "scored": int(r.scored),
            "exclusion_reason": r.exclusion_reason,
            **r.metrics,
            **r.counts,
        }
        for r in sorted(records, key=lambda r: r.sample_id)
    ]
    write_csv(path, rows, columns)
    return columns


def write_segmentation_csv(path: Path, aggregate: dict[str, Any]) -> None:
    macro = aggregate.get("macro", {})
    micro = aggregate.get("micro", {})
    boundary = aggregate.get("boundary_f1", {})
    rows = []
    for name, macro_iou in macro.get("per_class_iou", {}).items():
        rows.append(
            {
                "class": name,
                "macro_iou": macro_iou,
                "micro_iou": micro.get("per_class_iou", {}).get(name, float("nan")),
                "boundary_f1": boundary.get(name, float("nan")),
                "contributing_samples": macro.get("contributing_samples", {}).get(name, 0),
            }
        )
    write_csv(
        path,
        rows,
        ["class", "macro_iou", "micro_iou", "boundary_f1", "contributing_samples"],
    )


def write_topology_csv(path: Path, aggregate: dict[str, Any]) -> None:
    rows = [
        {"metric": key, "value": value, "denominator": aggregate.get("_denominator", 0)}
        for key, value in sorted(aggregate.items())
        if not key.startswith("_")
    ]
    write_csv(path, rows, ["metric", "value", "denominator"])


def write_baseline_comparison(
    path: Path,
    current: dict[str, float],
    baseline: dict[str, float],
    unavailable: tuple[str, ...],
) -> None:
    """Aligned metric names; unsupported baseline metrics stay 'unavailable'."""
    rows = []
    for metric in sorted(set(current) | set(baseline)):
        baseline_value: Any
        if metric in unavailable or metric not in baseline:
            baseline_value = "unavailable"
            delta: Any = ""
            note = "raw Raster-to-Graph produces no openings or circulation"
        else:
            baseline_value = baseline[metric]
            current_value = current.get(metric, float("nan"))
            delta = (
                current_value - baseline_value
                if not (math.isnan(current_value) or math.isnan(baseline_value))
                else ""
            )
            note = ""
        rows.append(
            {
                "metric": metric,
                "baseline_raw_r2g": baseline_value,
                "current_phase4": current.get(metric, float("nan")),
                "delta": delta,
                "note": note,
            }
        )
    write_csv(
        path,
        rows,
        ["metric", "baseline_raw_r2g", "current_phase4", "delta", "note"],
    )
