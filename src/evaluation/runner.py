"""Evaluation orchestration (spec_v006 §6, Task 42).

Scores one sample at a time, records every exclusion with a reason, and writes
the compact result schema. Nothing here modifies the model or the vectorizer.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation import adapters, references, report
from src.evaluation.aggregate import aggregate_topology, headline
from src.evaluation.circulation import build_partition, evaluate_circulation
from src.evaluation.openings import evaluate_openings
from src.evaluation.population import Population, base_plan_id
from src.evaluation.topology import evaluate_topology
from src.evaluation.transforms import CanvasTransform
from src.evaluation.types import Graph, Opening, SampleRecord


@dataclass
class PredictionBundle:
    """One sample's prediction, already in canvas space."""

    graph: Graph
    openings: list[Opening]
    transform: CanvasTransform


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:  # noqa: BLE001 - torch is optional for fixture-only runs
        pass


def score_sample(
    predicted_graph: Graph,
    predicted_openings: list[Opening],
    reference_graph: Graph,
    reference_openings: list[Opening],
    include_circulation: bool = True,
) -> tuple[dict[str, float], dict[str, int]]:
    """Score one sample across L2 and L3. Returns (metrics, counts)."""
    topology = evaluate_topology(predicted_graph, reference_graph)
    metrics = dict(topology.metrics)
    counts = dict(topology.counts)

    openings = evaluate_openings(predicted_openings, reference_openings, topology.edge_match_pairs)
    metrics.update(openings.metrics)
    counts.update(openings.counts)

    if include_circulation:
        predicted_partition = build_partition(predicted_graph, predicted_openings)
        reference_partition = build_partition(reference_graph, reference_openings)
        circulation = evaluate_circulation(predicted_partition, reference_partition)
        metrics.update(circulation.metrics)
        counts.update(circulation.counts)

    return metrics, counts


def default_reference_loader(
    dataset_root: Path, base_plan: str, transform: CanvasTransform
) -> tuple[Graph, list[Opening]]:
    graph, openings, _ = references.load_reference(dataset_root, base_plan, transform)
    return graph, openings


def evaluate_population(
    population: Population,
    predict: Any,
    dataset_root: Path,
    include_circulation: bool = True,
    on_progress: Any = None,
    reference_loader: Any = None,
) -> list[SampleRecord]:
    """Run the evaluation over a population.

    Args:
        predict: callable ``(entry) -> PredictionBundle``. Raising
            :class:`references.ReferenceUnavailable` or any other exception
            excludes the sample with a recorded reason rather than aborting.
        reference_loader: callable ``(dataset_root, base_plan, transform) ->
            (Graph, [Opening])``. Injectable so the fixture-only smoke path and
            the unit tests can supply references without CubiCasa5K on disk.
    """
    load_ref = reference_loader or default_reference_loader
    records: list[SampleRecord] = []

    for entry in population.entries:
        sample_id = entry["sample_id"]
        base = base_plan_id(sample_id)
        record = SampleRecord(
            sample_id=sample_id,
            base_plan_id=base,
            input_type=entry.get("input_type", ""),
        )

        try:
            bundle = predict(entry)
        except references.ReferenceUnavailable as exc:
            record.exclusion_reason = exc.reason
            record.exclusion_detail = exc.detail
            records.append(record)
            if on_progress:
                on_progress(record)
            continue
        except Exception as exc:  # noqa: BLE001 - a failed sample is data, not a crash
            record.exclusion_reason = "prediction_failed"
            record.exclusion_detail = f"{type(exc).__name__}: {exc}"
            records.append(record)
            if on_progress:
                on_progress(record)
            continue

        try:
            ref_graph, ref_openings = load_ref(dataset_root, base, bundle.transform)
        except references.ReferenceUnavailable as exc:
            record.exclusion_reason = exc.reason
            record.exclusion_detail = exc.detail
            records.append(record)
            if on_progress:
                on_progress(record)
            continue

        metrics, counts = score_sample(
            bundle.graph,
            bundle.openings,
            ref_graph,
            ref_openings,
            include_circulation=include_circulation,
        )
        record.scored = True
        record.metrics = metrics
        record.counts = counts
        record.transform = bundle.transform.as_dict()
        records.append(record)
        if on_progress:
            on_progress(record)

    return records


def write_results(
    output_dir: Path,
    provenance: dict[str, Any],
    records: list[SampleRecord],
    segmentation_aggregate: dict[str, Any] | None = None,
    baseline_records: list[SampleRecord] | None = None,
) -> dict[str, Any]:
    """Write the full spec_v006 §6.1 result set. Returns the summary payload."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current = aggregate_topology(records)
    report.write_per_sample(output_dir / "per_sample.csv", records)
    report.write_topology_csv(output_dir / "topology.csv", current)

    if segmentation_aggregate is not None:
        report.write_segmentation_csv(output_dir / "segmentation.csv", segmentation_aggregate)

    baseline = aggregate_topology(baseline_records) if baseline_records else {}
    unavailable = tuple(f"macro_{name}" for name in adapters.BASELINE_UNAVAILABLE) + tuple(
        f"micro_{name}" for name in adapters.BASELINE_UNAVAILABLE
    )
    report.write_baseline_comparison(
        output_dir / "baseline_comparison.csv",
        {k: v for k, v in current.items() if not k.startswith("_")},
        {k: v for k, v in baseline.items() if not k.startswith("_")},
        unavailable,
    )

    failures = report.collect_failures(records)
    selected = report.select_cases(records)
    report.write_json(output_dir / "failures.json", failures)
    report.write_json(output_dir / "selected_cases.json", selected)

    scored = sum(1 for r in records if r.scored)
    summary = {
        "spec": "spec_v006_evaluation",
        "denominators": {
            "population_size": len(records),
            "scored": scored,
            "excluded": len(records) - scored,
            "matched_openings": int(current.get("_matched_openings", 0)),
        },
        "primary_spatial_logic": headline(current),
        "secondary_segmentation": segmentation_aggregate or "not evaluated",
        "baseline_available": bool(baseline_records),
        "claim_hierarchy": (
            "primary: wall topology, circulation, and opening logic; "
            "secondary: seven-class segmentation including per-class mIoU; "
            "not a goal: pixel-perfect tracing"
        ),
    }
    report.write_json(output_dir / "summary.json", summary)
    provenance = dict(provenance)
    provenance["result_files"] = sorted(p.name for p in output_dir.iterdir() if p.is_file())
    report.write_json(output_dir / "provenance.json", provenance)
    return summary


def nan_safe(value: float) -> float | None:
    return None if math.isnan(value) else value
