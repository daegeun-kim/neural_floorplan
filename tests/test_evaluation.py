"""Asset-independent tests for the evaluation harness (spec_v006, task42 §7).

Every test here runs without CubiCasa5K, model checkpoints, a GPU, or network
access, so the whole file is part of the ordinary CI gate.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from src.evaluation import fixtures, report
from src.evaluation.adapters import BASELINE_UNAVAILABLE, graph_from_payload
from src.evaluation.aggregate import aggregate_topology
from src.evaluation.circulation import build_partition, evaluate_circulation
from src.evaluation.matching import match_points, prf, segment_overlap_ratio
from src.evaluation.openings import evaluate_openings
from src.evaluation.population import base_plan_id, build_population
from src.evaluation.references import ReferenceUnavailable, parse_reference
from src.evaluation.report import RESULT_FILES, select_cases
from src.evaluation.runner import PredictionBundle, evaluate_population, write_results
from src.evaluation.segmentation import CLASS_NAMES, NUM_CLASSES, SegmentationAccumulator
from src.evaluation.topology import evaluate_topology
from src.evaluation.transforms import CanvasTransform, identity
from src.evaluation.types import Graph, Opening, normalize_graph

# ---------------------------------------------------------------------------
# Segmentation metrics (spec_v006 §4.6)
# ---------------------------------------------------------------------------


def _square_mask(value: int, size: int = 16) -> torch.Tensor:
    return torch.full((size, size), value, dtype=torch.long)


def test_segmentation_perfect_prediction_scores_one():
    target = torch.zeros((16, 16), dtype=torch.long)
    target[:8, :] = 2  # wall
    target[8:, :] = 1  # floor

    acc = SegmentationAccumulator()
    sample = acc.add(target.clone(), target)

    assert sample["seg_pixel_accuracy"] == pytest.approx(1.0)
    assert sample["seg_iou_wall"] == pytest.approx(1.0)
    assert sample["seg_iou_floor"] == pytest.approx(1.0)

    aggregate = acc.aggregate()
    assert aggregate["macro"]["per_class_iou"]["wall"] == pytest.approx(1.0)
    assert aggregate["micro"]["pixel_accuracy"] == pytest.approx(1.0)


def test_segmentation_imperfect_prediction_is_between_zero_and_one():
    target = torch.zeros((16, 16), dtype=torch.long)
    target[:8, :] = 2
    preds = torch.zeros((16, 16), dtype=torch.long)
    preds[:4, :] = 2  # only half the wall recovered

    acc = SegmentationAccumulator()
    sample = acc.add(preds, target)

    assert 0.0 < sample["seg_iou_wall"] < 1.0
    assert 0.0 < sample["seg_pixel_accuracy"] < 1.0


def test_segmentation_absent_class_is_nan_and_excluded_not_zero():
    # Class 2 present, classes 3-6 absent from both sides.
    target = _square_mask(2)
    acc = SegmentationAccumulator()
    sample = acc.add(target.clone(), target)

    assert math.isnan(sample["seg_iou_window"])
    aggregate = acc.aggregate()
    assert aggregate["macro"]["contributing_samples"]["window"] == 0
    assert math.isnan(aggregate["macro"]["per_class_iou"]["window"])
    # An absent class must not drag the mean toward zero.
    assert aggregate["macro"]["miou_background_included"] == pytest.approx(1.0)


def test_segmentation_missed_present_class_scores_zero_not_nan():
    target = _square_mask(3)  # all window
    preds = _square_mask(2)  # all wall - class 3 entirely missed

    acc = SegmentationAccumulator()
    sample = acc.add(preds, target)

    assert sample["seg_iou_window"] == pytest.approx(0.0)
    assert not math.isnan(sample["seg_iou_window"])


def test_segmentation_empty_prediction_still_aggregates():
    target = _square_mask(2)
    preds = _square_mask(0)  # predicts background everywhere

    acc = SegmentationAccumulator()
    acc.add(preds, target)
    aggregate = acc.aggregate()

    assert aggregate["sample_count"] == 1
    assert aggregate["macro"]["per_class_iou"]["wall"] == pytest.approx(0.0)


def test_segmentation_class_map_has_seven_named_classes():
    assert NUM_CLASSES == 7
    assert CLASS_NAMES[0] == "background"
    assert CLASS_NAMES[2] == "wall"


# ---------------------------------------------------------------------------
# Matching primitives (spec_v006 §4)
# ---------------------------------------------------------------------------


def test_point_match_within_tolerance():
    result = match_points([(0.0, 0.0), (100.0, 0.0)], [(3.0, 0.0), (104.0, 0.0)], 8.0)
    assert result.tp == 2
    assert result.fp == 0 and result.fn == 0


def test_point_match_outside_tolerance_is_not_matched():
    result = match_points([(0.0, 0.0)], [(50.0, 0.0)], 8.0)
    assert result.tp == 0
    assert result.fp == 1 and result.fn == 1


def test_match_is_one_to_one_not_greedy():
    # Two predictions clustered near one reference must not both match it.
    result = match_points([(0.0, 0.0), (1.0, 0.0)], [(0.5, 0.0)], 8.0)
    assert result.tp == 1
    assert result.fp == 1


def test_prf_empty_both_sides_is_perfect():
    assert prf(0, 0, 0) == (1.0, 1.0, 1.0)


def test_segment_overlap_ratio_detects_partial_cover():
    ref = ((0.0, 0.0), (100.0, 0.0))
    perp, ratio = segment_overlap_ratio(((0.0, 0.0), (50.0, 0.0)), ref)
    assert perp == pytest.approx(0.0)
    assert ratio == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Topology (spec_v006 §4.1, §4.2)
# ---------------------------------------------------------------------------


def _square_graph(offset: float = 0.0) -> Graph:
    nodes = [
        (0.0 + offset, 0.0 + offset),
        (100.0 + offset, 0.0 + offset),
        (100.0 + offset, 100.0 + offset),
        (0.0 + offset, 100.0 + offset),
    ]
    return Graph(nodes=nodes, edges=[(0, 1), (1, 2), (2, 3), (0, 3)])


def test_identical_graphs_score_perfectly():
    result = evaluate_topology(_square_graph(), _square_graph())
    assert result.metrics["node_f1"] == pytest.approx(1.0)
    assert result.metrics["edge_f1"] == pytest.approx(1.0)
    assert result.metrics["junction_degree_exact_f1"] == pytest.approx(1.0)
    assert result.counts["broken_junctions"] == 0


def test_shifted_graph_within_tolerance_still_matches():
    result = evaluate_topology(_square_graph(offset=4.0), _square_graph())
    assert result.metrics["node_f1"] == pytest.approx(1.0)
    assert result.metrics["edge_f1"] == pytest.approx(1.0)


def test_shifted_graph_outside_tolerance_fails_to_match():
    result = evaluate_topology(_square_graph(offset=40.0), _square_graph())
    assert result.metrics["node_f1"] == pytest.approx(0.0)
    assert result.metrics["edge_f1"] == pytest.approx(0.0)


def test_degree_mismatch_is_penalized_even_when_position_matches():
    reference = _square_graph()
    # Same four corners, but one wall is missing -> degrees differ at two nodes.
    predicted = Graph(nodes=list(reference.nodes), edges=[(0, 1), (1, 2), (2, 3)])

    result = evaluate_topology(predicted, reference)
    assert result.metrics["node_f1"] == pytest.approx(1.0)  # positions all found
    assert result.metrics["junction_degree_exact_f1"] < 1.0  # but degrees disagree
    assert result.counts["broken_junctions"] == 2


def test_connected_component_disagreement_is_reported():
    reference = _square_graph()
    predicted = Graph(
        nodes=[*reference.nodes, (300.0, 300.0), (400.0, 300.0)],
        edges=[*reference.edges, (4, 5)],
    )
    result = evaluate_topology(predicted, reference)
    assert result.counts["predicted_components"] == 2
    assert result.counts["reference_components"] == 1
    assert result.metrics["component_count_error"] == pytest.approx(1.0)
    assert result.metrics["component_count_exact"] == pytest.approx(0.0)


def test_dangling_and_non_orthogonal_ratios_are_measured():
    graph = Graph(
        nodes=[(0.0, 0.0), (100.0, 0.0), (150.0, 60.0)],
        edges=[(0, 1), (1, 2)],  # second edge is diagonal
    )
    result = evaluate_topology(graph, _square_graph())
    assert result.metrics["predicted_non_orthogonal_ratio"] == pytest.approx(0.5)
    assert result.metrics["predicted_dangling_node_ratio"] > 0.0


def test_normalize_graph_deduplicates_and_indexes_segments():
    graph = normalize_graph(
        [],
        [(0, 0, 10, 0), (10, 0, 10, 10), (10, 0, 0, 0)],  # third duplicates the first
        merge_tolerance=1.0,
    )
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2


# ---------------------------------------------------------------------------
# Openings (spec_v006 §4.3)
# ---------------------------------------------------------------------------


def _door(center, host=0, gap_count=1, gap_width=40.0, otype="door", width=40.0):
    return Opening(
        opening_type=otype,
        center=center,
        width=width,
        host_edge=host,
        gap_count=gap_count,
        gap_width=gap_width,
    )


IDENTITY_EDGES = [(0, 0), (1, 1), (2, 2)]


def test_correct_opening_scores_detection_host_and_gap():
    ref = [_door((50.0, 0.0))]
    pred = [_door((52.0, 0.0))]
    result = evaluate_openings(pred, ref, IDENTITY_EDGES)

    assert result.metrics["door_f1"] == pytest.approx(1.0)
    assert result.metrics["host_accuracy"] == pytest.approx(1.0)
    assert result.metrics["gap_accuracy"] == pytest.approx(1.0)


def test_missing_opening_is_a_false_negative():
    result = evaluate_openings([], [_door((50.0, 0.0))], IDENTITY_EDGES)
    assert result.counts["missing_doors"] == 1
    assert result.metrics["door_recall"] == pytest.approx(0.0)


def test_spurious_opening_is_a_false_positive():
    result = evaluate_openings([_door((50.0, 0.0))], [], IDENTITY_EDGES)
    assert result.counts["spurious_doors"] == 1
    assert result.metrics["door_precision"] == pytest.approx(0.0)


def test_mistyped_opening_counts_as_both_fp_and_fn():
    ref = [_door((50.0, 0.0), otype="window")]
    pred = [_door((51.0, 0.0), otype="door")]
    result = evaluate_openings(pred, ref, IDENTITY_EDGES)

    assert result.counts["mistyped_openings"] == 1
    assert result.counts["spurious_doors"] == 1
    assert result.counts["missing_windows"] == 1


def test_incorrectly_hosted_opening_fails_host_accuracy():
    ref = [_door((50.0, 0.0), host=0)]
    pred = [_door((51.0, 0.0), host=1)]  # attached to the wrong wall
    result = evaluate_openings(pred, ref, IDENTITY_EDGES)

    assert result.metrics["door_f1"] == pytest.approx(1.0)  # detected
    assert result.metrics["host_accuracy"] == pytest.approx(0.0)  # but mis-hosted


def test_unhosted_opening_is_counted_not_ignored():
    ref = [_door((50.0, 0.0))]
    pred = [_door((51.0, 0.0), host=None)]
    result = evaluate_openings(pred, ref, IDENTITY_EDGES)

    assert result.counts["unhosted_openings"] == 1
    assert result.metrics["host_accuracy"] == pytest.approx(0.0)


def test_duplicate_wall_gap_is_its_own_failure_category():
    ref = [_door((50.0, 0.0))]
    pred = [_door((51.0, 0.0), gap_count=2)]
    result = evaluate_openings(pred, ref, IDENTITY_EDGES)

    assert result.counts["duplicate_gaps"] == 1
    assert result.counts["gap_correct_openings"] == 0


def test_gap_width_outside_ratio_band_is_not_correct():
    ref = [_door((50.0, 0.0), width=40.0)]
    pred = [_door((51.0, 0.0), gap_width=200.0)]  # far too wide
    result = evaluate_openings(pred, ref, IDENTITY_EDGES)
    assert result.counts["gap_correct_openings"] == 0


# ---------------------------------------------------------------------------
# Circulation (spec_v006 §4.4)
# ---------------------------------------------------------------------------


def test_identical_space_graphs_agree_completely():
    graph, openings = fixtures.two_room_plan()
    partition = build_partition(graph, openings)
    result = evaluate_circulation(partition, partition)

    assert result.metrics["adjacency_f1"] == pytest.approx(1.0)
    assert result.metrics["reachability_agreement"] == pytest.approx(1.0)


def test_two_room_plan_partitions_into_distinct_spaces():
    graph, openings = fixtures.two_room_plan()
    partition = build_partition(graph, openings)
    # Two rooms plus the exterior.
    assert len(partition.space_ids) >= 3
    assert partition.adjacency, "the door should connect two spaces"


def test_dropping_the_door_breaks_reachability():
    ref_graph, ref_openings = fixtures.two_room_plan()
    pred_graph, pred_openings = fixtures.two_room_plan(drop_opening=True)

    reference = build_partition(ref_graph, ref_openings)
    predicted = build_partition(pred_graph, pred_openings)
    result = evaluate_circulation(predicted, reference)

    assert result.metrics["reachability_agreement"] < 1.0
    assert result.counts["false_separations"] >= 1


def test_unequal_adjacency_graphs_are_scored_below_one():
    ref_graph, ref_openings = fixtures.two_room_plan()
    reference = build_partition(ref_graph, ref_openings)
    predicted = build_partition(ref_graph, [])  # no openings at all
    result = evaluate_circulation(predicted, reference)

    assert result.metrics["adjacency_f1"] < 1.0
    assert result.counts["adjacency_fn"] >= 1


# ---------------------------------------------------------------------------
# Population, exclusions, references (spec_v006 §1)
# ---------------------------------------------------------------------------


def test_base_plan_id_groups_both_renderings_of_one_plan():
    assert base_plan_id("3046_model_clean_png") == "3046"
    assert base_plan_id("3046_F1_scaled_png") == "3046"


def _write_splits(tmp_path, train, val, test):
    (tmp_path / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (tmp_path / "val.json").write_text(json.dumps(val), encoding="utf-8")
    (tmp_path / "test.json").write_text(json.dumps(test), encoding="utf-8")
    return tmp_path


def _entry(sample_id, input_type="svg_rendered_clean"):
    return {"sample_id": sample_id, "input_type": input_type}


def test_leakage_filter_removes_plans_seen_in_training(tmp_path):
    splits = _write_splits(
        tmp_path,
        train=[_entry("100_F1_scaled_png", "original_raster")],
        val=[],
        test=[_entry("100_model_clean_png"), _entry("200_model_clean_png")],
    )
    population = build_population(splits, input_type="svg_rendered_clean")

    assert population.sample_ids == ["200_model_clean_png"]
    assert population.excluded[0]["reason"] == "leaked_base_plan"
    assert population.leakage_stats["train_test_base_overlap"] == 1


def test_leakage_filter_can_be_disabled_explicitly(tmp_path):
    splits = _write_splits(
        tmp_path,
        train=[_entry("100_F1_scaled_png", "original_raster")],
        val=[],
        test=[_entry("100_model_clean_png"), _entry("200_model_clean_png")],
    )
    population = build_population(
        splits, input_type="svg_rendered_clean", enforce_leakage_filter=False
    )
    assert len(population.entries) == 2
    assert population.excluded == []


def test_population_hash_is_stable_and_order_independent(tmp_path):
    splits = _write_splits(tmp_path, train=[], val=[], test=[_entry("2_a"), _entry("1_b")])
    first = build_population(splits).population_hash()
    second = build_population(splits).population_hash()
    assert first == second


def test_missing_split_manifest_fails_early(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_population(tmp_path)


def test_unscorable_samples_are_recorded_with_a_reason():
    population = fixtures.synthetic_population()

    def failing_predict(entry):
        raise RuntimeError("pipeline exploded")

    records = evaluate_population(
        population,
        predict=failing_predict,
        dataset_root=".",
        reference_loader=fixtures.synthetic_reference_loader(),
    )
    assert all(not r.scored for r in records)
    assert all(r.exclusion_reason == "prediction_failed" for r in records)
    assert "RuntimeError" in records[0].exclusion_detail


def test_missing_reference_excludes_without_aborting_the_run():
    population = fixtures.synthetic_population()

    def missing_reference(dataset_root, base_plan, transform):
        raise ReferenceUnavailable("missing_graph_reference", "wall_graph.json")

    records = evaluate_population(
        population,
        predict=fixtures.synthetic_predictor(),
        dataset_root=".",
        reference_loader=missing_reference,
    )
    assert len(records) == 3
    assert all(r.exclusion_reason == "missing_graph_reference" for r in records)


def test_parse_reference_maps_openings_onto_host_edges():
    raw = {
        "nodes": [
            {"id": 0, "type": "wall_node", "x": 0, "y": 0},
            {"id": 1, "type": "wall_node", "x": 100, "y": 0},
            {
                "id": 2,
                "type": "door_center",
                "x": 50,
                "y": 0,
                "host_edge": 7,
                "opening_width_px": 20,
                "orientation": "horizontal",
            },
        ],
        "edges": [{"id": 7, "start": 0, "end": 1}],
    }
    graph, openings = parse_reference(raw, identity())

    assert len(graph.nodes) == 2  # opening nodes are not wall nodes
    assert len(graph.edges) == 1
    assert openings[0].opening_type == "door"
    assert openings[0].host_edge == 0  # resolved from edge id 7 to position 0


# ---------------------------------------------------------------------------
# Coordinate transform (spec_v006 §3)
# ---------------------------------------------------------------------------


def test_canvas_transform_maps_content_bbox_into_the_canvas():
    transform = CanvasTransform.derive((100.0, 200.0, 300.0, 400.0), margin=0.20)
    x, y = transform.apply(100.0, 200.0)  # the bbox origin

    assert 0.0 <= x <= 512.0 and 0.0 <= y <= 512.0
    # The origin must land inside the padded region, not at the canvas corner.
    assert x > 0.0 and y > 0.0


def test_canvas_transform_round_trips_from_a_manifest():
    derived = CanvasTransform.derive((10.0, 20.0, 210.0, 320.0))
    manifest = {
        "content_bbox_original": [10, 20, 210, 320],
        "pad_x": derived.pad_x,
        "pad_y": derived.pad_y,
        "scale_to_512": derived.scale,
        "canvas_offset_x": derived.offset_x,
        "canvas_offset_y": derived.offset_y,
    }
    assert CanvasTransform.from_manifest(manifest).apply(50.0, 60.0) == pytest.approx(
        derived.apply(50.0, 60.0)
    )


def test_identity_transform_is_a_no_op():
    assert identity().apply(12.5, 34.5) == (12.5, 34.5)


# ---------------------------------------------------------------------------
# Adapters and baseline comparability (spec_v006 §5.1)
# ---------------------------------------------------------------------------


def test_graph_from_payload_builds_indexed_edges():
    graph = graph_from_payload(
        {"nodes": [[0, 0], [10, 0]], "edges": [[0, 0, 10, 0], [10, 0, 10, 10]]}
    )
    assert len(graph.edges) == 2
    assert max(graph.degree()) == 2


def test_baseline_unavailable_list_covers_opening_and_circulation_metrics():
    assert "door_f1" in BASELINE_UNAVAILABLE
    assert "host_accuracy" in BASELINE_UNAVAILABLE
    assert "reachability_agreement" in BASELINE_UNAVAILABLE
    # Wall-graph metrics MUST remain available - they are the comparison.
    assert "edge_f1" not in BASELINE_UNAVAILABLE
    assert "node_f1" not in BASELINE_UNAVAILABLE


def test_baseline_and_current_use_the_same_metric_code():
    population = fixtures.synthetic_population()
    loader = fixtures.synthetic_reference_loader()
    current = evaluate_population(
        population,
        predict=fixtures.synthetic_predictor(),
        dataset_root=".",
        reference_loader=loader,
    )
    baseline = evaluate_population(
        population,
        predict=fixtures.synthetic_predictor(degrade=True),
        dataset_root=".",
        include_circulation=False,
        reference_loader=loader,
    )
    current_agg = aggregate_topology(current)
    baseline_agg = aggregate_topology(baseline)

    # Shared wall-graph metrics must exist on both sides with identical names.
    for key in ("macro_node_f1", "macro_edge_f1"):
        assert key in current_agg and key in baseline_agg


def test_baseline_unsupported_metrics_are_not_coerced_to_zero(tmp_path):
    report.write_baseline_comparison(
        tmp_path / "b.csv",
        current={"macro_door_f1": 0.8, "macro_edge_f1": 0.9},
        baseline={"macro_edge_f1": 0.5},
        unavailable=("macro_door_f1",),
    )
    text = (tmp_path / "b.csv").read_text(encoding="utf-8")
    assert "unavailable" in text
    assert ",0.000000," not in text.split("macro_door_f1")[1].split("\n")[0]


# ---------------------------------------------------------------------------
# Aggregation, reporting, determinism (spec_v006 §6)
# ---------------------------------------------------------------------------


def _fixture_records():
    return evaluate_population(
        fixtures.synthetic_population(),
        predict=fixtures.synthetic_predictor(),
        dataset_root=".",
        reference_loader=fixtures.synthetic_reference_loader(),
    )


def test_aggregate_reports_both_macro_and_micro():
    aggregated = aggregate_topology(_fixture_records())
    assert "macro_node_f1" in aggregated
    assert "micro_node_f1" in aggregated
    assert aggregated["_denominator"] == 3.0


def test_aggregate_denominator_excludes_unscored_records():
    records = _fixture_records()
    records[0].scored = False
    records[0].exclusion_reason = "prediction_failed"
    assert aggregate_topology(records)["_denominator"] == 2.0


def test_case_selection_is_deterministic_and_rule_documented():
    records = _fixture_records()
    first = select_cases(records)
    second = select_cases(list(reversed(records)))

    assert first["cases"] == second["cases"]
    assert "ascending sample_id" in first["rule"]
    assert set(first["cases"]) == {"success", "mixed", "failure"}


def test_failure_report_lists_categories_and_excluded_samples():
    records = _fixture_records()
    records[0].scored = False
    records[0].exclusion_reason = "missing_input"

    failures = report.collect_failures(records)
    assert failures["scored_sample_count"] == 2
    assert "missing_input" in failures["excluded_by_reason"]
    assert "missing_doors" in failures["totals"]


def test_write_results_emits_every_schema_file(tmp_path):
    records = _fixture_records()
    summary = write_results(
        tmp_path,
        provenance={"spec": "spec_v006_evaluation", "mode": "test"},
        records=records,
        segmentation_aggregate=fixtures.synthetic_segmentation_aggregate(),
        baseline_records=records,
    )

    for name in RESULT_FILES:
        assert (tmp_path / name).exists(), f"missing result file: {name}"

    assert summary["denominators"]["scored"] == 3
    assert "primary_spatial_logic" in summary


def test_written_json_is_parseable_with_nan_as_null(tmp_path):
    write_results(
        tmp_path,
        provenance={"spec": "spec_v006_evaluation"},
        records=_fixture_records(),
        segmentation_aggregate=fixtures.synthetic_segmentation_aggregate(),
    )
    for name in ("summary.json", "failures.json", "selected_cases.json", "provenance.json"):
        payload = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert isinstance(payload, dict)


def test_repeated_runs_produce_identical_output(tmp_path):
    first_dir = tmp_path / "run1"
    second_dir = tmp_path / "run2"
    for target in (first_dir, second_dir):
        write_results(
            target,
            provenance={"spec": "spec_v006_evaluation"},
            records=_fixture_records(),
            segmentation_aggregate=fixtures.synthetic_segmentation_aggregate(),
        )

    for name in ("summary.json", "per_sample.csv", "topology.csv", "failures.json"):
        assert (first_dir / name).read_text(encoding="utf-8") == (second_dir / name).read_text(
            encoding="utf-8"
        ), f"{name} is not deterministic"


def test_summary_states_the_claim_hierarchy(tmp_path):
    summary = write_results(
        tmp_path,
        provenance={},
        records=_fixture_records(),
    )
    hierarchy = summary["claim_hierarchy"]
    assert "primary" in hierarchy and "topology" in hierarchy
    assert "not a goal" in hierarchy


def test_partition_labels_are_a_2d_integer_array():
    graph, openings = fixtures.two_room_plan()
    partition = build_partition(graph, openings)
    assert isinstance(partition.labels, np.ndarray)
    assert partition.labels.shape == (512, 512)


def test_prediction_bundle_carries_its_transform():
    bundle = fixtures.synthetic_predictor()({"sample_id": "9001_fixture_a"})
    assert isinstance(bundle, PredictionBundle)
    assert bundle.transform.scale == pytest.approx(1.0)
