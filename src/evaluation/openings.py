"""Door and window metrics: detection, hosting, and wall gaps (spec_v006 §4.3).

Primary. An opening that is detected but attached to the wrong wall, or that
fails to interrupt its wall, is a spatial-logic failure even when its centre
lands in exactly the right place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.evaluation.matching import match_points, prf
from src.evaluation.types import Opening

TOL_OPENING = 12.0
GAP_WIDTH_RATIO_MIN = 0.7
GAP_WIDTH_RATIO_MAX = 1.4

OPENING_TYPES = ("door", "window")


@dataclass
class OpeningResult:
    counts: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def evaluate_openings(
    predicted: list[Opening],
    reference: list[Opening],
    edge_match_pairs: list[tuple[int, int]],
    tolerance: float = TOL_OPENING,
) -> OpeningResult:
    """Score detection, type agreement, host correctness, and gap correctness.

    ``edge_match_pairs`` is the (predicted edge, reference edge) correspondence
    from the topology layer; host correctness is defined against it.
    """
    result = OpeningResult()
    pred_edge_to_ref = dict(edge_match_pairs)

    matched_pairs: list[tuple[Opening, Opening]] = []

    for opening_type in OPENING_TYPES:
        preds = [o for o in predicted if o.opening_type == opening_type]
        refs = [o for o in reference if o.opening_type == opening_type]
        match = match_points([o.center for o in preds], [o.center for o in refs], tolerance)
        p, r, f1 = prf(match.tp, match.fp, match.fn)
        result.metrics[f"{opening_type}_precision"] = p
        result.metrics[f"{opening_type}_recall"] = r
        result.metrics[f"{opening_type}_f1"] = f1
        result.counts[f"{opening_type}_tp"] = match.tp
        result.counts[f"spurious_{opening_type}s"] = match.fp
        result.counts[f"missing_{opening_type}s"] = match.fn

        for pi, ri in match.pairs:
            matched_pairs.append((preds[pi], refs[ri]))

    _add_mistyped(result, predicted, reference, tolerance)
    _add_host_and_gap(result, matched_pairs, pred_edge_to_ref)
    return result


def _add_mistyped(
    result: OpeningResult,
    predicted: list[Opening],
    reference: list[Opening],
    tolerance: float,
) -> None:
    """Openings that match by position but disagree on type (spec_v006 §4.3).

    Counted only among openings the typed pass already failed to match, so a
    correctly typed detection is never also reported as a type confusion.
    """
    unmatched_pred: list[Opening] = []
    unmatched_ref: list[Opening] = []
    for opening_type in OPENING_TYPES:
        preds = [o for o in predicted if o.opening_type == opening_type]
        refs = [o for o in reference if o.opening_type == opening_type]
        match = match_points([o.center for o in preds], [o.center for o in refs], tolerance)
        unmatched_pred.extend(preds[i] for i in match.unmatched_pred)
        unmatched_ref.extend(refs[i] for i in match.unmatched_ref)

    cross = match_points(
        [o.center for o in unmatched_pred],
        [o.center for o in unmatched_ref],
        tolerance,
    )
    mistyped = sum(
        1
        for pi, ri in cross.pairs
        if unmatched_pred[pi].opening_type != unmatched_ref[ri].opening_type
    )
    result.counts["mistyped_openings"] = mistyped


def _add_host_and_gap(
    result: OpeningResult,
    matched_pairs: list[tuple[Opening, Opening]],
    pred_edge_to_ref: dict[int, int],
) -> None:
    matched = len(matched_pairs)
    correctly_hosted = 0
    unhosted = 0
    gap_correct = 0
    duplicate_gaps = 0

    for pred, ref in matched_pairs:
        if pred.host_edge is None:
            unhosted += 1
            continue
        mapped_ref_edge = pred_edge_to_ref.get(pred.host_edge)
        if mapped_ref_edge is None or mapped_ref_edge != ref.host_edge:
            continue
        correctly_hosted += 1

        if pred.gap_count > 1:
            duplicate_gaps += 1
            continue
        if pred.gap_count != 1:
            continue
        if ref.width <= 1e-9:
            continue
        ratio = pred.gap_width / ref.width
        if GAP_WIDTH_RATIO_MIN <= ratio <= GAP_WIDTH_RATIO_MAX:
            gap_correct += 1

    result.counts.update(
        matched_openings=matched,
        correctly_hosted_openings=correctly_hosted,
        unhosted_openings=unhosted,
        duplicate_gaps=duplicate_gaps,
        gap_correct_openings=gap_correct,
    )
    result.metrics["host_accuracy"] = correctly_hosted / matched if matched else float("nan")
    result.metrics["gap_accuracy"] = (
        gap_correct / correctly_hosted if correctly_hosted else float("nan")
    )
