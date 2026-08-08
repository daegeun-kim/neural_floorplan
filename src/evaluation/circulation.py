"""Space partition, adjacency, and reachability metrics (spec_v006 §4.4). Primary.

Spaces are derived from the reference graph and from the predicted graph by the
*same* code, so the comparison is fair. Pixel overlap is used only to establish
which predicted space corresponds to which reference space; the scored outcome
is connectivity logic, not room-boundary precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from src.evaluation.matching import match_by_cost, prf
from src.evaluation.transforms import CANVAS_SIZE
from src.evaluation.types import Graph, Opening

WALL_STAMP_WIDTH = 3
MIN_SPACE_AREA = 400
OPENING_PROBE_OFFSET = 5.0
MIN_SPACE_IOU = 0.3


@dataclass
class CirculationResult:
    counts: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class SpacePartition:
    """Labelled space regions plus the adjacency implied by openings."""

    labels: np.ndarray  # 0 = wall/unassigned, >=1 = space id
    space_ids: list[int]
    exterior_id: int
    adjacency: set[tuple[int, int]]


def build_partition(
    graph: Graph,
    openings: list[Opening],
    canvas_size: int = CANVAS_SIZE,
) -> SpacePartition:
    """Partition the canvas into spaces and connect them through openings.

    Openings are kept CLOSED while stamping walls so rooms become separate
    regions; connectivity is then decided by the openings themselves, not by
    incidental gaps in the wall geometry.
    """
    occupancy = np.zeros((canvas_size, canvas_size), dtype=bool)
    for (x1, y1), (x2, y2) in graph.edge_segments():
        _stamp_segment(occupancy, x1, y1, x2, y2, WALL_STAMP_WIDTH)

    labels, n_labels = ndimage.label(~occupancy)

    keep: list[int] = []
    for label_id in range(1, n_labels + 1):
        if int((labels == label_id).sum()) >= MIN_SPACE_AREA:
            keep.append(label_id)
        else:
            labels[labels == label_id] = 0

    exterior_id = 0
    border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    border_labels = [int(v) for v in np.unique(border) if v != 0]
    if border_labels:
        exterior_id = max(border_labels, key=lambda label_id: int((labels == label_id).sum()))

    adjacency: set[tuple[int, int]] = set()
    for opening in openings:
        pair = _probe_opening(labels, graph, opening, canvas_size)
        if pair is not None:
            adjacency.add(pair)

    return SpacePartition(
        labels=labels, space_ids=keep, exterior_id=exterior_id, adjacency=adjacency
    )


def _stamp_segment(
    occupancy: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
) -> None:
    """Rasterize one wall segment with a square brush of *width* pixels."""
    size = occupancy.shape[0]
    steps = max(int(max(abs(x2 - x1), abs(y2 - y1))) * 2, 1)
    half = max(width // 2, 0)
    for i in range(steps + 1):
        t = i / steps
        cx = int(round(x1 + (x2 - x1) * t))
        cy = int(round(y1 + (y2 - y1) * t))
        lo_x, hi_x = max(cx - half, 0), min(cx + half + 1, size)
        lo_y, hi_y = max(cy - half, 0), min(cy + half + 1, size)
        if lo_x < hi_x and lo_y < hi_y:
            occupancy[lo_y:hi_y, lo_x:hi_x] = True


def _probe_opening(
    labels: np.ndarray,
    graph: Graph,
    opening: Opening,
    canvas_size: int,
) -> tuple[int, int] | None:
    """Sample perpendicular to the host wall to find the two spaces an opening joins."""
    if opening.host_edge is None or opening.host_edge >= len(graph.edges):
        return None

    (x1, y1), (x2, y2) = graph.edge_segments()[opening.host_edge]
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-9:
        return None
    nx, ny = -dy / length, dx / length

    cx, cy = opening.center
    found: list[int] = []
    for sign in (1.0, -1.0):
        # Step outward until a labelled space is reached; the wall stamp itself
        # is unlabelled, so a single fixed probe distance can land inside it.
        for distance in (
            OPENING_PROBE_OFFSET,
            OPENING_PROBE_OFFSET * 2,
            OPENING_PROBE_OFFSET * 3,
        ):
            px = int(round(cx + nx * sign * distance))
            py = int(round(cy + ny * sign * distance))
            if not (0 <= px < canvas_size and 0 <= py < canvas_size):
                break
            label_id = int(labels[py, px])
            if label_id != 0:
                found.append(label_id)
                break

    if len(found) != 2 or found[0] == found[1]:
        return None
    return (min(found), max(found))


def evaluate_circulation(
    predicted: SpacePartition,
    reference: SpacePartition,
) -> CirculationResult:
    """Compare space adjacency and reachability under space correspondence."""
    result = CirculationResult()

    correspondence = _match_spaces(predicted, reference)
    pred_to_ref = dict(correspondence)

    result.counts.update(
        predicted_spaces=len(predicted.space_ids),
        reference_spaces=len(reference.space_ids),
        matched_spaces=len(correspondence),
        unmatched_predicted_spaces=len(predicted.space_ids) - len(correspondence),
        unmatched_reference_spaces=len(reference.space_ids) - len(correspondence),
    )
    result.metrics["space_count_error"] = float(len(predicted.space_ids) - len(reference.space_ids))

    matched_ref = set(pred_to_ref.values())

    pred_adjacency = {
        _ordered(pred_to_ref[a], pred_to_ref[b])
        for a, b in predicted.adjacency
        if a in pred_to_ref and b in pred_to_ref
    }
    ref_adjacency = {
        _ordered(a, b) for a, b in reference.adjacency if a in matched_ref and b in matched_ref
    }

    tp = len(pred_adjacency & ref_adjacency)
    fp = len(pred_adjacency - ref_adjacency)
    fn = len(ref_adjacency - pred_adjacency)
    p, r, f1 = prf(tp, fp, fn)
    result.metrics.update(adjacency_precision=p, adjacency_recall=r, adjacency_f1=f1)
    result.counts.update(adjacency_tp=tp, adjacency_fp=fp, adjacency_fn=fn)

    _add_reachability(result, sorted(matched_ref), pred_adjacency, ref_adjacency)
    return result


def _ordered(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _match_spaces(predicted: SpacePartition, reference: SpacePartition) -> list[tuple[int, int]]:
    """One-to-one space correspondence by maximum pixel IoU (identity only)."""
    if not predicted.space_ids or not reference.space_ids:
        return []

    cost = np.ones((len(predicted.space_ids), len(reference.space_ids)), dtype=float)
    pred_masks = {sid: (predicted.labels == sid) for sid in predicted.space_ids}
    ref_masks = {sid: (reference.labels == sid) for sid in reference.space_ids}

    for i, p_id in enumerate(predicted.space_ids):
        pm = pred_masks[p_id]
        for j, r_id in enumerate(reference.space_ids):
            rm = ref_masks[r_id]
            union = int((pm | rm).sum())
            if union == 0:
                continue
            iou = int((pm & rm).sum()) / union
            cost[i, j] = 1.0 - iou

    match = match_by_cost(cost, tolerance=1.0 - MIN_SPACE_IOU)
    return [(predicted.space_ids[pi], reference.space_ids[ri]) for pi, ri in match.pairs]


def _add_reachability(
    result: CirculationResult,
    ref_space_ids: list[int],
    pred_adjacency: set[tuple[int, int]],
    ref_adjacency: set[tuple[int, int]],
) -> None:
    """Reachable-pair agreement over the transitive closure of each graph."""
    pred_reach = _reachable_pairs(ref_space_ids, pred_adjacency)
    ref_reach = _reachable_pairs(ref_space_ids, ref_adjacency)

    total = len(ref_space_ids) * (len(ref_space_ids) - 1) // 2
    if total == 0:
        result.metrics["reachability_agreement"] = float("nan")
        result.counts.update(false_connections=0, false_separations=0, disconnected_circulation=0)
        return

    false_connections = len(pred_reach - ref_reach)
    false_separations = len(ref_reach - pred_reach)
    agree = total - false_connections - false_separations

    result.metrics["reachability_agreement"] = agree / total
    result.counts.update(
        false_connections=false_connections,
        false_separations=false_separations,
        disconnected_circulation=false_separations,
        reachable_pairs_reference=len(ref_reach),
        reachable_pairs_predicted=len(pred_reach),
    )


def _reachable_pairs(space_ids: list[int], adjacency: set[tuple[int, int]]) -> set[tuple[int, int]]:
    index = {sid: i for i, sid in enumerate(space_ids)}
    parent = list(range(len(space_ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in adjacency:
        if a not in index or b not in index:
            continue
        ra, rb = find(index[a]), find(index[b])
        if ra != rb:
            parent[ra] = rb

    pairs: set[tuple[int, int]] = set()
    for i in range(len(space_ids)):
        for j in range(i + 1, len(space_ids)):
            if find(i) == find(j):
                pairs.add((space_ids[i], space_ids[j]))
    return pairs
