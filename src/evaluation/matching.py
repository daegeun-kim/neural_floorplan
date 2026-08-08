"""One-to-one matching primitives shared by every spatial-logic metric.

spec_v006 §4: matching is optimal bipartite assignment on a stated cost,
restricted to pairs within tolerance. Centralizing it here is what keeps the
node, opening, and space metrics comparable — and is what lets the raw
Raster-to-Graph baseline reuse the identical matcher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchResult:
    """Outcome of one bipartite match."""

    pairs: list[tuple[int, int]]  # (prediction index, reference index)
    unmatched_pred: list[int]
    unmatched_ref: list[int]

    @property
    def tp(self) -> int:
        return len(self.pairs)

    @property
    def fp(self) -> int:
        return len(self.unmatched_pred)

    @property
    def fn(self) -> int:
        return len(self.unmatched_ref)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1. Empty-on-both-sides is a perfect score of 1.0."""
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def match_points(
    predicted: list[tuple[float, float]],
    reference: list[tuple[float, float]],
    tolerance: float,
) -> MatchResult:
    """Match two point sets by minimum total Euclidean distance within *tolerance*."""
    return match_by_cost(_distance_matrix(predicted, reference), tolerance)


def match_by_cost(cost: np.ndarray, tolerance: float) -> MatchResult:
    """Optimal assignment on an arbitrary cost matrix, dropping over-tolerance pairs.

    Infeasible pairs are given a large finite cost rather than infinity so the
    solver always succeeds; they are then filtered out by the tolerance test.
    """
    n_pred, n_ref = cost.shape
    if n_pred == 0 or n_ref == 0:
        return MatchResult([], list(range(n_pred)), list(range(n_ref)))

    big = float(tolerance) * 1000.0 + 1.0
    feasible = np.where(cost <= tolerance, cost, big)
    rows, cols = linear_sum_assignment(feasible)

    pairs: list[tuple[int, int]] = []
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        if cost[r, c] <= tolerance:
            pairs.append((r, c))

    matched_pred = {p for p, _ in pairs}
    matched_ref = {r for _, r in pairs}
    return MatchResult(
        pairs=sorted(pairs),
        unmatched_pred=[i for i in range(n_pred) if i not in matched_pred],
        unmatched_ref=[i for i in range(n_ref) if i not in matched_ref],
    )


def _distance_matrix(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=float)
    pa = np.asarray(a, dtype=float)
    pb = np.asarray(b, dtype=float)
    diff = pa[:, None, :] - pb[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def segment_overlap_ratio(
    pred: tuple[tuple[float, float], tuple[float, float]],
    ref: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    """Return (perpendicular offset, overlap ratio) of *pred* against *ref*.

    The overlap ratio is the length of the prediction's projection onto the
    reference segment, divided by the reference length. This is the second
    spec_v006 §4.2 view: a correct wall split at a different junction should not
    score as a total miss.
    """
    (rx1, ry1), (rx2, ry2) = ref
    dx, dy = rx2 - rx1, ry2 - ry1
    ref_len = math.hypot(dx, dy)
    if ref_len < 1e-9:
        return math.inf, 0.0

    ux, uy = dx / ref_len, dy / ref_len

    ts: list[float] = []
    perps: list[float] = []
    for px, py in pred:
        vx, vy = px - rx1, py - ry1
        ts.append(vx * ux + vy * uy)
        perps.append(abs(vx * -uy + vy * ux))

    lo = max(0.0, min(ts))
    hi = min(ref_len, max(ts))
    overlap = max(0.0, hi - lo)
    return max(perps), overlap / ref_len
