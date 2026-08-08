"""Normalize pipeline output into evaluation types (spec_v006 §4).

Both evaluated paths funnel through here so they are scored by identical code:

    baseline : raw Raster-to-Graph graph, before Phase 4 alignment/recovery
    current  : the full Phase 4 pipeline, with hosted openings and wall gaps

The baseline produces no openings and no circulation, so §4.3-§4.5 are reported
``unavailable`` for it rather than coerced to zero (spec_v006 §5.1).
"""

from __future__ import annotations

from typing import Any

from src.evaluation.types import Graph, Opening, normalize_graph

# Metric families the raw baseline structurally cannot produce.
BASELINE_UNAVAILABLE = (
    "door_precision",
    "door_recall",
    "door_f1",
    "window_precision",
    "window_recall",
    "window_f1",
    "host_accuracy",
    "gap_accuracy",
    "adjacency_precision",
    "adjacency_recall",
    "adjacency_f1",
    "reachability_agreement",
    "space_count_error",
)


def graph_from_payload(payload: dict[str, Any]) -> Graph:
    """Build a Graph from an R2G-style ``{"nodes": [...], "edges": [...]}`` payload.

    Edges are ``[x1, y1, x2, y2]`` segments, so endpoints are re-indexed onto
    nodes. A small snap tolerance is used because merge and alignment can leave
    an endpoint a fraction of a pixel from its node.
    """
    nodes = [
        (float(n[0]), float(n[1]))
        for n in payload.get("nodes", [])
        if isinstance(n, list | tuple) and len(n) >= 2
    ]
    segments = [
        (float(e[0]), float(e[1]), float(e[2]), float(e[3]))
        for e in payload.get("edges", [])
        if isinstance(e, list | tuple) and len(e) >= 4
    ]
    return normalize_graph(nodes, segments, merge_tolerance=1.0)


def openings_from_phase4(
    hosted_doors: list[Any],
    hosted_windows: list[Any],
    graph: Graph,
    opening_gaps: list[dict] | None = None,
) -> list[Opening]:
    """Convert hosted Phase 4 openings into evaluation openings.

    The pipeline reports its host as an index into its own aligned edge list and
    the host edge's raw coordinates. Since :func:`graph_from_payload` re-indexes
    edges, the host is resolved by matching those raw coordinates back onto the
    normalized graph rather than trusting the index to survive.
    """
    segment_lookup = _segment_lookup(graph)
    gap_counts, gap_widths = _gap_stats(opening_gaps or [])

    openings: list[Opening] = []
    for hosted in list(hosted_doors) + list(hosted_windows):
        center = _center_of(hosted)
        if center is None:
            continue
        host_edge = _resolve_host(hosted, segment_lookup)
        source_id = getattr(hosted, "source_component_id", None)
        key = (getattr(hosted, "opening_type", ""), source_id)
        openings.append(
            Opening(
                opening_type=getattr(hosted, "opening_type", ""),
                center=center,
                width=float(getattr(hosted, "width_px", 0.0)),
                host_edge=host_edge,
                gap_count=gap_counts.get(key, 1 if host_edge is not None else 0),
                gap_width=gap_widths.get(key, float(getattr(hosted, "width_px", 0.0))),
            )
        )
    return openings


def _center_of(hosted: Any) -> tuple[float, float] | None:
    points = getattr(hosted, "snapped_points", None) or getattr(hosted, "raw_points", None)
    if not points or len(points) < 2:
        return None
    (x1, y1), (x2, y2) = points[0], points[1]
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def _segment_lookup(graph: Graph) -> dict[tuple[int, int, int, int], int]:
    lookup: dict[tuple[int, int, int, int], int] = {}
    for idx, ((x1, y1), (x2, y2)) in enumerate(graph.edge_segments()):
        lookup.setdefault(_segment_key(x1, y1, x2, y2), idx)
    return lookup


def _segment_key(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    a = (int(round(x1)), int(round(y1)))
    b = (int(round(x2)), int(round(y2)))
    lo, hi = (a, b) if a <= b else (b, a)
    return (*lo, *hi)


def _resolve_host(hosted: Any, lookup: dict[tuple[int, int, int, int], int]) -> int | None:
    raw = getattr(hosted, "host_edge_raw", None)
    if raw and len(raw) >= 4:
        key = _segment_key(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        if key in lookup:
            return lookup[key]
    return None


def _gap_stats(
    opening_gaps: list[dict],
) -> tuple[dict[tuple[str, Any], int], dict[tuple[str, Any], float]]:
    """Count gaps per opening so a doubled gap is distinguishable from a single one."""
    counts: dict[tuple[str, Any], int] = {}
    widths: dict[tuple[str, Any], float] = {}
    for gap in opening_gaps:
        key = (gap.get("opening_type", ""), gap.get("source_component_id"))
        counts[key] = counts.get(key, 0) + 1
        width = gap.get("width_px")
        if width is None:
            t0, t1 = gap.get("t_start"), gap.get("t_end")
            length = gap.get("edge_length_px")
            if t0 is not None and t1 is not None and length:
                width = abs(float(t1) - float(t0)) * float(length)
        if width is not None:
            widths[key] = float(width)
    return counts, widths
