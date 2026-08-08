"""Shared value types for the evaluation harness (spec_v006).

These are deliberately plain: a graph is nodes plus edge index pairs, an opening
is a typed centre with an optional host edge. Both the reference side and the
prediction side are normalized into these types before any metric runs, so the
same matching code scores both without special cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# spec_v006 §1.6 - the only reasons a sample may leave the denominator.
EXCLUSION_REASONS = (
    "leaked_base_plan",
    "missing_input",
    "missing_semantic_reference",
    "missing_graph_reference",
    "unusable_graph_reference",
    "prediction_failed",
)

Point = tuple[float, float]


@dataclass
class Graph:
    """A wall graph in one coordinate frame.

    ``nodes`` are junction/endpoint coordinates. ``edges`` are index pairs into
    ``nodes``. Storing edges as index pairs (rather than as four floats) is what
    makes degree well defined, which spec_v006 §4.1 requires.
    """

    nodes: list[Point] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)

    def degree(self) -> list[int]:
        deg = [0] * len(self.nodes)
        for a, b in self.edges:
            if a == b:
                continue
            deg[a] += 1
            deg[b] += 1
        return deg

    def edge_segments(self) -> list[tuple[Point, Point]]:
        return [(self.nodes[a], self.nodes[b]) for a, b in self.edges]

    def component_count(self) -> int:
        """Number of connected components over nodes that carry at least one edge."""
        parent = list(range(len(self.nodes)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in self.edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        incident = {i for e in self.edges for i in e}
        return len({find(i) for i in incident})


@dataclass
class Opening:
    """A door or window centre, optionally hosted on a wall edge."""

    opening_type: str  # "door" | "window"
    center: Point
    width: float = 0.0
    orientation: str = ""  # "horizontal" | "vertical" | ""
    host_edge: int | None = None  # index into the owning Graph's edges
    gap_count: int = 0
    gap_width: float = 0.0


@dataclass
class SampleRecord:
    """Everything the report layer needs about one evaluated sample."""

    sample_id: str
    base_plan_id: str
    input_type: str = ""
    scored: bool = False
    exclusion_reason: str = ""
    exclusion_detail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    transform: dict[str, float] = field(default_factory=dict)


def normalize_graph(
    nodes: list[Point],
    segments: list[tuple[float, float, float, float]],
    merge_tolerance: float = 1e-6,
) -> Graph:
    """Build a :class:`Graph` from loose node coordinates and raw segments.

    Segment endpoints are snapped onto existing nodes within *merge_tolerance*;
    endpoints with no match become new nodes. This is what turns the pipeline's
    ``[x1, y1, x2, y2]`` edge lists into an index-pair graph with usable degrees.
    """
    out_nodes: list[Point] = [(float(x), float(y)) for x, y in nodes]

    def index_of(pt: Point) -> int:
        for i, existing in enumerate(out_nodes):
            dx = existing[0] - pt[0]
            dy = existing[1] - pt[1]
            if (dx * dx + dy * dy) <= merge_tolerance * merge_tolerance:
                return i
        out_nodes.append(pt)
        return len(out_nodes) - 1

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x1, y1, x2, y2 in segments:
        a = index_of((float(x1), float(y1)))
        b = index_of((float(x2), float(y2)))
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)

    return Graph(nodes=out_nodes, edges=edges)
