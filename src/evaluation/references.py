"""Load reference wall graphs and openings (spec_v006 §1.5).

The reference is ``<sample>/masks/wall_graph.json``, produced by
``src/generate_wall_graphs.py`` from ``model.svg``. It carries its own
``status`` signal, which is the authoritative reference-validity check.

Reference geometry is in the original raster frame and is mapped forward onto
the 512x512 canvas by the caller-supplied transform, so predictions and
references are compared in one frame.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.transforms import CanvasTransform
from src.evaluation.types import Graph, Opening

GRAPH_FILENAME = "wall_graph.json"
METRICS_FILENAME = "wall_graph_metrics.json"

_OPENING_NODE_TYPES = {"door_center": "door", "window_center": "window"}


class ReferenceUnavailable(RuntimeError):
    """Raised when a sample has no usable reference; carries an exclusion reason."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def reference_paths(dataset_root: Path, base_plan: str) -> tuple[Path, Path]:
    sample_dir = Path(dataset_root) / base_plan
    return sample_dir / "masks" / GRAPH_FILENAME, sample_dir / "masks" / METRICS_FILENAME


def load_reference(
    dataset_root: Path,
    base_plan: str,
    transform: CanvasTransform,
) -> tuple[Graph, list[Opening], dict]:
    """Return (wall graph, openings, raw json) mapped into canvas space."""
    graph_path, metrics_path = reference_paths(dataset_root, base_plan)
    if not graph_path.exists():
        raise ReferenceUnavailable("missing_graph_reference", str(graph_path.name))

    with open(graph_path, encoding="utf-8") as handle:
        raw = json.load(handle)

    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as handle:
            ref_metrics = json.load(handle)
        if ref_metrics.get("status") == "unusable":
            raise ReferenceUnavailable(
                "unusable_graph_reference",
                ",".join(ref_metrics.get("reasons", [])) or "status=unusable",
            )

    return (*parse_reference(raw, transform), raw)


def parse_reference(raw: dict, transform: CanvasTransform) -> tuple[Graph, list[Opening]]:
    """Convert a wall_graph.json payload into evaluation types."""
    wall_nodes: list[dict] = []
    opening_nodes: list[dict] = []
    for node in raw.get("nodes", []):
        if node.get("type") == "wall_node":
            wall_nodes.append(node)
        elif node.get("type") in _OPENING_NODE_TYPES:
            opening_nodes.append(node)

    node_index = {node["id"]: i for i, node in enumerate(wall_nodes)}
    graph = Graph(
        nodes=[transform.apply(float(n["x"]), float(n["y"])) for n in wall_nodes],
        edges=[],
    )

    # Reference edges carry their own ids; openings name a host edge by that id,
    # so the id -> position map must be built here and used for host_edge.
    edge_id_to_index: dict[int, int] = {}
    for edge in raw.get("edges", []):
        start = node_index.get(edge.get("start"))
        end = node_index.get(edge.get("end"))
        if start is None or end is None or start == end:
            continue
        edge_id_to_index[edge.get("id")] = len(graph.edges)
        graph.edges.append((start, end) if start < end else (end, start))

    openings: list[Opening] = []
    for node in opening_nodes:
        openings.append(
            Opening(
                opening_type=_OPENING_NODE_TYPES[node["type"]],
                center=transform.apply(float(node["x"]), float(node["y"])),
                width=transform.apply_length(float(node.get("opening_width_px", 0.0))),
                orientation=node.get("orientation", ""),
                host_edge=edge_id_to_index.get(node.get("host_edge")),
                gap_count=1,
                gap_width=transform.apply_length(float(node.get("opening_width_px", 0.0))),
            )
        )

    return graph, openings
