"""Tests for src/generate_wall_graphs.py (spec_v003-1 wall graph generation)."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest
from lxml import etree

from src.generate_semantic_masks import SVG_NAME
from src.generate_wall_graphs import (
    DEBUG_PNG_FILENAME,
    DEBUG_SVG_FILENAME,
    GRAPH_FILENAME,
    METRICS_FILENAME,
    generate_wall_graph,
    svg_to_pixel_transform,
)

# A closed rectangular room: 4 outer walls (band thickness 10) forming a loop,
# one interior partition wall (guarantees a real T-junction so skeletonize
# doesn't hit the no-junction closed-loop edge case), a Window on the right
# wall, and a Door (Threshold + Panel) on the bottom wall.
ROOM_SVG = """\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <g class="Floor">
    <g class="Floorplan Floor-1">
      <g id="space-uuid" class="Space Room" fill="#ffffff" stroke="#ffffff">
        <polygon points="10,10 190,10 190,190 10,190"/>
      </g>
      <g id="Wall" class="Wall Top" fill="#000000" stroke="#000000">
        <polygon points="0,0 200,0 200,10 0,10"/>
      </g>
      <g id="Wall" class="Wall Partition" fill="#000000" stroke="#000000">
        <polygon points="95,10 105,10 105,100 95,100"/>
      </g>
      <g id="Wall" class="Wall Bottom" fill="#000000" stroke="#000000">
        <polygon points="0,190 200,190 200,200 0,200"/>
        <g id="Door" class="Door Swing Beside" fill="#ffffff" stroke="#000000">
          <polygon points="80,190 120,190 120,200 80,200"/>
          <g id="Threshold" class="Threshold">
            <polygon points="80,190 120,190 120,200 80,200"/>
          </g>
          <g id="Panel" class="Panel Left Positive" fill="none">
            <g id="PanelArea" class="PanelArea" fill="none" stroke="none">
              <polygon points="80,190 120,190 120,200 80,200"/>
            </g>
            <path d="M120,200 q-40,0 -40,-40 l40,0 Z"/>
          </g>
        </g>
      </g>
      <g id="Wall" class="Wall Left" fill="#000000" stroke="#000000">
        <polygon points="0,0 10,0 10,200 0,200"/>
      </g>
      <g id="Wall" class="Wall Right" fill="#000000" stroke="#000000">
        <polygon points="190,0 200,0 200,200 190,200"/>
        <g id="Window" class="Window Regular" fill="#f0f0ff" stroke="#000000">
          <polygon points="190,80 200,80 200,120 190,120"/>
          <g id="Glass" class="Glass"><polygon points="190,80 200,80 200,120 190,120"/></g>
          <g id="Panel" class="Panel"><line x1="195" x2="195" y1="80" y2="120"/></g>
        </g>
      </g>
    </g>
  </g>
</svg>
"""

# No Wall elements at all -> empty wall mask -> degenerate/unusable graph.
NO_WALL_SVG = """\
<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" viewBox="0 0 50 50">
  <g class="Floor">
    <g class="Floorplan Floor-1">
      <g id="space-uuid" class="Space Room" fill="#ffffff" stroke="#ffffff">
        <polygon points="5,5 45,5 45,45 5,45"/>
      </g>
    </g>
  </g>
</svg>
"""


def _write_sample(tmp_path: Path, content: str) -> Path:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / SVG_NAME).write_text(content, encoding="utf-8")
    return sample_dir


def _load_graph(sample_dir: Path) -> dict:
    return json.loads((sample_dir / "masks" / GRAPH_FILENAME).read_text())


def _load_metrics(sample_dir: Path) -> dict:
    return json.loads((sample_dir / "masks" / METRICS_FILENAME).read_text())


# ---------------------------------------------------------------------------
# svg_to_pixel_transform
# ---------------------------------------------------------------------------


def test_svg_to_pixel_transform_with_offset_and_scale():
    svg_root = etree.fromstring('<svg viewBox="10 20 100 50"/>')
    transform = svg_to_pixel_transform(svg_root, width=200, height=100)
    assert transform == pytest.approx((10.0, 20.0, 2.0, 2.0))


def test_svg_to_pixel_transform_no_viewbox_is_identity():
    svg_root = etree.fromstring("<svg/>")
    transform = svg_to_pixel_transform(svg_root, width=200, height=100)
    assert transform == (0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Full room fixture
# ---------------------------------------------------------------------------


def test_room_graph_structure(tmp_path):
    sample_dir = _write_sample(tmp_path, ROOM_SVG)
    result = generate_wall_graph(sample_dir, verbose=False)

    assert result["status"] == "ok", result

    graph = _load_graph(sample_dir)
    assert graph["sample_id"] == "sample"
    assert graph["coordinate_space"] == "raster"
    assert graph["image_width"] == 200
    assert graph["image_height"] == 200

    wall_nodes = [n for n in graph["nodes"] if n["type"] == "wall_node"]
    door_nodes = [n for n in graph["nodes"] if n["type"] == "door_center"]
    window_nodes = [n for n in graph["nodes"] if n["type"] == "window_center"]

    assert len(wall_nodes) >= 4
    assert len(door_nodes) == 1
    assert len(window_nodes) == 1
    assert len(graph["edges"]) >= 4

    # Every edge must be orthogonal (x1==x2 or y1==y2).
    node_xy = {n["id"]: (n["x"], n["y"]) for n in graph["nodes"]}
    for edge in graph["edges"]:
        x1, y1 = node_xy[edge["start"]]
        x2, y2 = node_xy[edge["end"]]
        assert x1 == pytest.approx(x2) or y1 == pytest.approx(y2)

    # Opening-center nodes must never be edge endpoints.
    wall_node_ids = {n["id"] for n in wall_nodes}
    edge_endpoint_ids = {e["start"] for e in graph["edges"]} | {e["end"] for e in graph["edges"]}
    assert edge_endpoint_ids <= wall_node_ids
    door_id = door_nodes[0]["id"]
    window_id = window_nodes[0]["id"]
    assert door_id not in edge_endpoint_ids
    assert window_id not in edge_endpoint_ids

    # Opening nodes must reference a valid host edge with sensible metadata.
    edge_ids = {e["id"] for e in graph["edges"]}
    assert door_nodes[0]["host_edge"] in edge_ids
    assert window_nodes[0]["host_edge"] in edge_ids
    assert door_nodes[0]["opening_width_px"] > 0
    assert window_nodes[0]["opening_width_px"] > 0
    assert door_nodes[0]["orientation"] in ("horizontal", "vertical")
    assert len(door_nodes[0]["source_bbox"]) == 4

    metrics = _load_metrics(sample_dir)
    assert metrics["status"] == "ok"
    assert metrics["reasons"] == []
    assert metrics["door_center_count"] == 1
    assert metrics["window_center_count"] == 1

    assert (sample_dir / "masks" / DEBUG_PNG_FILENAME).stat().st_size > 0
    assert (sample_dir / "masks" / DEBUG_SVG_FILENAME).stat().st_size > 0


def test_room_graph_stays_connected_across_door_and_window_gaps(tmp_path):
    sample_dir = _write_sample(tmp_path, ROOM_SVG)
    generate_wall_graph(sample_dir)
    graph = _load_graph(sample_dir)

    g = nx.Graph()
    for n in graph["nodes"]:
        if n["type"] == "wall_node":
            g.add_node(n["id"])
    for e in graph["edges"]:
        g.add_edge(e["start"], e["end"])

    assert nx.number_connected_components(g) == 1


def test_overwrite_flag_controls_regeneration(tmp_path):
    sample_dir = _write_sample(tmp_path, ROOM_SVG)
    first = generate_wall_graph(sample_dir)
    assert first["status"] == "ok"

    skipped = generate_wall_graph(sample_dir, overwrite=False)
    assert skipped["status"] == "skipped"

    redone = generate_wall_graph(sample_dir, overwrite=True)
    assert redone["status"] == "ok"


# ---------------------------------------------------------------------------
# Degenerate sample
# ---------------------------------------------------------------------------


def test_no_wall_sample_is_marked_unusable(tmp_path):
    sample_dir = _write_sample(tmp_path, NO_WALL_SVG)
    result = generate_wall_graph(sample_dir)

    assert result["status"] == "unusable"
    assert result["reasons"]

    metrics = _load_metrics(sample_dir)
    assert metrics["status"] == "unusable"
    assert "too_few_nodes" in metrics["reasons"] or "no_edges" in metrics["reasons"]

    # Still writes a (empty) graph + debug artifacts rather than raising.
    graph = _load_graph(sample_dir)
    assert graph["nodes"] == []
    assert graph["edges"] == []
    assert (sample_dir / "masks" / DEBUG_PNG_FILENAME).exists()


def test_missing_svg_is_reported(tmp_path):
    sample_dir = tmp_path / "no_svg_sample"
    sample_dir.mkdir()
    result = generate_wall_graph(sample_dir)
    assert result["status"] == "missing_svg"
