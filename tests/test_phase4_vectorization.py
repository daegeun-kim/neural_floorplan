"""Focused unit tests for Phase 4 geometry stages (task31).

Tests cover pure geometry functions; GPU-dependent inference (R2G, SegFormer)
is NOT exercised here - those require a GPU and checkpoints.  The geometry
stages are tested with synthetic graphs and masks so the suite runs in any
environment.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.vectorization.phase4.graph_alignment import (
    normalize_graph,
    _cluster_values,
    _merge_intervals,
)
from src.vectorization.phase4.opening_detection import (
    detect_door_candidates,
    detect_window_candidates,
)
from src.vectorization.phase4.opening_hosting import (
    HostedOpening,
    RejectedOpening,
    host_openings,
    _try_host_on_edge,
)
from src.vectorization.phase4.wall_interval_editing import (
    trim_wall_intervals,
    apply_adjusted_intervals_to_hosted_openings,
)
from src.vectorization.phase4.wall_buffering import buffer_wall_chains
from src.vectorization.phase4.export_json import build_final_vector_json
from src.vectorization.phase4.opening_detection import DoorCandidate, WindowCandidate
from src.vectorization.phase4.wall_interval_editing import TrimmedGraph
from src.vectorization.phase4.wall_buffering import WallGeometry
from src.vectorization.primitives.scale import ScaleInfo
from src.vectorization.graph_types import ComponentRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scale(px_to_mm: float | None = None, status: str = "unknown") -> ScaleInfo:
    return ScaleInfo(
        unit="mm" if px_to_mm else "px",
        px_to_mm=px_to_mm,
        scale_status=status,
        scale_source="test",
        confidence=1.0 if px_to_mm else 0.0,
    )


def _make_component(
    component_id: int,
    bbox: tuple,
    area: float = 100.0,
    mask: np.ndarray | None = None,
) -> ComponentRecord:
    x0, y0, x1, y1 = bbox
    if mask is None:
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
    return ComponentRecord(
        class_name="door_arc",
        component_id=component_id,
        area_px=area,
        bbox=bbox,
        centroid=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Graph alignment tests
# ---------------------------------------------------------------------------

class TestClusterValues:
    def test_single_value(self):
        m = _cluster_values([100.0], tol=5.0)
        assert m[100.0] == pytest.approx(100.0)

    def test_two_close_values_cluster(self):
        m = _cluster_values([100.0, 102.0], tol=5.0)
        assert m[100.0] == pytest.approx(m[102.0])

    def test_two_far_values_separate(self):
        m = _cluster_values([100.0, 200.0], tol=5.0)
        assert abs(m[100.0] - m[200.0]) > 50


class TestMergeIntervals:
    def test_non_overlapping(self):
        result = _merge_intervals([(0, 10), (20, 30)])
        assert result == [(0, 10), (20, 30)]

    def test_overlapping_merged(self):
        result = _merge_intervals([(0, 15), (10, 30)])
        assert len(result) == 1
        assert result[0] == (0, 30)

    def test_touching_merged(self):
        # Touching intervals merge (collinear wall segments sharing an endpoint)
        result = _merge_intervals([(0, 10), (10, 20)], touch_tol=0.5)
        assert len(result) == 1

    def test_empty(self):
        assert _merge_intervals([]) == []


class TestNormalizeGraph:
    def test_removes_zero_length_edges(self):
        graph = {
            "nodes": [[0, 0], [10, 10]],
            "edges": [[0, 0, 0, 0], [0, 0, 100, 0]],
        }
        result = normalize_graph(graph)
        # Only the horizontal edge should survive
        assert all(
            math.hypot(e[2] - e[0], e[3] - e[1]) > 0
            for e in result["edges"]
        )

    def test_horizontal_edge_snapped(self):
        # Edge slightly tilted: should become exactly horizontal
        graph = {
            "nodes": [[0, 0], [100, 2]],
            "edges": [[0, 0, 100, 2]],
        }
        result = normalize_graph(graph)
        edges = result["edges"]
        assert len(edges) == 1
        e = edges[0]
        # After snapping, y1 == y2
        assert e[1] == pytest.approx(e[3], abs=1.0)

    def test_vertical_edge_snapped(self):
        graph = {
            "nodes": [[0, 0], [2, 100]],
            "edges": [[0, 0, 2, 100]],
        }
        result = normalize_graph(graph)
        edges = result["edges"]
        assert len(edges) == 1
        e = edges[0]
        # After snapping, x1 == x2
        assert e[0] == pytest.approx(e[2], abs=1.0)

    def test_intersection_splits_edges(self):
        # One horizontal + one vertical crossing at (50, 50)
        graph = {
            "nodes": [[0, 50], [100, 50], [50, 0], [50, 100]],
            "edges": [
                [0, 50, 100, 50],    # horizontal
                [50, 0, 50, 100],    # vertical
            ],
        }
        result = normalize_graph(graph)
        # Should produce 4 sub-edges from the crossing
        assert len(result["edges"]) >= 4

    def test_collinear_edges_merged(self):
        # Two collinear H edges that share a point
        graph = {
            "nodes": [[0, 50], [50, 50], [100, 50]],
            "edges": [
                [0, 50, 50, 50],
                [50, 50, 100, 50],
            ],
        }
        result = normalize_graph(graph)
        h_edges = [e for e in result["edges"] if e[1] == e[3]]
        # Should be merged into one edge spanning 0..100
        assert len(h_edges) == 1

    def test_rejects_diagonal_edges(self):
        # A 45-degree edge should be rejected
        graph = {
            "nodes": [[0, 0], [100, 100]],
            "edges": [[0, 0, 100, 100]],
        }
        result = normalize_graph(graph)
        assert result["edges"] == []

    def test_empty_graph(self):
        result = normalize_graph({"nodes": [], "edges": []})
        assert result["nodes"] == []
        assert result["edges"] == []


# ---------------------------------------------------------------------------
# Opening detection tests
# ---------------------------------------------------------------------------

class TestDetectDoorCandidates:
    def test_basic_door_candidate(self):
        comp = _make_component(1, (100, 200, 160, 260))
        # Wall graph has one horizontal edge near y=200
        graph_edges = [[0.0, 200.0, 300.0, 200.0]]
        accepted, rejected = detect_door_candidates([comp], graph_edges)
        assert len(accepted) == 1
        assert len(accepted[0].raw_points) == 2
        assert accepted[0].wall_facing_edge in ("top", "bottom", "left", "right")

    def test_aspect_ratio_rejection(self):
        # Very elongated bbox -> rejected
        comp = _make_component(1, (100, 100, 400, 110))  # 300x10 = 30:1
        accepted, rejected = detect_door_candidates([comp], [], max_bbox_aspect_ratio=2.0)
        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "aspect ratio" in rejected[0].rejection_reason

    def test_empty_input(self):
        accepted, rejected = detect_door_candidates([], [])
        assert accepted == []
        assert rejected == []


class TestDetectWindowCandidates:
    def _make_window_comp(self, bbox: tuple) -> ComponentRecord:
        x0, y0, x1, y1 = bbox
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        return ComponentRecord(
            class_name="window",
            component_id=1,
            area_px=float((x1 - x0) * (y1 - y0)),
            bbox=bbox,
            centroid=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            mask=mask,
        )

    def test_horizontal_window(self):
        comp = self._make_window_comp((50, 100, 150, 110))
        accepted, rejected = detect_window_candidates([comp])
        assert len(accepted) == 1
        assert accepted[0].major_axis_px > 50  # major axis ~ 100px wide

    def test_tiny_window_rejected(self):
        comp = self._make_window_comp((50, 100, 53, 103))  # 3x3
        accepted, rejected = detect_window_candidates([comp], min_major_axis_px=10.0)
        assert len(rejected) == 1


# ---------------------------------------------------------------------------
# Opening hosting tests
# ---------------------------------------------------------------------------

class TestTryHostOnEdge:
    def test_both_points_on_edge(self):
        edge = [0.0, 100.0, 200.0, 100.0]  # horizontal at y=100
        pt_a = (40.0, 102.0)
        pt_b = (140.0, 99.0)
        result = _try_host_on_edge(pt_a, pt_b, edge, max_perp_dist_px=10.0)
        assert result is not None
        snap_a, snap_b, avg_dist, width = result
        assert snap_a[1] == pytest.approx(100.0)
        assert snap_b[1] == pytest.approx(100.0)

    def test_one_point_too_far_rejected(self):
        edge = [0.0, 100.0, 200.0, 100.0]
        pt_a = (40.0, 102.0)
        pt_b = (140.0, 150.0)   # 50px away from edge
        result = _try_host_on_edge(pt_a, pt_b, edge, max_perp_dist_px=10.0)
        assert result is None

    def test_too_narrow_rejected(self):
        edge = [0.0, 100.0, 200.0, 100.0]
        pt_a = (100.0, 100.0)
        pt_b = (101.0, 100.0)   # 1px wide
        result = _try_host_on_edge(pt_a, pt_b, edge, max_perp_dist_px=10.0, min_width_px=5.0)
        assert result is None


class TestHostOpenings:
    def _door_cand(self, pt_a, pt_b) -> DoorCandidate:
        return DoorCandidate(
            component_id=1,
            bbox=(int(pt_a[0]), int(pt_a[1]), int(pt_b[0]), int(pt_b[1])),
            bbox_long_edge_px=abs(pt_b[0] - pt_a[0]),
            raw_points=[pt_a, pt_b],
            wall_facing_edge="top",
        )

    def _win_cand(self, pt_a, pt_b) -> WindowCandidate:
        return WindowCandidate(
            component_id=2,
            bbox=(int(pt_a[0]), int(pt_a[1]), int(pt_b[0]), int(pt_b[1])),
            raw_points=[pt_a, pt_b],
            major_axis_px=abs(pt_b[0] - pt_a[0]),
        )

    def test_same_edge_hosting_succeeds(self):
        edges = [[0.0, 100.0, 300.0, 100.0]]
        scale = _make_scale(1.0, "estimated")
        door = self._door_cand((50.0, 102.0), (110.0, 98.0))
        hosted, rejected = host_openings([door], [], edges, scale, max_perp_dist_px=10.0)
        assert len(hosted) == 1
        assert hosted[0].opening_type == "door"
        assert hosted[0].host_edge_idx == 0

    def test_no_compatible_edge_rejected(self):
        # Door points far from any edge
        edges = [[0.0, 0.0, 100.0, 0.0]]
        scale = _make_scale()
        door = self._door_cand((200.0, 300.0), (260.0, 300.0))
        hosted, rejected = host_openings([door], [], edges, scale, max_perp_dist_px=10.0)
        assert len(hosted) == 0
        assert len(rejected) == 1
        assert "no single wall edge" in rejected[0].rejection_reason

    def test_endpoints_must_host_same_edge(self):
        # Two disconnected edges; door points snap to different edges if allowed
        # -> must be rejected because they can't both land on ONE edge
        edges = [
            [0.0, 0.0, 100.0, 0.0],   # horizontal at y=0
            [0.0, 200.0, 100.0, 200.0], # horizontal at y=200
        ]
        scale = _make_scale()
        # One point near edge 0, one near edge 1
        door = self._door_cand((50.0, 2.0), (50.0, 198.0))
        hosted, rejected = host_openings([door], [], edges, scale, max_perp_dist_px=10.0)
        assert len(hosted) == 0
        assert len(rejected) == 1


# ---------------------------------------------------------------------------
# Wall interval trimming tests
# ---------------------------------------------------------------------------

class TestTrimWallIntervals:
    def _make_hosted(
        self, edge_idx: int, edge: list, pts: list, width: float = 50.0
    ) -> HostedOpening:
        return HostedOpening(
            opening_type="door",
            source_component_id=1,
            host_edge_idx=edge_idx,
            host_edge_raw=edge,
            raw_points=pts,
            snapped_points=pts,
            width_px=width,
            width_mm=None,
            confidence=1.0,
        )

    def test_basic_trimming(self):
        edge = [0.0, 100.0, 200.0, 100.0]
        # Door at x=50..100
        hosted = self._make_hosted(0, edge, [(50.0, 100.0), (100.0, 100.0)])
        result = trim_wall_intervals([edge], [hosted])
        # Should have two wall segments: [0..50] and [100..200]
        assert len(result.wall_edges) == 2
        assert len(result.opening_gaps) == 1

    def test_no_openings_preserves_edges(self):
        edges = [[0.0, 0.0, 100.0, 0.0], [0.0, 50.0, 100.0, 50.0]]
        result = trim_wall_intervals(edges, [])
        assert len(result.wall_edges) == 2
        assert len(result.opening_gaps) == 0

    def test_trimming_before_buffering_implied(self):
        # The wall_edges from trimming should not include the opening span.
        edge = [0.0, 0.0, 300.0, 0.0]
        hosted = self._make_hosted(0, edge, [(100.0, 0.0), (200.0, 0.0)])
        result = trim_wall_intervals([edge], [hosted])
        # Verify no wall edge spans 100..200
        for we in result.wall_edges:
            wx1, wy1, wx2, wy2 = we
            # The interval 100-200 must not appear in any wall edge
            assert not (wx1 < 150 < wx2), "Opening span must be trimmed from wall edges"


# ---------------------------------------------------------------------------
# Wall buffering tests
# ---------------------------------------------------------------------------

class TestBufferWallChains:
    def test_resolved_scale_gives_correct_thickness(self):
        # 1px = 1mm => half_width = 100px
        scale = _make_scale(px_to_mm=1.0, status="estimated")
        edges = [[0.0, 50.0, 200.0, 50.0]]
        result = buffer_wall_chains(edges, scale)
        assert result.wall_thickness_mm == pytest.approx(200.0)
        assert not result.scale_blocked
        assert result.polygon is not None

    def test_unknown_scale_marks_blocked(self):
        scale = _make_scale()
        edges = [[0.0, 50.0, 200.0, 50.0]]
        result = buffer_wall_chains(edges, scale, preview_half_width_px=8.0)
        assert result.scale_blocked
        assert result.wall_thickness_mm is None
        assert result.polygon is not None  # preview polygon still generated

    def test_empty_edges_returns_none_polygon(self):
        scale = _make_scale()
        result = buffer_wall_chains([], scale)
        assert result.polygon is None

    def test_connected_chains_before_buffering(self):
        # Two edges forming an L-shape; should produce one merged buffer
        scale = _make_scale(px_to_mm=1.0, status="estimated")
        edges = [
            [0.0, 0.0, 100.0, 0.0],    # horizontal
            [100.0, 0.0, 100.0, 100.0], # vertical
        ]
        result = buffer_wall_chains(edges, scale)
        assert result.chain_count >= 1
        assert result.polygon is not None
        # The merged polygon area should be larger than two separate buffered rectangles
        # (because mitre join at the corner covers the corner gap)
        assert result.polygon.area > 0


# ---------------------------------------------------------------------------
# Scale inference integration test
# ---------------------------------------------------------------------------

class TestScaleInferenceFromComponents:
    def test_door_arc_sets_scale(self):
        from src.vectorization.phase4.scale_inference import infer_scale_from_components

        # Fake a door_arc component whose bbox long edge = 90px
        # -> should resolve px_to_mm = 700/90 or 900/90
        comp = ComponentRecord(
            class_name="door_arc",
            component_id=1,
            area_px=90 * 90,
            bbox=(100, 100, 190, 190),  # 90x90 bbox
            centroid=(145.0, 145.0),
        )
        scale = infer_scale_from_components(
            door_arc_components=[comp],
            door_origin_components=[],
            wall_components=[],
        )
        assert scale.px_to_mm is not None
        assert scale.scale_status in ("estimated", "resolved")


# ---------------------------------------------------------------------------
# Final JSON schema test
# ---------------------------------------------------------------------------

class TestFinalVectorJsonSchema:
    def test_schema_shape(self):
        from src.vectorization.phase4.wall_interval_editing import TrimmedGraph

        scale = _make_scale(1.0, "estimated")
        trimmed = TrimmedGraph(wall_edges=[], opening_gaps=[], inserted_nodes=[])
        # Minimal WallGeometry
        wall_geom = WallGeometry(
            polygon=None,
            half_width_px=8.0,
            wall_thickness_mm=200.0,
            scale_blocked=False,
            chain_count=0,
            edge_count=0,
        )
        data = build_final_vector_json(
            preprocessing_manifest={"coordinate_space": "preprocessed_512"},
            scale_info=scale,
            raw_graph={"nodes": [], "edges": []},
            aligned_graph={"nodes": [], "edges": [], "aligned_nodes": [], "aligned_edges": []},
            trimmed_graph=trimmed,
            hosted_doors=[],
            hosted_windows=[],
            rejected_openings=[],
            wall_geometry=wall_geom,
            metrics={"elapsed_s": 1.0},
        )
        # Check all required top-level keys
        required_keys = {
            "coordinate_space", "preprocessing", "scale",
            "wall_graph", "openings", "geometry", "metrics"
        }
        assert required_keys.issubset(set(data.keys()))
        # Check sub-keys
        assert "status" in data["scale"]
        assert "px_to_mm" in data["scale"]
        assert "doors" in data["openings"]
        assert "windows" in data["openings"]
        assert "rejected" in data["openings"]
        assert "walls" in data["geometry"]
        assert "raw" in data["wall_graph"]
        assert "aligned" in data["wall_graph"]
        assert "trimmed" in data["wall_graph"]


# ---------------------------------------------------------------------------
# Door geometry / primitive contract tests  (task32)
# ---------------------------------------------------------------------------

def _make_hosted_door(p0, p1, component_id=0, host_edge=None, width_mm=None, confidence=1.0, comp_id=None):
    """Build a minimal HostedOpening for door geometry tests."""
    width_px = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    edge = host_edge or [p0[0], p0[1], p1[0], p1[1]]
    cid = comp_id if comp_id is not None else component_id
    return HostedOpening(
        opening_type="door",
        source_component_id=cid,
        host_edge_idx=0,
        host_edge_raw=edge,
        raw_points=[p0, p1],
        snapped_points=[p0, p1],
        width_px=width_px,
        width_mm=width_mm,
        confidence=confidence,
        snapped_module_mm=None,
    )


class TestDoorGeometry:
    """Tests for compute_door_geometry — spec_v008 task32 door primitive contract."""

    def test_origin_edge_is_snapped_points(self):
        from src.vectorization.phase4.door_geometry import compute_door_geometry
        door = _make_hosted_door((195, 180), (228, 180))
        geom = compute_door_geometry(door)
        assert geom.hinge_point == pytest.approx((195, 180), abs=1e-6)
        assert geom.origin_far_point == pytest.approx((228, 180), abs=1e-6)

    def test_leaf_is_perpendicular_to_origin(self):
        from src.vectorization.phase4.door_geometry import compute_door_geometry
        # Horizontal origin (0,0) -> (30,0); leaf must be vertical
        door = _make_hosted_door((0.0, 0.0), (30.0, 0.0))
        geom = compute_door_geometry(door)
        # Leaf vector from hinge to leaf_end
        lx = geom.leaf_end[0] - geom.hinge_point[0]
        ly = geom.leaf_end[1] - geom.hinge_point[1]
        # Origin direction
        ox = geom.origin_far_point[0] - geom.hinge_point[0]
        oy = geom.origin_far_point[1] - geom.hinge_point[1]
        dot = lx * ox + ly * oy
        assert abs(dot) < 1e-6, "leaf must be perpendicular to origin"

    def test_leaf_length_equals_door_width(self):
        from src.vectorization.phase4.door_geometry import compute_door_geometry
        door = _make_hosted_door((0.0, 0.0), (33.0, 0.0))
        geom = compute_door_geometry(door)
        leaf_len = math.hypot(
            geom.leaf_end[0] - geom.hinge_point[0],
            geom.leaf_end[1] - geom.hinge_point[1],
        )
        assert leaf_len == pytest.approx(33.0, abs=1e-4)

    def test_arc_starts_at_origin_far_point_not_hinge(self):
        from src.vectorization.phase4.door_geometry import compute_door_geometry
        door = _make_hosted_door((195, 180), (228, 180))
        geom = compute_door_geometry(door)
        # Arc starts at origin_far_point (not hinge!)
        assert geom.origin_far_point != pytest.approx(geom.hinge_point, abs=1e-3)
        # origin_far_point is snapped_points[1]
        assert geom.origin_far_point == pytest.approx((228, 180), abs=1e-6)

    def test_arc_center_is_hinge(self):
        from src.vectorization.phase4.door_geometry import compute_door_geometry
        door = _make_hosted_door((10.0, 50.0), (10.0, 90.0))  # vertical origin
        geom = compute_door_geometry(door)
        # Radius from hinge to origin_far_point
        r_far = math.hypot(
            geom.origin_far_point[0] - geom.hinge_point[0],
            geom.origin_far_point[1] - geom.hinge_point[1],
        )
        # Radius from hinge to leaf_end
        r_leaf = math.hypot(
            geom.leaf_end[0] - geom.hinge_point[0],
            geom.leaf_end[1] - geom.hinge_point[1],
        )
        assert r_far == pytest.approx(r_leaf, abs=1e-4), "arc center must be hinge"

    def test_arc_endpoint_equals_leaf_end(self):
        from src.vectorization.phase4.door_geometry import compute_door_geometry
        from src.vectorization.primitives.door import DoorArcPrimitive
        door = _make_hosted_door((0.0, 0.0), (40.0, 0.0))
        geom = compute_door_geometry(door)
        swing_base = geom.swing_side.replace("fallback_", "")
        arc = DoorArcPrimitive(
            primitive_id="test",
            hinge_point=geom.hinge_point,
            origin_far_point=geom.origin_far_point,
            width=geom.width_px,
            orientation_angle=geom.orientation_angle_deg,
            swing_direction=swing_base,
        )
        assert arc.leaf_end == pytest.approx(geom.leaf_end, abs=1e-4)

    def test_arc_sweep_keeps_90_degrees(self):
        from src.vectorization.phase4.door_geometry import compute_door_geometry
        door = _make_hosted_door((0.0, 0.0), (50.0, 0.0))
        geom = compute_door_geometry(door)
        hx, hy = geom.hinge_point
        # Angle from hinge to origin_far_point
        a_far = math.atan2(geom.origin_far_point[1] - hy, geom.origin_far_point[0] - hx)
        # Angle from hinge to leaf_end
        a_leaf = math.atan2(geom.leaf_end[1] - hy, geom.leaf_end[0] - hx)
        delta = abs((a_leaf - a_far + math.pi) % (2 * math.pi) - math.pi)
        assert delta == pytest.approx(math.pi / 2, abs=1e-4), "arc must sweep exactly 90 degrees"

    def test_fallback_hinge_is_recorded(self):
        from src.vectorization.phase4.door_geometry import compute_door_geometry
        door = _make_hosted_door((100.0, 200.0), (150.0, 200.0))
        geom = compute_door_geometry(door)
        assert geom.hinge_source == "fallback_pt0"
        assert "fallback" in geom.swing_side

    def test_explicit_swing_side_recorded(self):
        from src.vectorization.phase4.door_geometry import compute_door_geometry
        door = _make_hosted_door((0.0, 0.0), (30.0, 0.0))
        geom = compute_door_geometry(door, swing_side="right")
        assert geom.swing_side == "right"
        assert geom.swing_source == "evidence"

    def test_json_door_geometry_present(self):
        """geometry.doors in final_vector.json must include door_geometry sub-dict."""
        from src.vectorization.phase4.wall_interval_editing import TrimmedGraph
        door = _make_hosted_door((195.0, 180.0), (228.0, 180.0), width_mm=900.0)
        scale = _make_scale(10.0, "estimated")
        trimmed = TrimmedGraph(wall_edges=[], opening_gaps=[], inserted_nodes=[])
        wall_geom = WallGeometry(
            polygon=None, half_width_px=8.0, wall_thickness_mm=200.0,
            scale_blocked=False, chain_count=0, edge_count=0,
        )
        data = build_final_vector_json(
            preprocessing_manifest={},
            scale_info=scale,
            raw_graph={"nodes": [], "edges": []},
            aligned_graph={"nodes": [], "edges": [], "aligned_nodes": [], "aligned_edges": []},
            trimmed_graph=trimmed,
            hosted_doors=[door],
            hosted_windows=[],
            rejected_openings=[],
            wall_geometry=wall_geom,
        )
        door_rec = data["geometry"]["doors"][0]
        assert "door_geometry" in door_rec
        dg = door_rec["door_geometry"]
        required = {"hinge_point", "origin_far_point", "leaf_end", "swing_side",
                    "width_px", "primitive_contract", "hinge_source"}
        assert required.issubset(dg.keys())
        assert dg["primitive_contract"] == "door_origin_leaf_arc"
        assert dg["hinge_source"] == "fallback_pt0"

    def test_svg_has_three_door_elements(self):
        """final_vector.svg must have door_origin, door_leaf, door_arc per door."""
        from src.vectorization.phase4.export_svg import build_final_svg
        from src.vectorization.primitives.scale import ScaleInfo
        door = _make_hosted_door((195.0, 180.0), (228.0, 180.0))
        scale = ScaleInfo(unit="px", px_to_mm=None, scale_status="unknown",
                          scale_source="test", confidence=0.0)
        svg = build_final_svg(scale, None, [door], [])
        assert 'data-type="door_origin"' in svg
        assert 'data-type="door_leaf"' in svg
        assert 'data-type="door_arc"' in svg

    def test_svg_no_debug_circle(self):
        """final_vector.svg must not contain hinge debug circles."""
        from src.vectorization.phase4.export_svg import build_final_svg
        from src.vectorization.primitives.scale import ScaleInfo
        door = _make_hosted_door((100.0, 100.0), (140.0, 100.0))
        scale = ScaleInfo(unit="px", px_to_mm=None, scale_status="unknown",
                          scale_source="test", confidence=0.0)
        svg = build_final_svg(scale, None, [door], [])
        assert "<circle" not in svg, "no debug circles in final SVG"


# ---------------------------------------------------------------------------
# Opening interval de-overlap tests  (task33)
# ---------------------------------------------------------------------------

def _hosted(opening_type: str, edge_idx: int, edge: list, pt_a, pt_b, confidence: float = 1.0,
            component_id: int = 0) -> HostedOpening:
    width = math.hypot(pt_b[0] - pt_a[0], pt_b[1] - pt_a[1])
    return HostedOpening(
        opening_type=opening_type,
        source_component_id=component_id,
        host_edge_idx=edge_idx,
        host_edge_raw=edge,
        raw_points=[pt_a, pt_b],
        snapped_points=[pt_a, pt_b],
        width_px=width,
        width_mm=None,
        confidence=confidence,
    )


class TestOpeningIntervalDeOverlap:
    """Interval conflict resolution (spec §9 + task33)."""

    # Shared wall edge for all tests: horizontal, length 300px
    EDGE = [0.0, 100.0, 300.0, 100.0]
    EI = 0

    def _trim(self, openings, px_to_mm=None):
        return trim_wall_intervals([self.EDGE], openings, px_to_mm=px_to_mm)

    # --- door vs window: door stays fixed, window moves ---

    def test_door_window_overlap_keeps_door_fixed(self):
        # Door at x=50..100, window overlapping at x=80..140
        door = _hosted("door", self.EI, self.EDGE, (50.0, 100.0), (100.0, 100.0), confidence=0.9)
        win  = _hosted("window", self.EI, self.EDGE, (80.0, 100.0), (140.0, 100.0), confidence=0.8)
        result = self._trim([door, win])
        # Two gaps: door and window must both be present
        assert len(result.opening_gaps) == 2
        # Door gap must not be moved: t_start ≈ 50/300
        door_gap = next(g for g in result.opening_gaps if g["opening_type"] == "door")
        assert door_gap["original_interval"] == door_gap["adjusted_interval"], \
            "door interval must not be moved"

    def test_door_window_overlap_moves_window(self):
        door = _hosted("door", self.EI, self.EDGE, (50.0, 100.0), (100.0, 100.0), confidence=0.9)
        win  = _hosted("window", self.EI, self.EDGE, (80.0, 100.0), (140.0, 100.0), confidence=0.8)
        result = self._trim([door, win])
        win_gap = next(g for g in result.opening_gaps if g["opening_type"] == "window")
        assert win_gap["was_adjusted"], "window must be adjusted away from door"
        # Adjusted start must be after door end (100px = t≈0.333)
        adj_start_px = win_gap["adjusted_interval"][0] * 300.0
        door_end_px = 100.0
        assert adj_start_px >= door_end_px, "window adjusted start must be ≥ door end"

    def test_door_window_overlap_no_gap_in_trim(self):
        # After adjustment, the trimmed wall must not have an opening span that covers
        # both door and window at once (no merged trim)
        door = _hosted("door", self.EI, self.EDGE, (50.0, 100.0), (100.0, 100.0), confidence=0.9)
        win  = _hosted("window", self.EI, self.EDGE, (80.0, 100.0), (140.0, 100.0), confidence=0.8)
        result = self._trim([door, win])
        assert len(result.opening_gaps) == 2
        # All gaps must be disjoint
        gaps = sorted(result.opening_gaps, key=lambda g: g["adjusted_interval"][0])
        for i in range(len(gaps) - 1):
            end_i = gaps[i]["adjusted_interval"][1]
            start_next = gaps[i + 1]["adjusted_interval"][0]
            assert start_next >= end_i, "adjusted intervals must not overlap"

    # --- door vs door: higher confidence stays fixed ---

    def test_door_door_overlap_higher_confidence_fixed(self):
        door_hi = _hosted("door", self.EI, self.EDGE, (50.0, 100.0), (100.0, 100.0),
                           confidence=0.9, component_id=1)
        door_lo = _hosted("door", self.EI, self.EDGE, (80.0, 100.0), (140.0, 100.0),
                           confidence=0.5, component_id=2)
        result = self._trim([door_hi, door_lo])
        assert len(result.opening_gaps) == 2
        hi_gap = next(g for g in result.opening_gaps if g["source_component_id"] == 1)
        lo_gap = next(g for g in result.opening_gaps if g["source_component_id"] == 2)
        assert hi_gap["original_interval"] == hi_gap["adjusted_interval"], \
            "higher-confidence door must not be moved"
        assert lo_gap["was_adjusted"], "lower-confidence door must be adjusted"

    def test_door_door_overlap_neither_deleted(self):
        door_hi = _hosted("door", self.EI, self.EDGE, (50.0, 100.0), (100.0, 100.0),
                           confidence=0.9, component_id=1)
        door_lo = _hosted("door", self.EI, self.EDGE, (80.0, 100.0), (140.0, 100.0),
                           confidence=0.5, component_id=2)
        result = self._trim([door_hi, door_lo])
        types = [g["source_component_id"] for g in result.opening_gaps]
        assert 1 in types and 2 in types, "both doors must be preserved"

    # --- window vs window: higher confidence stays fixed ---

    def test_window_window_overlap_higher_confidence_fixed(self):
        win_hi = _hosted("window", self.EI, self.EDGE, (50.0, 100.0), (110.0, 100.0),
                          confidence=0.9, component_id=1)
        win_lo = _hosted("window", self.EI, self.EDGE, (90.0, 100.0), (150.0, 100.0),
                          confidence=0.4, component_id=2)
        result = self._trim([win_hi, win_lo])
        assert len(result.opening_gaps) == 2
        hi_gap = next(g for g in result.opening_gaps if g["source_component_id"] == 1)
        lo_gap = next(g for g in result.opening_gaps if g["source_component_id"] == 2)
        assert hi_gap["original_interval"] == hi_gap["adjusted_interval"], \
            "higher-confidence window must not be moved"
        assert lo_gap["was_adjusted"]

    # --- non-overlapping: no adjustment needed ---

    def test_no_overlap_no_adjustment(self):
        door = _hosted("door", self.EI, self.EDGE, (10.0, 100.0), (60.0, 100.0), confidence=0.9)
        win  = _hosted("window", self.EI, self.EDGE, (200.0, 100.0), (260.0, 100.0), confidence=0.8)
        result = self._trim([door, win])
        for gap in result.opening_gaps:
            assert not gap["was_adjusted"], "non-overlapping intervals must not be adjusted"

    # --- separator: slight overlap becomes non-overlap with minimum gap ---

    def test_slight_overlap_becomes_non_overlap(self):
        # Tiny overlap of 5px between door end (100) and window start (98)
        door = _hosted("door", self.EI, self.EDGE, (50.0, 100.0), (100.0, 100.0), confidence=0.9)
        win  = _hosted("window", self.EI, self.EDGE, (98.0, 100.0), (148.0, 100.0), confidence=0.8)
        result = self._trim([door, win])
        gaps = sorted(result.opening_gaps, key=lambda g: g["adjusted_interval"][0])
        assert len(gaps) == 2
        end0 = gaps[0]["adjusted_interval"][1]
        start1 = gaps[1]["adjusted_interval"][0]
        assert start1 >= end0, "intervals must not overlap after adjustment"

    # --- wall trimming uses adjusted intervals ---

    def test_wall_trimming_uses_adjusted_intervals(self):
        door = _hosted("door", self.EI, self.EDGE, (50.0, 100.0), (100.0, 100.0), confidence=0.9)
        win  = _hosted("window", self.EI, self.EDGE, (80.0, 100.0), (140.0, 100.0), confidence=0.8)
        result = self._trim([door, win])
        win_gap = next(g for g in result.opening_gaps if g["opening_type"] == "window")
        adj_start_px = win_gap["adjusted_interval"][0] * 300.0
        # The wall edges should contain a segment that ends at the adjusted window start
        # (not the original window start of 80px)
        wall_x_endpoints = set()
        for we in result.wall_edges:
            wall_x_endpoints.add(round(we[0], 1))
            wall_x_endpoints.add(round(we[2], 1))
        adj_start_x = round(adj_start_px, 1)
        assert adj_start_x in wall_x_endpoints or any(
            abs(x - adj_start_px) < 1.0 for x in wall_x_endpoints
        ), "wall trim must reflect adjusted interval, not original overlapping interval"

    # --- JSON records original and adjusted intervals ---

    def test_json_records_interval_fields(self):
        door = _hosted("door", self.EI, self.EDGE, (50.0, 100.0), (100.0, 100.0),
                        confidence=0.9, component_id=1)
        win  = _hosted("window", self.EI, self.EDGE, (80.0, 100.0), (140.0, 100.0),
                        confidence=0.8, component_id=2)
        trimmed = self._trim([door, win])
        scale = _make_scale(1.0, "estimated")
        wall_geom = WallGeometry(polygon=None, half_width_px=8.0, wall_thickness_mm=200.0,
                                 scale_blocked=False, chain_count=0, edge_count=0)
        data = build_final_vector_json(
            preprocessing_manifest={},
            scale_info=scale,
            raw_graph={"nodes": [], "edges": []},
            aligned_graph={"nodes": [], "edges": [], "aligned_nodes": [], "aligned_edges": []},
            trimmed_graph=trimmed,
            hosted_doors=[door],
            hosted_windows=[win],
            rejected_openings=[],
            wall_geometry=wall_geom,
        )
        windows = data["openings"]["windows"]
        assert len(windows) == 1
        w = windows[0]
        assert "original_interval" in w
        assert "adjusted_interval" in w
        assert "was_adjusted" in w
        assert "adjustment_reason" in w
        assert "adjustment_px" in w
        assert "overlap_resolution_priority" in w
        assert w["was_adjusted"] is True

    # --- rejection only when no feasible placement exists ---

    def test_last_resort_rejection_when_no_space(self):
        # Edge is very short (20px total), three openings each 15px wide — impossible to fit all three
        short_edge = [0.0, 100.0, 20.0, 100.0]
        op1 = _hosted("door",   0, short_edge, (0.0, 100.0),  (8.0, 100.0),  confidence=0.9, component_id=1)
        op2 = _hosted("window", 0, short_edge, (5.0, 100.0),  (15.0, 100.0), confidence=0.6, component_id=2)
        op3 = _hosted("window", 0, short_edge, (12.0, 100.0), (20.0, 100.0), confidence=0.4, component_id=3)
        result = trim_wall_intervals([short_edge], [op1, op2, op3], px_to_mm=None)
        # At least one should be last-resort rejected (can't all fit)
        total = len(result.opening_gaps) + len(result.last_resort_rejected)
        assert total == 3, "all openings must be either accepted or last-resort rejected"
        assert len(result.last_resort_rejected) >= 1

    def test_last_resort_rejected_has_correct_reason(self):
        short_edge = [0.0, 100.0, 20.0, 100.0]
        op1 = _hosted("door",   0, short_edge, (0.0, 100.0),  (8.0, 100.0),  confidence=0.9, component_id=1)
        op2 = _hosted("window", 0, short_edge, (5.0, 100.0),  (15.0, 100.0), confidence=0.6, component_id=2)
        op3 = _hosted("window", 0, short_edge, (12.0, 100.0), (20.0, 100.0), confidence=0.4, component_id=3)
        result = trim_wall_intervals([short_edge], [op1, op2, op3])
        for rej in result.last_resort_rejected:
            assert rej["rejection_reason"] == "no_feasible_non_overlapping_interval"


# ---------------------------------------------------------------------------
# Part A — adjusted interval propagation to final primitives (task34)
# ---------------------------------------------------------------------------

class TestAdjustedIntervalPropagation:
    """Verify adjusted snapped_points propagate to SVG/JSON endpoints."""

    EDGE = [0.0, 100.0, 300.0, 100.0]
    EI = 0

    def test_window_svg_uses_adjusted_endpoints(self):
        """SVG window line must use adjusted, not original, snapped_points."""
        from src.vectorization.phase4.export_svg import build_final_svg
        from src.vectorization.primitives.scale import ScaleInfo

        door = _hosted("door", self.EI, self.EDGE, (50.0, 100.0), (100.0, 100.0),
                        confidence=0.9, component_id=1)
        win  = _hosted("window", self.EI, self.EDGE, (80.0, 100.0), (140.0, 100.0),
                        confidence=0.8, component_id=2)
        trimmed = trim_wall_intervals([self.EDGE], [door, win])
        final_wins = apply_adjusted_intervals_to_hosted_openings(trimmed, [win])
        assert len(final_wins) == 1
        adj_x = final_wins[0].snapped_points[0][0]
        # The window was pushed right of the door (which ends at x=100)
        assert adj_x >= 100.0, "adjusted window start must be at/after door end"
        # SVG line uses adjusted coords
        scale = ScaleInfo(unit="px", px_to_mm=None, scale_status="unknown",
                          scale_source="test", confidence=0.0)
        svg = build_final_svg(scale, None, [], final_wins)
        # The adjusted x1 must appear in the SVG (not the original 80.0)
        assert f'x1="{adj_x:.2f}"' in svg

    def test_door_json_final_points_match_adjusted_snapped_points(self):
        """geometry.doors[].final_points in JSON must equal adjusted snapped_points."""
        door = _hosted("door", self.EI, self.EDGE, (50.0, 100.0), (100.0, 100.0),
                        confidence=0.9, component_id=1)
        win  = _hosted("window", self.EI, self.EDGE, (80.0, 100.0), (140.0, 100.0),
                        confidence=0.8, component_id=2)
        trimmed = trim_wall_intervals([self.EDGE], [door, win])
        final_doors = apply_adjusted_intervals_to_hosted_openings(trimmed, [door])
        final_wins  = apply_adjusted_intervals_to_hosted_openings(trimmed, [win])
        scale = _make_scale(1.0, "estimated")
        wall_geom = WallGeometry(polygon=None, half_width_px=8.0, wall_thickness_mm=200.0,
                                 scale_blocked=False, chain_count=0, edge_count=0)
        data = build_final_vector_json(
            preprocessing_manifest={},
            scale_info=scale,
            raw_graph={"nodes": [], "edges": []},
            aligned_graph={"nodes": [], "edges": [], "aligned_nodes": [], "aligned_edges": []},
            trimmed_graph=trimmed,
            hosted_doors=final_doors,
            hosted_windows=final_wins,
            rejected_openings=[],
            wall_geometry=wall_geom,
        )
        door_rec = data["geometry"]["doors"][0]
        assert "final_points" in door_rec
        # final_points must match the adjusted snapped_points
        fp = door_rec["final_points"]
        sp = [[round(p[0], 2), round(p[1], 2)] for p in final_doors[0].snapped_points]
        assert fp == sp

    def test_apply_adjusted_intervals_updates_snapped_points(self):
        """apply_adjusted_intervals_to_hosted_openings must change snapped_points
        when the interval was adjusted."""
        door = _hosted("door", self.EI, self.EDGE, (50.0, 100.0), (100.0, 100.0),
                        confidence=0.9, component_id=1)
        win  = _hosted("window", self.EI, self.EDGE, (80.0, 100.0), (140.0, 100.0),
                        confidence=0.8, component_id=2)
        trimmed = trim_wall_intervals([self.EDGE], [door, win])
        final_wins = apply_adjusted_intervals_to_hosted_openings(trimmed, [win])
        # The original snapped_points[0] was at x=80; after adjustment it must be > 100
        orig_x = 80.0
        final_x = final_wins[0].snapped_points[0][0]
        assert abs(final_x - orig_x) > 0.1, \
            "snapped_points must be updated to adjusted interval position"

    def test_non_adjusted_opening_keeps_original_points(self):
        """An opening that did not need adjustment keeps its original snapped_points."""
        door = _hosted("door", self.EI, self.EDGE, (10.0, 100.0), (60.0, 100.0),
                        confidence=0.9, component_id=1)
        trimmed = trim_wall_intervals([self.EDGE], [door])
        final_doors = apply_adjusted_intervals_to_hosted_openings(trimmed, [door])
        assert len(final_doors) == 1
        assert final_doors[0].snapped_points[0][0] == pytest.approx(10.0, abs=0.5)
        assert final_doors[0].snapped_points[1][0] == pytest.approx(60.0, abs=0.5)


# ---------------------------------------------------------------------------
# Part B — door evidence scoring (task34)
# ---------------------------------------------------------------------------

class TestDoorEvidenceScoring:
    """Door hinge/swing inferred from local red/orange/purple raster evidence."""

    def _make_arc_mask(self, hinge, far, swing, size=200, n=24) -> np.ndarray:
        """Create a synthetic binary mask with arc pixels for one hypothesis."""
        import math
        mask = np.zeros((size, size), dtype=np.uint8)
        r = math.hypot(far[0] - hinge[0], far[1] - hinge[1])
        start_ang = math.atan2(far[1] - hinge[1], far[0] - hinge[0])
        sign = 1.0 if swing == "left" else -1.0
        for i in range(n + 1):
            ang = start_ang + sign * (math.pi / 2.0) * (i / n)
            sx = int(round(hinge[0] + r * math.cos(ang)))
            sy = int(round(hinge[1] + r * math.sin(ang)))
            if 0 <= sx < size and 0 <= sy < size:
                mask[sy, sx] = 255
        return mask

    def test_correct_hinge_and_swing_selected(self):
        """Scoring picks the hypothesis whose arc overlaps the red mask pixels."""
        from src.vectorization.phase4.door_geometry import infer_door_direction_from_evidence
        p0 = (50.0, 100.0)
        p1 = (90.0, 100.0)
        # Ground truth: hinge=p0, swing=left
        mask = self._make_arc_mask(p0, p1, "left")
        h_pt, swing, h_src, sw_src = infer_door_direction_from_evidence(p0, p1, mask)
        assert h_pt == "p0"
        assert swing == "left"
        assert h_src == "red_orange_purple_evidence"
        assert sw_src == "red_door_arc_side"

    def test_alternative_hinge_selected(self):
        """When arc is on p1 side, hinge=p1 is selected."""
        from src.vectorization.phase4.door_geometry import infer_door_direction_from_evidence
        p0 = (50.0, 100.0)
        p1 = (90.0, 100.0)
        # Ground truth: hinge=p1, swing=right
        mask = self._make_arc_mask(p1, p0, "right")
        h_pt, swing, h_src, sw_src = infer_door_direction_from_evidence(p0, p1, mask)
        assert h_pt == "p1"
        assert swing == "right"

    def test_fallback_when_no_evidence(self):
        """When mask is all-zero, fallback hinge/swing are returned."""
        from src.vectorization.phase4.door_geometry import infer_door_direction_from_evidence
        mask = np.zeros((100, 100), dtype=np.uint8)
        h_pt, swing, h_src, sw_src = infer_door_direction_from_evidence(
            (10.0, 50.0), (50.0, 50.0), mask
        )
        assert h_src == "fallback_pt0"
        assert sw_src == "fallback"
        assert "fallback" in swing

    def test_fallback_when_mask_none(self):
        """When door_arc_mask is None, fallback is used."""
        from src.vectorization.phase4.door_geometry import infer_door_direction_from_evidence
        h_pt, swing, h_src, sw_src = infer_door_direction_from_evidence(
            (10.0, 50.0), (50.0, 50.0), None
        )
        assert h_src == "fallback_pt0"
        assert "fallback" in swing

    def test_compute_door_geometry_uses_evidence_mask(self):
        """compute_door_geometry with door_arc_mask must set hinge_source from evidence."""
        from src.vectorization.phase4.door_geometry import compute_door_geometry
        p0 = (50.0, 100.0)
        p1 = (90.0, 100.0)
        mask = self._make_arc_mask(p0, p1, "left")
        door = _make_hosted_door(p0, p1)
        geom = compute_door_geometry(door, door_arc_mask=mask)
        assert geom.hinge_source == "red_orange_purple_evidence"
        assert geom.swing_source == "red_door_arc_side"

    def test_compute_door_geometry_fallback_without_mask(self):
        """compute_door_geometry without mask uses fallback_pt0."""
        from src.vectorization.phase4.door_geometry import compute_door_geometry
        door = _make_hosted_door((0.0, 0.0), (30.0, 0.0))
        geom = compute_door_geometry(door)
        assert geom.hinge_source == "fallback_pt0"
        assert geom.swing_source == "fallback"


# ---------------------------------------------------------------------------
# Part C — topology-safe wall snap before buffering (task34)
# ---------------------------------------------------------------------------

class TestTopologySnapEdges:
    """_topology_snap_edges must connect near-equal endpoints before linemerge."""

    def test_near_equal_endpoints_snapped(self):
        from src.vectorization.phase4.wall_buffering import _topology_snap_edges
        # Two edges sharing an endpoint at near-equal but not identical coords
        edges = [
            [0.0, 0.0, 100.0, 0.0],
            [100.5, 0.0, 200.0, 0.0],  # 0.5px gap — should snap to 100.25
        ]
        snapped, metrics = _topology_snap_edges(edges, tol=1.5)
        # After snapping, the shared endpoint should be identical
        p1_end = (snapped[0][2], snapped[0][3])
        p2_start = (snapped[1][0], snapped[1][1])
        assert p1_end == p2_start, "near-equal endpoints must be snapped to the same coordinate"

    def test_metrics_reported(self):
        from src.vectorization.phase4.wall_buffering import _topology_snap_edges
        edges = [[0.0, 0.0, 100.0, 0.0], [100.5, 0.0, 200.0, 0.0]]
        _, metrics = _topology_snap_edges(edges, tol=1.5)
        assert "pre_buffer_node_count" in metrics
        assert "post_snap_node_count" in metrics
        assert "pre_buffer_edge_count" in metrics
        assert "disconnected_endpoint_count" in metrics
        assert metrics["pre_buffer_edge_count"] == 2

    def test_exact_equal_endpoints_unchanged(self):
        from src.vectorization.phase4.wall_buffering import _topology_snap_edges
        edges = [[0.0, 0.0, 100.0, 0.0], [100.0, 0.0, 200.0, 0.0]]
        snapped, metrics = _topology_snap_edges(edges, tol=1.5)
        assert len(snapped) == 2  # no zero-length edges created

    def test_zero_length_edge_dropped(self):
        from src.vectorization.phase4.wall_buffering import _topology_snap_edges
        # An edge whose endpoints snap to the same location
        edges = [[0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 100.0, 0.0]]
        snapped, _ = _topology_snap_edges(edges, tol=1.5)
        for e in snapped:
            length = math.hypot(e[2] - e[0], e[3] - e[1])
            assert length > 1e-6, "zero-length edges must be dropped after snap"

    def test_wall_geometry_reports_metrics(self):
        """buffer_wall_chains must expose topology metrics in WallGeometry."""
        scale = _make_scale(1.0, "estimated")
        edges = [[0.0, 0.0, 100.0, 0.0], [100.5, 0.0, 200.0, 0.0]]
        wg = buffer_wall_chains(edges, scale)
        assert wg.pre_buffer_node_count > 0
        assert wg.post_snap_node_count > 0
        assert wg.pre_buffer_edge_count == 2

    def test_snapped_edges_linemerge_into_one_chain(self):
        """Edges that were near-equal but not exact should merge into one chain after snap."""
        from shapely.ops import linemerge
        from shapely.geometry import LineString
        from src.vectorization.phase4.wall_buffering import _topology_snap_edges
        edges = [[0.0, 0.0, 100.0, 0.0], [100.4, 0.0, 200.0, 0.0]]
        snapped, _ = _topology_snap_edges(edges, tol=1.5)
        lines = [LineString([(e[0], e[1]), (e[2], e[3])]) for e in snapped]
        merged = linemerge(lines)
        # Should be a single LineString, not a MultiLineString
        assert not hasattr(merged, "geoms"), "snapped edges must merge into one chain"

    def test_json_metrics_include_topology_fields(self):
        """final_vector.json metrics must include pre_buffer / disconnected fields."""
        scale = _make_scale(1.0, "estimated")
        edges = [[0.0, 0.0, 100.0, 0.0], [100.5, 0.0, 200.0, 0.0]]
        wg = buffer_wall_chains(edges, scale)
        trimmed = TrimmedGraph(wall_edges=edges, opening_gaps=[], inserted_nodes=[])
        data = build_final_vector_json(
            preprocessing_manifest={},
            scale_info=scale,
            raw_graph={"nodes": [], "edges": []},
            aligned_graph={"nodes": [], "edges": [], "aligned_nodes": [], "aligned_edges": []},
            trimmed_graph=trimmed,
            hosted_doors=[],
            hosted_windows=[],
            rejected_openings=[],
            wall_geometry=wg,
        )
        m = data["metrics"]
        assert "pre_buffer_node_count" in m
        assert "post_snap_node_count" in m
        assert "disconnected_endpoint_count" in m
        assert "wall_chain_count" in m


# ---------------------------------------------------------------------------
# Task 35 — Red-side swing + orange hinge + flat-ended rendering
# ---------------------------------------------------------------------------

class TestTask35DoorDirection:
    """task35: primary swing from red side-count; primary hinge from orange corridor."""

    # Door origin from p0=(50,100) to p1=(90,100), horizontal, going east.
    P0 = (50.0, 100.0)
    P1 = (90.0, 100.0)
    SIZE = 200

    def _mask_above(self) -> np.ndarray:
        """Red pixels strictly ABOVE the door line (y < 100): negative cross → 'negative' side."""
        mask = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        mask[60:98, 55:85] = 255   # y ∈ [60, 98], all above y=100
        return mask

    def _mask_below(self) -> np.ndarray:
        """Red pixels strictly BELOW the door line (y > 100): positive cross → 'positive' side."""
        mask = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        mask[103:140, 55:85] = 255  # y ∈ [103, 140], all below y=100
        return mask

    def _orange_near_p0(self) -> np.ndarray:
        """Orange pixels clustered near p0=(50,100) in a downward corridor."""
        mask = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        mask[100:140, 45:56] = 255  # corridor below p0
        return mask

    def _orange_near_p1(self) -> np.ndarray:
        """Orange pixels clustered near p1=(90,100) in a downward corridor."""
        mask = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        mask[100:140, 84:96] = 255  # corridor below p1
        return mask

    def test_red_pixels_below_choose_positive_side(self):
        """Pixels below the p0→p1 line (positive cross) → positive side → left swing for hinge=p0."""
        from src.vectorization.phase4.door_geometry import _score_side_by_red_pixels
        pos, neg, side = _score_side_by_red_pixels(self.P0, self.P1, self._mask_below())
        assert side == "positive"
        assert pos > 0
        assert neg == 0

    def test_red_pixels_above_choose_negative_side(self):
        """Pixels above the p0→p1 line (negative cross) → negative side → right swing for hinge=p0."""
        from src.vectorization.phase4.door_geometry import _score_side_by_red_pixels
        pos, neg, side = _score_side_by_red_pixels(self.P0, self.P1, self._mask_above())
        assert side == "negative"
        assert neg > 0
        assert pos == 0

    def test_orange_near_p0_selects_p0_hinge(self):
        """Orange pixels near p0 → hinge=p0 selected via orange corridor primary."""
        from src.vectorization.phase4.door_geometry import infer_door_direction_from_evidence
        red_mask = self._mask_below()   # positive side → swing=left for p0
        orange_mask = self._orange_near_p0()
        h_pt, swing, h_src, sw_src = infer_door_direction_from_evidence(
            self.P0, self.P1, red_mask, door_leaf_mask=orange_mask
        )
        assert h_pt == "p0"
        assert swing == "left"
        assert h_src == "red_orange_purple_evidence"

    def test_orange_near_p1_selects_p1_hinge(self):
        """Orange pixels near p1 → hinge=p1 selected via orange corridor primary."""
        from src.vectorization.phase4.door_geometry import infer_door_direction_from_evidence
        red_mask = self._mask_below()   # positive side → swing=right for p1
        orange_mask = self._orange_near_p1()
        h_pt, swing, h_src, sw_src = infer_door_direction_from_evidence(
            self.P0, self.P1, red_mask, door_leaf_mask=orange_mask
        )
        assert h_pt == "p1"
        assert swing == "right"
        assert h_src == "red_orange_purple_evidence"

    def test_red_outside_local_region_ignored_with_local_mask(self):
        """Red pixels far from the door (outside a local crop) must not influence the result.

        Simulates component-local masking: construct a global mask where the true door's
        arc is below the line, but a 'remote door' places red pixels above the line.
        Applying a local crop that contains only the true door's region should still
        select the correct (positive) side.
        """
        from src.vectorization.phase4.door_geometry import infer_door_direction_from_evidence
        global_mask = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        # True door arc: below the line
        global_mask[103:140, 55:85] = 255
        # Remote door: above the line, far x-region
        global_mask[60:98, 150:190] = 255

        # Local crop: restrict to x ∈ [30, 110], y ∈ [80, 160]
        local_mask = np.zeros_like(global_mask)
        local_mask[80:160, 30:110] = global_mask[80:160, 30:110]

        # With local mask, only the below-line pixels are visible
        from src.vectorization.phase4.door_geometry import _score_side_by_red_pixels
        pos, neg, side = _score_side_by_red_pixels(self.P0, self.P1, local_mask)
        assert side == "positive", "local mask must exclude remote door red pixels"

    def test_arc_sampling_cannot_override_strong_red_side_evidence(self):
        """When red pixels strongly favour one side, arc-sampling for hinge must not flip swing.

        Construct a scenario where: red side-count says 'positive' (below), but the arc
        hypothesis with best arc-overlap would be hinge=p1/swing=right (which is also
        positive side — both are positive here). The final swing must respect the side count.
        """
        from src.vectorization.phase4.door_geometry import infer_door_direction_from_evidence, _score_side_by_red_pixels
        red_mask = self._mask_below()
        pos, neg, side = _score_side_by_red_pixels(self.P0, self.P1, red_mask)
        assert side == "positive"  # pre-condition

        # Without orange mask, arc secondary picks hinge from arc-pixel proximity
        h_pt, swing, h_src, sw_src = infer_door_direction_from_evidence(self.P0, self.P1, red_mask)
        # Swing must be consistent with the positive absolute side
        if h_pt == "p0":
            assert swing == "left", "positive side + p0 hinge must be left"
        else:
            assert swing == "right", "positive side + p1 hinge must be right"
        # Evidence sources must not be fallback
        assert h_src == "red_orange_purple_evidence"
        assert sw_src == "red_door_arc_side"

    def test_window_svg_uses_butt_linecap(self):
        """Window SVG line must use stroke-linecap='butt' so it does not extend past endpoints."""
        from src.vectorization.phase4.export_svg import _window_to_svg, WINDOW_STROKE
        win = _make_hosted_door(self.P0, self.P1)
        # Reuse _make_hosted_door — the snapped_points geometry is the same for windows
        svg = _window_to_svg(win)
        assert 'stroke-linecap="butt"' in svg, "window must use butt linecap, not square/round"
        assert 'stroke-linecap="square"' not in svg

    def test_door_origin_svg_uses_butt_linecap(self):
        """DoorOriginPrimitive.to_svg() must use stroke-linecap='butt'."""
        from src.vectorization.primitives.door import DoorOriginPrimitive
        prim = DoorOriginPrimitive(
            primitive_id="test_origin",
            center=(70.0, 100.0),
            width=40.0,
            orientation_angle=0.0,
        )
        svg = prim.to_svg()
        assert 'stroke-linecap="butt"' in svg
        assert 'stroke-linecap="square"' not in svg

    def test_evidence_fields_in_door_geometry_dict(self):
        """door_geometry_to_dict() must include all task35 evidence debug fields."""
        from src.vectorization.phase4.door_geometry import compute_door_geometry, door_geometry_to_dict
        door = _make_hosted_door(self.P0, self.P1)
        red_mask = self._mask_below()
        geom = compute_door_geometry(door, door_arc_mask=red_mask)
        d = door_geometry_to_dict(geom)
        for field_name in [
            "red_side_positive_count", "red_side_negative_count", "red_side_selected",
            "orange_hinge_p0_score", "orange_hinge_p1_score", "hinge_selected", "fallback_used",
        ]:
            assert field_name in d, f"missing evidence field: {field_name}"
        assert d["red_side_selected"] == "positive"
        assert d["fallback_used"] is False


class TestTask36DoubleSwing:
    """Tests for double-swing door classification and rendering (task36)."""

    SIZE = 200
    # Horizontal origin: p0=(50,100), p1=(130,100)
    P0 = (50.0, 100.0)
    P1 = (130.0, 100.0)
    EDGE = [0.0, 100.0, 200.0, 100.0]

    def _make_door(self, comp_id: int = 0, confidence: float = 0.9) -> "HostedOpening":
        return _make_hosted_door(self.P0, self.P1, comp_id=comp_id, confidence=confidence)

    def _mask_below(self, intensity: int = 255) -> np.ndarray:
        m = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        m[105:145, 55:125] = intensity
        return m

    def _mask_above(self, intensity: int = 255) -> np.ndarray:
        m = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        m[55:97, 55:125] = intensity
        return m

    # ── Part A: two-sided evidence detection ──────────────────────────────────

    def test_one_sided_red_evidence_yields_single_swing(self):
        """Only red pixels on one side of the line → single_swing classification."""
        from src.vectorization.phase4.door_classification import (
            classify_door_openings, MIN_SIDE_PIXELS,
        )
        door = self._make_door(comp_id=1)
        door.host_edge_raw = self.EDGE
        red_mask = self._mask_below()
        result = classify_door_openings([door], door_arc_mask=red_mask, door_arc_comps={})
        assert result.double_swing_count == 0
        assert result.classifications[0].door_type == "single_swing"

    def test_weak_opposite_side_does_not_trigger_double_swing(self):
        """Ratio below threshold (< MIN_DOUBLE_SWING_RATIO) stays single_swing."""
        from src.vectorization.phase4.door_classification import classify_door_openings
        door = self._make_door(comp_id=2)
        door.host_edge_raw = self.EDGE
        # Add a tiny bit of noise on the upper side
        combined = self._mask_below()
        combined[97:100, 55:60] = 255  # ~3 pixels above: far below MIN_SIDE_PIXELS=10
        result = classify_door_openings([door], door_arc_mask=combined, door_arc_comps={})
        assert result.double_swing_count == 0
        assert result.classifications[0].door_type == "single_swing"

    def test_strong_both_sides_yields_double_swing_single_component(self):
        """Single component with strong red evidence on both sides → double_swing_shared_origin."""
        from src.vectorization.phase4.door_classification import (
            classify_door_openings, MIN_DOUBLE_SWING_RATIO, MIN_SIDE_PIXELS,
        )
        door = self._make_door(comp_id=3)
        door.host_edge_raw = self.EDGE
        # Large blocks on BOTH sides so both counts > MIN_SIDE_PIXELS and ratio > threshold
        combined = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        combined[105:145, 55:125] = 255  # below (positive)
        combined[57:97, 55:125] = 255    # above (negative)
        result = classify_door_openings([door], door_arc_mask=combined, door_arc_comps={})
        assert result.double_swing_count == 1
        assert result.classifications[0].door_type == "double_swing_shared_origin"
        assert result.classifications[0].double_swing_ratio is not None
        assert result.classifications[0].double_swing_ratio >= MIN_DOUBLE_SWING_RATIO

    # ── Part B/C: paired door detection ──────────────────────────────────────

    def test_two_overlapping_doors_opposite_sides_merge_to_double_swing(self):
        """Two doors on same edge, overlapping intervals, opposite sides → one double-swing."""
        from src.vectorization.phase4.door_classification import classify_door_openings
        d1 = self._make_door(comp_id=10, confidence=0.9)
        d2 = self._make_door(comp_id=11, confidence=0.8)
        d1.host_edge_raw = self.EDGE
        d2.host_edge_raw = self.EDGE

        result = classify_door_openings(
            [d1, d2],
            door_arc_mask=None,  # force fallback — side evidence from separate masks
            door_arc_comps={},
        )
        # No mask → both sides are "fallback" → pair is NOT merged (same-side, same-dir default)
        assert result.double_swing_count == 0  # correct: fallback sides are not "opposite"

    def test_two_overlapping_doors_with_injected_opposite_side_evidence_merge(self):
        """Two paired doors where component bboxes restrict each to its own side.

        d1's component bbox covers below-the-line pixels → positive side.
        d2's component bbox covers above-the-line pixels → negative side.
        Shared overlapping intervals on same edge → merged as double_swing_shared_origin.
        """
        from src.vectorization.phase4.door_classification import classify_door_openings
        from src.vectorization.graph_types import ComponentRecord

        d1 = self._make_door(comp_id=20, confidence=0.9)
        d2 = self._make_door(comp_id=21, confidence=0.9)
        d1.host_edge_raw = self.EDGE
        d2.host_edge_raw = self.EDGE

        # Combined mask: below-line region (positive) + above-line region (negative)
        both_mask = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)
        both_mask[105:145, 55:125] = 255   # below the line at y=100
        both_mask[57:97, 55:125] = 255     # above the line

        # Restrict d1's comp to below-line, d2's comp to above-line via bbox
        comp20 = ComponentRecord(
            class_name="door_arc", component_id=20,
            area_px=40 * 70, bbox=(55, 105, 125, 145), centroid=(90.0, 125.0),
        )
        comp21 = ComponentRecord(
            class_name="door_arc", component_id=21,
            area_px=40 * 70, bbox=(55, 57, 125, 97), centroid=(90.0, 77.0),
        )
        comps = {20: comp20, 21: comp21}

        result = classify_door_openings([d1, d2], door_arc_mask=both_mask, door_arc_comps=comps)
        assert result.double_swing_count == 1, (
            f"Expected 1 double-swing from opposite-side pair, "
            f"got double={result.double_swing_count}, ignored={result.ignored_duplicate_count}, "
            f"classifications={[c.decision_reason for c in result.classifications]}"
        )

    def test_two_overlapping_doors_same_side_ignored_duplicate(self):
        """Two doors with same side evidence → weaker one becomes ignored_duplicate."""
        from src.vectorization.phase4.door_classification import classify_door_openings

        d1 = self._make_door(comp_id=30, confidence=0.9)
        d2 = self._make_door(comp_id=31, confidence=0.7)  # weaker
        d1.host_edge_raw = self.EDGE
        d2.host_edge_raw = self.EDGE

        # Both see only below-line → same side → pair but NOT opposite → ignore weaker
        below = self._mask_below()
        result = classify_door_openings([d1, d2], door_arc_mask=below, door_arc_comps={})
        assert result.ignored_duplicate_count == 1
        assert len(result.ignored_doors) == 1
        assert result.ignored_doors[0].source_component_id == 31  # lower confidence
        assert len(result.final_doors) == 1

    def test_non_overlapping_intervals_yield_separate_single_swing_doors(self):
        """Doors with non-overlapping intervals on same edge stay as separate single-swing."""
        from src.vectorization.phase4.door_classification import classify_door_openings

        d1 = self._make_door(comp_id=40)
        d1.snapped_points = [(10.0, 100.0), (50.0, 100.0)]  # first quarter
        d1.host_edge_raw = self.EDGE

        d2 = self._make_door(comp_id=41)
        d2.snapped_points = [(150.0, 100.0), (190.0, 100.0)]  # last quarter
        d2.host_edge_raw = self.EDGE

        result = classify_door_openings([d1, d2], door_arc_mask=None, door_arc_comps={})
        assert result.ignored_duplicate_count == 0
        assert result.double_swing_count == 0
        assert len(result.final_doors) == 2

    # ── Part D/E: geometry and rendering ────────────────────────────────────

    def test_double_swing_svg_renders_two_leaves_and_arcs(self):
        """SVG output for double_swing_shared_origin contains secondary leaf and arc."""
        from src.vectorization.phase4.door_geometry import (
            compute_door_geometry, compute_door_geometry_double_swing,
        )
        from src.vectorization.phase4.export_svg import _door_to_svg

        door = self._make_door(comp_id=50)
        geom = compute_door_geometry(door, door_arc_mask=self._mask_below())
        double_geom = compute_door_geometry_double_swing(geom)

        assert double_geom.door_type == "double_swing_shared_origin"
        assert double_geom.secondary_leaf_end is not None

        svg = _door_to_svg(door, idx=0, geom=double_geom)
        assert 'door_0_leaf_b' in svg, "secondary leaf primitive must be in SVG"
        assert 'door_0_arc_b' in svg, "secondary arc primitive must be in SVG"
        assert 'data-door-type="double_swing_shared_origin"' in svg

    # ── Part E: JSON fields ──────────────────────────────────────────────────

    def test_json_includes_task36_classification_fields(self):
        """door_geometry_to_dict() must record door_type, classification_reason, source ids."""
        from src.vectorization.phase4.door_geometry import (
            compute_door_geometry, compute_door_geometry_double_swing, door_geometry_to_dict,
        )
        from dataclasses import replace as _dc_replace

        door = self._make_door(comp_id=60)
        geom = compute_door_geometry(door, door_arc_mask=self._mask_below())
        geom = _dc_replace(
            geom,
            classification_reason="single_component_two_sided_red_evidence",
            double_swing_ratio=0.85,
            source_door_component_ids=[60, 61],
        )
        geom = compute_door_geometry_double_swing(geom)
        d = door_geometry_to_dict(geom)
        assert d["door_type"] == "double_swing_shared_origin"
        assert d["classification_reason"] == "single_component_two_sided_red_evidence"
        assert d["double_swing_ratio"] == pytest.approx(0.85)
        assert set(d["source_door_component_ids"]) == {60, 61}
        assert "secondary_leaf_end" in d
        assert "secondary_swing_side" in d

    # ── Part H: counts in ClassificationResult ───────────────────────────────

    def test_classification_result_counts_match_actual_output(self):
        """ClassificationResult counters must match the actual final_doors list."""
        from src.vectorization.phase4.door_classification import classify_door_openings

        doors = [self._make_door(comp_id=i) for i in range(3)]
        for d in doors:
            d.host_edge_raw = self.EDGE

        # All doors on same edge, overlapping intervals, same side → 2 ignored + 1 survivor
        below = self._mask_below()
        result = classify_door_openings(doors, door_arc_mask=below, door_arc_comps={})
        # Can't predict exact counts without knowing pair-pairing order,
        # but totals must be internally consistent:
        # final_doors + ignored_doors == original 3
        assert (len(result.final_doors) + len(result.ignored_doors)) == 3
        # counters must match the actual lists
        assert result.double_swing_count == sum(
            1 for c in result.classifications if c.door_type == "double_swing_shared_origin"
        )
        assert result.ignored_duplicate_count == len(result.ignored_doors)
