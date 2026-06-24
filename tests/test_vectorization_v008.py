"""Tests for the v008 orthogonal point-graph mask-to-vector pipeline (run3).

Organized by module, following the reconstruction order in
spec_v008_mask_to_vector.md SS7, and covering the validation requirements in
SS17.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.vectorization.components import extract_components
from src.vectorization.decode_prediction import CLASS_PALETTE, IncompatibleMaskError, decode_class_id_mask, decode_color_mask
from src.vectorization.door_geometry import generate_door_geometry
from src.vectorization.export_svg import build_svg, save_svg
from src.vectorization.graph_types import Attachment, GraphEdge, GraphPoint, ValidationIssue
from src.vectorization.masks import split_class_masks
from src.vectorization.point_alignment import align_points
from src.vectorization.point_connection import connect_points, validate_graph
from src.vectorization.point_detection import build_wall_skeleton_graph, detect_points, validate_points
from src.vectorization.primitives import DoorArcPrimitive, DoorLeafPrimitive, DoorOriginPrimitive, WallPrimitive, WindowPrimitive
from src.vectorization.primitives.scale import ScaleInfo
from src.vectorization.scale import resolve_scale_from_components
from src.vectorization.wall_geometry import segments_to_polygon, wall_edges_to_primitives, window_edges_to_primitives

RESOLVED_SCALE = ScaleInfo(unit="mm", px_to_mm=10.0, scale_status="resolved", confidence=1.0)
UNKNOWN_SCALE = ScaleInfo()


# ---------------------------------------------------------------------------
# Synthetic mask fixtures
# ---------------------------------------------------------------------------


def _l_corner_mask(h: int = 80, w: int = 80) -> np.ndarray:
    """Horizontal + vertical wall meeting at a right-angle corner."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:14, 10:60] = 255
    mask[10:60, 10:14] = 255
    return mask


def _t_junction_mask(h: int = 80, w: int = 80) -> np.ndarray:
    """Horizontal wall with a vertical branch dropping from its midpoint."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:14, 5:75] = 255
    mask[14:60, 38:42] = 255
    return mask


def _cross_mask(h: int = 80, w: int = 80) -> np.ndarray:
    """Horizontal wall with vertical branches both above and below the midpoint."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[36:40, 5:75] = 255
    mask[5:36, 38:42] = 255
    mask[40:75, 38:42] = 255
    return mask


def _free_segment_mask(h: int = 40, w: int = 80) -> np.ndarray:
    """A single straight wall segment with no other wall evidence nearby."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[18:22, 10:70] = 255
    return mask


def _diagonal_mask(h: int = 80, w: int = 80) -> np.ndarray:
    """A genuinely diagonal wall stroke - must be rejected, not snapped."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(60):
        mask[10 + i, 10 + i] = 255
        mask[10 + i, 11 + i] = 255
    return mask


def _wall_with_window_gap_mask(h: int = 40, w: int = 140) -> np.ndarray:
    """Horizontal wall with a real gap (no wall pixels) where a window sits."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:14, 10:50] = 255
    mask[10:14, 85:120] = 255
    return mask


def _window_mask(h: int = 40, w: int = 140) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:14, 50:85] = 255
    return mask


def _wall_with_door_gap_mask(h: int = 100, w: int = 40) -> np.ndarray:
    """Vertical wall with a real gap where a door sits."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:30, 10:14] = 255
    mask[44:90, 10:14] = 255
    return mask


def _door_origin_mask(h: int = 100, w: int = 40) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[30:44, 9:15] = 255
    return mask


def _door_leaf_mask(h: int = 100, w: int = 40) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[44:48, 9:15] = 255
    return mask


def _door_arc_mask(h: int = 100, w: int = 40) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[48:60, 9:30] = 255
    return mask


def _make_color_image(h: int = 64, w: int = 64) -> np.ndarray:
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:] = CLASS_PALETTE[0]
    rgb[10:54, 5:59] = CLASS_PALETTE[1]
    rgb[2:6, :] = CLASS_PALETTE[2]
    rgb[2:6, 20:28] = CLASS_PALETTE[3]
    rgb[56:60, 10:14] = CLASS_PALETTE[4]
    rgb[56:60, 20:24] = CLASS_PALETTE[5]
    rgb[56:60, 30:34] = CLASS_PALETTE[6]
    return rgb


# ---------------------------------------------------------------------------
# decode_prediction (SS17 items 1-3)
# ---------------------------------------------------------------------------


class TestDecodeColorMask:
    def test_all_classes_decoded(self):
        rgb = _make_color_image()
        result = decode_color_mask(rgb)
        assert set(np.unique(result)).issuperset({0, 1, 2, 3, 4, 5, 6})

    def test_retired_5class_palette_raises(self):
        rgb = np.full((16, 16, 3), (200, 80, 80), dtype=np.uint8)
        with pytest.raises(IncompatibleMaskError):
            decode_color_mask(rgb, tolerance=5)


class TestDecodeClassIdMask:
    def test_valid_range_passes_through(self):
        mask = np.array([[0, 1, 2], [3, 4, 6]], dtype=np.uint8)
        assert (decode_class_id_mask(mask) == mask).all()

    def test_retired_5class_value_above_max_class_id_raises(self):
        mask = np.array([[0, 1], [2, 99]], dtype=np.uint8)
        with pytest.raises(IncompatibleMaskError):
            decode_class_id_mask(mask)


class TestSplitClassMasks:
    def test_floor_key_present_but_ignored_downstream(self):
        # spec_v008 SS1/SS2: floor is decoded but ignored for this restart -
        # the key exists for debug/decoded_masks, run_mask_to_vector excludes
        # it before component extraction.
        class_map = np.zeros((8, 8), dtype=np.uint8)
        class_map[2:4, :] = 1
        masks = split_class_masks(class_map)
        assert "floor" in masks
        assert (masks["floor"] == 255).any()


# ---------------------------------------------------------------------------
# components.py
# ---------------------------------------------------------------------------


class TestComponents:
    def test_small_component_rejected_and_recorded(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[1, 1] = 255  # 1px noise
        mask[10:14, 10:20] = 255  # real wall
        components, rejected = extract_components(mask, "wall", min_area_px=8)
        assert len(components) == 1
        assert len(rejected) == 1
        assert rejected[0].kind == "wall_component_too_small"

    def test_door_origin_not_closed_preserves_two_separate_components(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10, 5:8] = 255
        mask[10, 20:23] = 255
        components, _ = extract_components(mask, "door_origin", min_area_px=1)
        assert len(components) == 2

    def test_wall_component_has_skeleton_and_rect_size(self):
        mask = _l_corner_mask()
        components, _ = extract_components(mask, "wall", min_area_px=4)
        assert len(components) == 1
        assert len(components[0].skeleton_points) > 0
        assert components[0].rect_size is not None


# ---------------------------------------------------------------------------
# scale.py (SS17 items 5-6)
# ---------------------------------------------------------------------------


class TestScaleResolution:
    """task12 SS1: red door_arc bbox long-edge clustering is the primary -
    and only - metric-setting source; door_origin/wall are debug-only cross
    checks (task12 scale priority items 3-5)."""

    @staticmethod
    def _arc(component_id: int, long_edge_px: float):
        from src.vectorization.graph_types import ComponentRecord

        return ComponentRecord(
            "door_arc", component_id, area_px=long_edge_px * 10,
            bbox=(0, 0, int(long_edge_px), 10), centroid=(long_edge_px / 2, 5),
        )

    def test_red_arc_bbox_long_edge_resolves_700mm(self):
        # task12 required test 1.
        scale_info = resolve_scale_from_components([self._arc(1, 70.0)], [], [], min_confidence=0.5)
        assert scale_info.scale_status in ("resolved", "estimated")
        assert scale_info.px_to_mm == pytest.approx(10.0)
        assert 700.0 in scale_info.diagnostics["red_arc_selected_modules_mm"]

    def test_red_arc_bbox_long_edge_resolves_900mm(self):
        # task12 required test 2. A single bbox long edge is ambiguous
        # between the 700mm and 900mm modules at different candidate
        # scales (any homogeneous cluster ties under both, since the
        # voting tolerance is relative) - pairing it with a second cluster
        # that is only self-consistent as a 700mm door at the *same*
        # px_to_mm is what lets the 900mm reading win unambiguously.
        scale_info = resolve_scale_from_components([self._arc(1, 90.0), self._arc(2, 70.0)], [], [], min_confidence=0.5)
        assert scale_info.scale_status in ("resolved", "estimated")
        assert scale_info.px_to_mm == pytest.approx(10.0)
        assert 900.0 in scale_info.diagnostics["red_arc_selected_modules_mm"]

    def test_multiple_red_clusters_use_robust_median_voting(self):
        # task12 required test 3: an outlier cluster must not move the
        # winning candidate's median, and should be reported as rejected.
        arcs = [self._arc(1, 70.0), self._arc(2, 71.0), self._arc(3, 69.0), self._arc(4, 200.0)]
        scale_info = resolve_scale_from_components(arcs, [], [], min_confidence=0.5)
        assert scale_info.scale_status in ("resolved", "estimated")
        assert scale_info.px_to_mm == pytest.approx(10.0, abs=0.01)
        assert 200.0 in scale_info.diagnostics["scale_rejected_outliers"]

    def test_red_arc_scale_beats_conflicting_wall_thickness(self):
        # task12 required test 4: noisy wall-thickness evidence must not
        # override the red door_arc scale.
        from src.vectorization.graph_types import ComponentRecord

        walls = [
            ComponentRecord("wall", 1, area_px=400, bbox=(0, 0, 100, 14), centroid=(50, 7), rect_size=(100.0, 14.0)),
        ]
        scale_info = resolve_scale_from_components([self._arc(1, 70.0)], [], walls, min_confidence=0.5)
        assert scale_info.px_to_mm == pytest.approx(10.0)
        assert scale_info.scale_source == "door_arc_bbox_long_edge_clustering"

    def test_insufficient_evidence_is_unknown(self):
        # task12 SS1 priority item 5: unknown unless a usable red door_arc
        # cluster exists - door_origin/wall evidence alone never resolves it.
        from src.vectorization.graph_types import ComponentRecord

        door_origin = [ComponentRecord("door_origin", 1, area_px=200, bbox=(0, 0, 70, 4), centroid=(35, 2), rect_size=(70.0, 4.0))]
        walls = [ComponentRecord("wall", 1, area_px=400, bbox=(0, 0, 100, 10), centroid=(50, 5), rect_size=(100.0, 10.0))]
        scale_info = resolve_scale_from_components([], door_origin, walls)
        assert scale_info.scale_status == "unknown"
        assert scale_info.px_to_mm is None

    def test_majority_cluster_resolves_scale_even_below_default_confidence(self):
        # Bug A regression (rules 8/9/19): 4 clusters cleanly agree on one
        # px_to_mm and 3 are ordinary noise outliers - winning-group fraction
        # is 4/7 ~= 0.57, below the default min_scale_confidence_for_metric
        # (0.70), but rule 19 only requires "no usable red door_arc cluster"
        # to report unknown, which is not the case here.
        arcs = [self._arc(1, 54.0), self._arc(2, 13.0), self._arc(3, 5.0),
                self._arc(4, 64.0), self._arc(5, 56.0), self._arc(6, 28.0), self._arc(7, 51.0)]
        scale_info = resolve_scale_from_components(arcs, [], [])  # default min_confidence=0.70
        assert scale_info.scale_status == "estimated"
        assert scale_info.px_to_mm is not None
        assert scale_info.confidence < 0.70

    def test_zero_red_arcs_stays_unknown_regardless_of_other_evidence(self):
        # The only legitimate "unknown" trigger per rule 19 is a complete
        # absence of usable red door_arc cluster lengths.
        scale_info = resolve_scale_from_components([], [], [])
        assert scale_info.scale_status == "unknown"
        assert scale_info.px_to_mm is None


# ---------------------------------------------------------------------------
# point_detection - wall points (SS9.1, SS17 items 8, 9, 10, 11, 12)
# ---------------------------------------------------------------------------


class TestWallSkeletonGraph:
    def test_l_corner_produces_one_clean_2_wall_point(self):
        mask = _l_corner_mask()
        components, _ = extract_components(mask, "wall", min_area_px=4)
        points, rejected, _edges = detect_points({"wall": components}, {}, UNKNOWN_SCALE)
        corners = [p for p in points if p.point_type == "2_wall_point"]
        assert len(corners) == 1
        assert len(corners[0].attachments) == 2
        dirs = {a.direction for a in corners[0].attachments}
        assert dirs == {"right", "down"} or dirs == {"left", "up"} or len(dirs) == 2

    def test_free_segment_produces_two_1_wall_points(self):
        # SS17 item 11: a 1_wall_point is a legitimate free end and must not
        # be force-extended just because it exists in isolation.
        mask = _free_segment_mask()
        components, _ = extract_components(mask, "wall", min_area_px=4)
        points, rejected, _edges = detect_points({"wall": components}, {}, UNKNOWN_SCALE)
        free_ends = [p for p in points if p.point_type == "1_wall_point"]
        assert len(free_ends) == 2
        for p in free_ends:
            assert len(p.attachments) == 1

    def test_t_junction_produces_3_wall_point_not_forced_free_ends(self):
        # SS17 item 12: branch evidence must produce a 3_wall_point, not a
        # forced 1_wall_point extension.
        mask = _t_junction_mask()
        components, _ = extract_components(mask, "wall", min_area_px=4)
        points, rejected, _edges = detect_points({"wall": components}, {}, UNKNOWN_SCALE)
        t_points = [p for p in points if p.point_type == "3_wall_point"]
        assert len(t_points) == 1
        assert len(t_points[0].attachments) == 3

    def test_cross_produces_4_wall_point(self):
        mask = _cross_mask()
        components, _ = extract_components(mask, "wall", min_area_px=4)
        points, rejected, _edges = detect_points({"wall": components}, {}, UNKNOWN_SCALE)
        cross_points = [p for p in points if p.point_type == "4_wall_point"]
        assert len(cross_points) == 1
        assert len(cross_points[0].attachments) == 4

    def test_diagonal_evidence_rejected_not_snapped(self):
        # SS17 item 8: 45-degree evidence must be rejected to debug, never
        # become a final diagonal wall point/edge.
        mask = _diagonal_mask()
        components, _ = extract_components(mask, "wall", min_area_px=4)
        node_edges, rejected = build_wall_skeleton_graph(components, cardinal_tolerance_deg=20.0)
        assert any(r.kind == "diagonal_wall_edge" for r in rejected)

    def test_every_point_is_one_of_seven_allowed_types(self):
        # SS17 item 9/10: direct search, no unresolved category.
        mask = _cross_mask()
        components, _ = extract_components(mask, "wall", min_area_px=4)
        points, _rejected, _edges = detect_points({"wall": components}, {}, UNKNOWN_SCALE)
        from src.vectorization.graph_types import ALL_POINT_TYPES

        for p in points:
            assert p.point_type in ALL_POINT_TYPES
            for a in p.attachments:
                assert a.direction in ("left", "right", "up", "down")


# ---------------------------------------------------------------------------
# point_detection - window points (SS9.2, SS17 items 13, 14, 15, 16)
# ---------------------------------------------------------------------------


class TestWindowPointDetection:
    def _components(self, scale_info):
        wall_components, _ = extract_components(_wall_with_window_gap_mask(), "wall", min_area_px=4)
        window_components, _ = extract_components(_window_mask(), "window", min_area_px=4)
        return {"wall": wall_components, "window": window_components, "door_arc": [], "door_origin": []}

    def test_window_produces_two_paired_wall_window_points(self):
        components = self._components(RESOLVED_SCALE)
        points, rejected, _edges = detect_points(
            components, {}, RESOLVED_SCALE, {"min_hosted_width_px": 5.0}
        )
        win_points = [p for p in points if p.point_type == "wall_window_point"]
        assert len(win_points) == 2
        assert win_points[0].source_component_ids == win_points[1].source_component_ids

    def test_window_endpoints_face_each_other(self):
        # SS17 item 13/14: opposing window directions determine pairing/axis.
        components = self._components(RESOLVED_SCALE)
        points, _rejected, _edges = detect_points(
            components, {}, RESOLVED_SCALE, {"min_hosted_width_px": 5.0}
        )
        win_points = [p for p in points if p.point_type == "wall_window_point"]
        d0 = win_points[0].attachment_of("window").direction
        d1 = win_points[1].attachment_of("window").direction
        from src.vectorization.graph_types import OPPOSITE_DIRECTION

        assert OPPOSITE_DIRECTION[d0] == d1

    def test_window_below_300mm_minimum_is_rejected(self):
        # SS17 item 15: window length must be >= 300mm when scale is known.
        narrow_wall_mask = np.zeros((40, 60), dtype=np.uint8)
        narrow_wall_mask[10:14, 10:20] = 255
        narrow_wall_mask[10:14, 25:35] = 255
        narrow_window_mask = np.zeros((40, 60), dtype=np.uint8)
        narrow_window_mask[10:14, 20:25] = 255  # 5px -> 50mm at 10mm/px

        wall_components, _ = extract_components(narrow_wall_mask, "wall", min_area_px=4)
        window_components, _ = extract_components(narrow_window_mask, "window", min_area_px=4)
        components = {"wall": wall_components, "window": window_components, "door_arc": [], "door_origin": []}
        points, rejected, _edges = detect_points(components, {}, RESOLVED_SCALE, {"min_hosted_width_px": 2.0})
        assert not any(p.point_type == "wall_window_point" for p in points)
        assert any(r.kind == "window_too_narrow_mm" for r in rejected)

    def test_window_scale_unknown_blocks_window(self):
        components = self._components(UNKNOWN_SCALE)
        points, rejected, _edges = detect_points(
            components, {}, UNKNOWN_SCALE, {"min_hosted_width_px": 5.0}
        )
        assert not any(p.point_type == "wall_window_point" for p in points)
        assert any(r.kind == "window_scale_blocked" for r in rejected)

    def test_window_low_confidence_but_estimated_scale_still_creates_window(self):
        # Bug A regression: confidence is reporting metadata (rule 114), not
        # a creation gate - only an unresolved (status="unknown") scale
        # blocks the window, per rule 19/50.
        low_confidence_scale = ScaleInfo(unit="mm", px_to_mm=10.0, scale_status="estimated", confidence=0.2)
        components = self._components(low_confidence_scale)
        points, rejected, _edges = detect_points(
            components, {}, low_confidence_scale, {"min_hosted_width_px": 5.0}
        )
        assert any(p.point_type == "wall_window_point" for p in points)
        assert not any(r.kind == "window_scale_blocked" for r in rejected)


# ---------------------------------------------------------------------------
# point_detection - door points (SS9.3, SS17 items 17-22)
# ---------------------------------------------------------------------------


class TestDoorPointDetection:
    def _components(self):
        wall_components, _ = extract_components(_wall_with_door_gap_mask(), "wall", min_area_px=4)
        door_origin_components, _ = extract_components(_door_origin_mask(), "door_origin", min_area_px=2)
        door_arc_components, _ = extract_components(_door_arc_mask(), "door_arc", min_area_px=4)
        return {
            "wall": wall_components,
            "window": [],
            "door_arc": door_arc_components,
            "door_origin": door_origin_components,
        }

    def _masks(self):
        return {"door_leaf": _door_leaf_mask(), "door_origin": _door_origin_mask()}

    def test_door_with_red_arc_produces_hinge_and_end_points(self):
        components = self._components()
        points, rejected, _edges = detect_points(components, self._masks(), RESOLVED_SCALE)
        hinge = [p for p in points if p.point_type == "wall_door_hinge_point"]
        end = [p for p in points if p.point_type == "wall_door_end_point"]
        assert len(hinge) == 1
        assert len(end) == 1
        assert hinge[0].source_component_ids == end[0].source_component_ids

    def test_no_door_arc_means_no_door(self):
        # SS17 item 17/18: red door_arc components are the sole standard for
        # door count - origin/leaf evidence alone never creates a door.
        components = self._components()
        components["door_arc"] = []
        points, rejected, _edges = detect_points(components, self._masks(), RESOLVED_SCALE)
        assert not any(p.point_type in ("wall_door_hinge_point", "wall_door_end_point") for p in points)
        assert any(r.kind == "unresolved_door_origin" for r in rejected)

    def test_hinge_prefers_orange_purple_intersection(self):
        # SS17 item 20: hinge lands at the origin/leaf-evidence end, not the far end.
        components = self._components()
        points, _rejected, _edges = detect_points(components, self._masks(), RESOLVED_SCALE)
        hinge = next(p for p in points if p.point_type == "wall_door_hinge_point")
        end = next(p for p in points if p.point_type == "wall_door_end_point")
        # The leaf/arc evidence sits near y=44 (origin's near end); the far
        # end should land further away along the wall axis.
        assert abs(hinge.coordinate[1] - 44.0) < abs(end.coordinate[1] - 44.0)

    def test_hinge_falls_back_to_arc_geometry_without_intersection(self):
        # SS17 item 19/21: missing/no leaf evidence still resolves a door via
        # the arc-geometry + host-wall fallback. The fallback picks whichever
        # arc-bbox corner is nearest the host wall's infinite line, which is
        # ambiguous between the two near-wall corners for an axis-aligned
        # arc against a vertical wall - a generous probe radius is needed so
        # pairing with the door_origin evidence still succeeds regardless of
        # which of the two (equidistant) corners is chosen.
        components = self._components()
        masks = {"door_leaf": np.zeros((100, 40), dtype=np.uint8), "door_origin": _door_origin_mask()}
        points, rejected, _edges = detect_points(
            components, masks, RESOLVED_SCALE, {"hinge_probe_radius": 30.0}
        )
        assert any(p.point_type == "wall_door_hinge_point" for p in points)

    def test_door_width_snaps_to_700_or_900mm(self):
        # SS17 item 22.
        components = self._components()
        points, _rejected, _edges = detect_points(components, self._masks(), RESOLVED_SCALE)
        hinge = next(p for p in points if p.point_type == "wall_door_hinge_point")
        width_px = hinge.attachment_of("door_origin").evidence_length_px
        assert width_px * RESOLVED_SCALE.px_to_mm in (700.0, 900.0)

    def test_door_scale_unknown_blocks_door(self):
        components = self._components()
        points, rejected, _edges = detect_points(components, self._masks(), UNKNOWN_SCALE)
        assert not any(p.point_type == "wall_door_hinge_point" for p in points)
        assert any(r.kind == "unresolved_door_scale_blocked" for r in rejected)

    def test_door_low_confidence_but_estimated_scale_still_creates_door(self):
        # Bug A regression: same as the window case - confidence alone must
        # not block door creation once scale is resolved/estimated.
        low_confidence_scale = ScaleInfo(unit="mm", px_to_mm=10.0, scale_status="estimated", confidence=0.2)
        components = self._components()
        points, rejected, _edges = detect_points(components, self._masks(), low_confidence_scale)
        assert any(p.point_type == "wall_door_hinge_point" for p in points)
        assert not any(r.kind == "unresolved_door_scale_blocked" for r in rejected)


# ---------------------------------------------------------------------------
# task13: red door_arc clusters are guaranteed door objects - forceful
# hinge/end inference and the per-cluster DoorCandidateRecord report.
# ---------------------------------------------------------------------------


class TestForcefulDoorInference:
    def _components(self, door_origin_components=None):
        wall_components, _ = extract_components(_wall_with_door_gap_mask(), "wall", min_area_px=4)
        door_arc_components, _ = extract_components(_door_arc_mask(), "door_arc", min_area_px=4)
        return {
            "wall": wall_components,
            "window": [],
            "door_arc": door_arc_components,
            "door_origin": door_origin_components if door_origin_components is not None else [],
        }

    def _masks(self):
        return {"door_leaf": _door_leaf_mask(), "door_origin": _door_origin_mask()}

    def test_missing_purple_evidence_does_not_delete_the_door(self):
        # task13 required test 4/10: no door_origin component or mask at all
        # near the red cluster - the door must still be created by forcing
        # both the hinge (arc-geometry fallback) and the end point
        # (red-cluster-geometry fallback) from red+wall evidence alone.
        components = self._components(door_origin_components=[])
        masks = {"door_leaf": _door_leaf_mask(), "door_origin": np.zeros((100, 40), dtype=np.uint8)}
        points, rejected, _edges = detect_points(components, masks, RESOLVED_SCALE, {"hinge_probe_radius": 30.0})
        hinge = [p for p in points if p.point_type == "wall_door_hinge_point"]
        end = [p for p in points if p.point_type == "wall_door_end_point"]
        assert len(hinge) == 1
        assert len(end) == 1
        assert not any(r.kind == "unresolved_door_hinge" for r in rejected)

    def test_missing_orange_evidence_does_not_delete_the_door(self):
        # task13 required test 5: door_leaf is entirely absent, but
        # door_origin evidence still lets the door resolve.
        door_origin_components, _ = extract_components(_door_origin_mask(), "door_origin", min_area_px=2)
        components = self._components(door_origin_components=door_origin_components)
        masks = {"door_leaf": np.zeros((100, 40), dtype=np.uint8), "door_origin": _door_origin_mask()}
        points, rejected, _edges = detect_points(components, masks, RESOLVED_SCALE, {"hinge_probe_radius": 30.0})
        assert sum(1 for p in points if p.point_type == "wall_door_hinge_point") == 1
        assert sum(1 for p in points if p.point_type == "wall_door_end_point") == 1

    def test_door_count_equals_accepted_red_cluster_count(self):
        # task13 acceptance criterion: door count == accepted red cluster count.
        door_origin_components, _ = extract_components(_door_origin_mask(), "door_origin", min_area_px=2)
        components = self._components(door_origin_components=door_origin_components)
        points, _rejected, _edges = detect_points(components, self._masks(), RESOLVED_SCALE)
        issues = validate_points(points, accepted_door_arc_count=len(components["door_arc"]))
        assert not any(i.rule == "door_count_mismatch" for i in issues)

    def test_one_door_candidate_record_per_red_cluster(self):
        # task13 required test 1/12: every accepted red door_arc component
        # gets exactly one DoorCandidateRecord, marked created.
        from src.vectorization.point_detection import build_door_candidate_records

        door_origin_components, _ = extract_components(_door_origin_mask(), "door_origin", min_area_px=2)
        components = self._components(door_origin_components=door_origin_components)
        masks = self._masks()
        points, rejected, _edges = detect_points(components, masks, RESOLVED_SCALE)
        records = build_door_candidate_records(
            components["door_arc"], points, rejected, masks, components["wall"], RESOLVED_SCALE
        )
        assert len(records) == len(components["door_arc"])
        assert all(r.created_door_candidate for r in records)
        assert all(r.red_component_id == c.component_id for r, c in zip(records, components["door_arc"]))

    def test_door_candidate_support_classes_reflect_available_evidence(self):
        # task13 required test 6/7/9: with red/orange/purple/black all
        # present, the hinge/end support-class lists include every
        # available evidence type that's actually near the final points.
        from src.vectorization.point_detection import build_door_candidate_records

        door_origin_components, _ = extract_components(_door_origin_mask(), "door_origin", min_area_px=2)
        components = self._components(door_origin_components=door_origin_components)
        masks = self._masks()
        points, rejected, _edges = detect_points(components, masks, RESOLVED_SCALE)
        records = build_door_candidate_records(
            components["door_arc"], points, rejected, masks, components["wall"], RESOLVED_SCALE
        )
        rec = records[0]
        assert "red" in rec.hinge_candidate_support_classes
        assert "red" in rec.end_candidate_support_classes
        assert rec.door_confidence > 0.0

    def test_forced_inference_lowers_confidence_but_keeps_the_door(self):
        # task13 "Forceful Inference Rule": weak/missing evidence lowers
        # door_confidence but never removes the record.
        from src.vectorization.point_detection import build_door_candidate_records

        components = self._components(door_origin_components=[])
        masks = {"door_leaf": np.zeros((100, 40), dtype=np.uint8), "door_origin": np.zeros((100, 40), dtype=np.uint8)}
        points, rejected, _edges = detect_points(components, masks, RESOLVED_SCALE, {"hinge_probe_radius": 30.0})
        records = build_door_candidate_records(
            components["door_arc"], points, rejected, masks, components["wall"], RESOLVED_SCALE
        )
        assert len(records) == 1
        assert records[0].created_door_candidate is True
        assert records[0].door_confidence < 1.0

    def test_fragmented_paired_door_origin_falls_back_to_arc_geometry(self):
        # Bug D regression (rules 47/50/51): the nearest door_origin
        # component within probe radius of the hinge is a tiny fragment
        # (segmentation broke the purple stroke into pieces, same pattern as
        # the red-arc fragmentation in sample_003) - its own projected width
        # is implausibly small. The door must still resolve via the
        # arc-geometry fallback rather than being rejected outright.
        wall_components, _ = extract_components(_wall_with_door_gap_mask(), "wall", min_area_px=4)
        door_arc_components, _ = extract_components(_door_arc_mask(), "door_arc", min_area_px=4)
        tiny_origin_mask = np.zeros((100, 40), dtype=np.uint8)
        tiny_origin_mask[30:33, 9:12] = 255  # 3x3 fragment, not the full origin stroke
        door_origin_components, _ = extract_components(tiny_origin_mask, "door_origin", min_area_px=2)
        components = {
            "wall": wall_components, "window": [],
            "door_arc": door_arc_components, "door_origin": door_origin_components,
        }
        masks = {"door_leaf": tiny_origin_mask.copy(), "door_origin": tiny_origin_mask}

        points, rejected, _edges = detect_points(components, masks, RESOLVED_SCALE)
        hinge = [p for p in points if p.point_type == "wall_door_hinge_point"]
        end = [p for p in points if p.point_type == "wall_door_end_point"]
        assert len(hinge) == 1
        assert len(end) == 1
        assert not any(r.kind == "unresolved_door_too_narrow" for r in rejected)
        # The fallback width must come from the arc, not the 3px fragment.
        width_px = math.hypot(end[0].coordinate[0] - hinge[0].coordinate[0], end[0].coordinate[1] - hinge[0].coordinate[1])
        assert width_px > 10.0

    def test_rejected_door_evidence_attributes_to_the_correct_red_component(self):
        # Bug C regression: rejections produced inside _detect_door_points
        # (scale-blocked, too-narrow, non-cardinal-axis) must attribute back
        # to the originating red door_arc cluster (class_name="door_arc",
        # component_id=arc.component_id), not to a possibly-None origin_id,
        # so build_door_candidate_records can report the real reason instead
        # of falling back to a generic note.
        from src.vectorization.point_detection import build_door_candidate_records

        components = self._components()
        masks = self._masks()
        points, rejected, _edges = detect_points(components, masks, UNKNOWN_SCALE)
        records = build_door_candidate_records(
            components["door_arc"], points, rejected, masks, components["wall"], UNKNOWN_SCALE
        )
        assert len(records) == 1
        assert records[0].created_door_candidate is False
        assert records[0].door_inference_notes == "scale not resolved"

    def test_debug_overlay_renders_door_candidate_bbox(self):
        # task13 "Debug Overlay Requirements": each red cluster is drawn as
        # a door candidate, with its bbox outline color set by confidence.
        from src.vectorization.debug import build_debug_overlay
        from src.vectorization.point_detection import build_door_candidate_records

        door_origin_components, _ = extract_components(_door_origin_mask(), "door_origin", min_area_px=2)
        components = self._components(door_origin_components=door_origin_components)
        masks = self._masks()
        points, rejected, _edges = detect_points(components, masks, RESOLVED_SCALE)
        records = build_door_candidate_records(
            components["door_arc"], points, rejected, masks, components["wall"], RESOLVED_SCALE
        )
        rgb = np.zeros((100, 40, 3), dtype=np.uint8)
        overlay = build_debug_overlay(rgb, points, [], rejected, RESOLVED_SCALE, records)
        x0, y0, x1, y1 = records[0].red_bbox
        pixels = np.array(overlay)[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1]
        assert pixels.any()


# ---------------------------------------------------------------------------
# point_detection.validate_points (SS10)
# ---------------------------------------------------------------------------


class TestValidatePoints:
    def test_even_window_point_count_passes(self):
        p1 = GraphPoint("a", "wall_window_point", (0.0, 0.0), [])
        p2 = GraphPoint("b", "wall_window_point", (10.0, 0.0), [])
        assert validate_points([p1, p2]) == []

    def test_odd_window_point_count_flagged(self):
        p1 = GraphPoint("a", "wall_window_point", (0.0, 0.0), [])
        issues = validate_points([p1])
        assert any(i.rule == "odd_window_point_count" for i in issues)

    def test_hinge_end_count_mismatch_flagged(self):
        p1 = GraphPoint("a", "wall_door_hinge_point", (0.0, 0.0), [])
        issues = validate_points([p1])
        assert any(i.rule == "door_hinge_end_mismatch" for i in issues)


# ---------------------------------------------------------------------------
# point_alignment + point_connection (SS11/SS12, SS17 items 7, 16, 25)
# ---------------------------------------------------------------------------


class TestPointAlignmentAndConnection:
    def test_l_corner_walls_align_and_connect_into_orthogonal_edges(self):
        mask = _l_corner_mask()
        components, _ = extract_components(mask, "wall", min_area_px=4)
        points, _rejected, wall_edges = detect_points({"wall": components}, {}, UNKNOWN_SCALE)
        aligned, _issues = align_points(points, components, UNKNOWN_SCALE, {}, wall_edges)
        edges, _graph_issues = connect_points(aligned, wall_edges, UNKNOWN_SCALE)

        assert len(edges) == 2
        for e in edges:
            assert e.edge_type == "wall"
            dx = abs(e.end[0] - e.start[0])
            dy = abs(e.end[1] - e.start[1])
            assert dx < 1e-6 or dy < 1e-6, f"edge not orthogonal: {e.start} -> {e.end}"

    def test_window_edge_replaces_wall_interval(self):
        # SS17 item 16: the window graph edge spans the gap; no wall edge
        # duplicates that interval.
        wall_components, _ = extract_components(_wall_with_window_gap_mask(), "wall", min_area_px=4)
        window_components, _ = extract_components(_window_mask(), "window", min_area_px=4)
        components_dict = {"wall": wall_components, "window": window_components, "door_arc": [], "door_origin": []}
        points, _rejected, wall_edges = detect_points(
            components_dict, {}, RESOLVED_SCALE, {"min_hosted_width_px": 5.0}
        )
        aligned, _issues = align_points(points, wall_components, RESOLVED_SCALE, {}, wall_edges)
        edges, _graph_issues = connect_points(aligned, wall_edges, RESOLVED_SCALE)

        window_edges = [e for e in edges if e.edge_type == "window"]
        assert len(window_edges) == 1
        win_span = sorted([window_edges[0].start[0], window_edges[0].end[0]])
        for e in edges:
            if e.edge_type != "wall":
                continue
            wall_span = sorted([e.start[0], e.end[0]])
            # no wall edge should overlap the window's x-span
            assert wall_span[1] <= win_span[0] + 1.0 or wall_span[0] >= win_span[1] - 1.0

    def test_validate_graph_flags_orphan_window_point(self):
        p1 = GraphPoint("a", "wall_window_point", (0.0, 0.0), [], source_component_ids=[1])
        issues = validate_graph([p1], [])
        assert any(i.rule == "orphan_window_point" for i in issues)

    def test_validate_graph_does_not_flag_healthy_wall_plus_opening_pair(self):
        # Bug E regression: rules 77/78 require a window point to connect to
        # BOTH its window edge and the adjacent host wall edge - that is
        # exactly 2 edges total and must not be flagged as a conflict.
        win = GraphPoint("w", "wall_window_point", (0.0, 0.0), [], source_component_ids=[1])
        wall_pt = GraphPoint("n", "2_wall_point", (10.0, 0.0), [])
        edges = [
            GraphEdge("win_e", "window", "w", "w2", (0.0, 0.0), (5.0, 0.0)),
            GraphEdge("wall_e", "wall", "w", "n", (0.0, 0.0), (10.0, 0.0)),
        ]
        issues = validate_graph([win, wall_pt], edges)
        assert not any(i.rule == "opening_point_multiple_edges" for i in issues)

    def test_validate_graph_flags_floating_window_point(self):
        # Rules 76/104/105: a window point with an opening edge but no wall
        # edge at all is floating, not hosted on wall topology.
        win = GraphPoint("w", "wall_window_point", (0.0, 0.0), [], source_component_ids=[1])
        edges = [GraphEdge("win_e", "window", "w", "w2", (0.0, 0.0), (5.0, 0.0))]
        issues = validate_graph([win], edges)
        assert any(i.rule == "floating_window_point" for i in issues)

    def test_validate_graph_flags_real_multiple_opening_edge_conflict(self):
        # A point with two edges of its own opening type is still a genuine
        # conflict.
        win = GraphPoint("w", "wall_window_point", (0.0, 0.0), [], source_component_ids=[1])
        edges = [
            GraphEdge("win_e1", "window", "w", "w2", (0.0, 0.0), (5.0, 0.0)),
            GraphEdge("win_e2", "window", "w", "w3", (0.0, 0.0), (5.0, 5.0)),
        ]
        issues = validate_graph([win], edges)
        assert any(i.rule == "opening_point_multiple_edges" for i in issues)


# ---------------------------------------------------------------------------
# door_geometry (SS13, SS17 items 23, 24)
# ---------------------------------------------------------------------------


class TestDoorGeometry:
    def _door_origin_edge_and_points(self):
        hinge = GraphPoint(
            "hinge", "wall_door_hinge_point", (50.0, 50.0),
            [Attachment("wall", "up", "wall"), Attachment("door_origin", "down", "door_origin")],
        )
        end = GraphPoint(
            "end", "wall_door_end_point", (50.0, 80.0),
            [Attachment("wall", "down", "wall"), Attachment("door_origin", "up", "door_origin")],
        )
        edge = GraphEdge("e1", "door_origin", "hinge", "end", (50.0, 50.0), (50.0, 80.0), length_mm=700.0)
        return [hinge, end], [edge]

    def test_leaf_is_perpendicular_and_hinge_anchored(self):
        points, edges = self._door_origin_edge_and_points()
        origins, leaves, arcs = generate_door_geometry(points, edges)
        assert len(leaves) == 1
        leaf = leaves[0]
        assert leaf.hinge_point == (50.0, 50.0)
        hx, hy = leaf.hinge_point
        lx, ly = leaf.leaf_end
        assert abs(lx - hx) > 1e-3  # perpendicular to the vertical origin -> horizontal leaf

    def test_arc_spans_exactly_90_degrees_centered_on_hinge(self):
        points, edges = self._door_origin_edge_and_points()
        origins, leaves, arcs = generate_door_geometry(points, edges)
        arc = arcs[0]
        hx, hy = arc.hinge_point
        ox, oy = arc.origin_far_point
        ex, ey = arc.leaf_end
        v1, v2 = (ox - hx, oy - hy), (ex - hx, ey - hy)
        cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
        deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
        assert deg == pytest.approx(90.0, abs=1e-3)

    def test_origin_width_mm_carried_from_edge(self):
        points, edges = self._door_origin_edge_and_points()
        origins, _leaves, _arcs = generate_door_geometry(points, edges)
        assert origins[0].width_mm == 700.0


# ---------------------------------------------------------------------------
# wall_geometry / export_svg (SS14, SS17 items 25-32)
# ---------------------------------------------------------------------------


class TestWallGeometryRendering:
    def test_l_shaped_chain_produces_one_clean_mitred_polygon(self):
        segs = [((0.0, 0.0), (50.0, 0.0)), ((50.0, 0.0), (50.0, 50.0))]
        geom = segments_to_polygon(segs, half_width_px=8.0)
        assert geom is not None
        assert geom.geom_type == "Polygon"
        assert len(list(geom.exterior.coords)) == 7

    def test_wall_edges_to_primitives_normalizes_thickness_to_module(self):
        edge = GraphEdge("w1", "wall", "a", "b", (0.0, 0.0), (100.0, 0.0), thickness_px=20.0)
        walls = wall_edges_to_primitives([edge], RESOLVED_SCALE)
        assert walls[0].thickness_mm in (100.0, 200.0)

    def test_window_thickness_is_fixed_100mm_when_scale_resolved(self):
        # Rule 15: window total thickness is always 100mm once scale is
        # known, regardless of the host wall's own thickness module (100mm
        # or 200mm, rule 13) - not half of whatever the host measured.
        edge_thin_host = GraphEdge("win1", "window", "a", "b", (0.0, 0.0), (40.0, 0.0), thickness_px=10.0, length_mm=400.0)
        edge_thick_host = GraphEdge("win2", "window", "a", "b", (0.0, 0.0), (40.0, 0.0), thickness_px=20.0, length_mm=400.0)
        windows = window_edges_to_primitives([edge_thin_host, edge_thick_host], RESOLVED_SCALE)
        for w in windows:
            assert w.thickness == pytest.approx(10.0)  # 100mm at 10mm/px

    def test_window_thickness_falls_back_to_half_host_when_scale_unknown(self):
        edge = GraphEdge("win1", "window", "a", "b", (0.0, 0.0), (40.0, 0.0), thickness_px=16.0)
        windows = window_edges_to_primitives([edge], UNKNOWN_SCALE)
        assert windows[0].thickness == pytest.approx(8.0)


class TestExportSvgFinalGroups:
    def _primitives(self):
        wall = WallPrimitive("w0", (0.0, 0.0), (100.0, 0.0))
        window = WindowPrimitive("win0", center=(70.0, 0.0), width=20.0, host_wall_id="w0")
        origin = DoorOriginPrimitive("door_origin_0001", center=(20.0, 0.0), width=20.0, host_wall_id="w0")
        leaf = DoorLeafPrimitive("door_leaf_0001", hinge_point=(10.0, 0.0), width=20.0, host_wall_id="w0")
        arc = DoorArcPrimitive("door_arc_0001", hinge_point=(10.0, 0.0), origin_far_point=(30.0, 0.0), width=20.0, host_wall_id="w0")
        return wall, window, origin, leaf, arc

    def test_exactly_three_final_groups_in_order(self):
        wall, window, origin, leaf, arc = self._primitives()
        svg = build_svg(128, 128, [wall], [window], [origin], [leaf], [arc])
        positions = [svg.find('<g id="wall">'), svg.find('<g id="window">'), svg.find('<g id="door">')]
        assert all(p != -1 for p in positions)
        assert positions == sorted(positions)
        assert '<g id="floor">' not in svg

    def test_no_debug_or_retired_groups(self):
        wall, window, origin, leaf, arc = self._primitives()
        svg = build_svg(128, 128, [wall], [window], [origin], [leaf], [arc])
        for forbidden in ('<g id="floor">', '<g id="rooms">', '<g id="opening">', '<g id="icon">', '<g id="room">', 'id="debug"', "dasharray"):
            assert forbidden not in svg

    def test_door_group_contains_origin_leaf_arc(self):
        wall, window, origin, leaf, arc = self._primitives()
        svg = build_svg(128, 128, [wall], [window], [origin], [leaf], [arc])
        door_start = svg.find('<g id="door">')
        door_block = svg[door_start:]
        assert 'data-type="door_origin"' in door_block
        assert 'data-type="door_leaf"' in door_block
        assert 'data-type="door_arc"' in door_block

    def test_wall_is_black_closed_polygon(self):
        wall, window, origin, leaf, arc = self._primitives()
        svg = build_svg(128, 128, [wall], [], [], [], [])
        assert "#000000" in svg
        assert "<path" in svg

    def test_window_is_blue_closed_polygon(self):
        wall, window, origin, leaf, arc = self._primitives()
        svg = build_svg(128, 128, [wall], [window], [], [], [])
        assert "#3c78dc" in svg

    def test_door_origin_is_thin_purple_line(self):
        origin = DoorOriginPrimitive("door_origin_0001", center=(20.0, 0.0), width=20.0)
        svg = origin.to_svg()
        assert "<line" in svg
        assert "#a046b4" in svg

    def test_door_leaf_is_thin_orange_line(self):
        leaf = DoorLeafPrimitive("door_leaf_0001", hinge_point=(10.0, 0.0), width=20.0)
        svg = leaf.to_svg()
        assert "<line" in svg
        assert "#eb8c50" in svg

    def test_door_arc_is_thin_red_arc(self):
        arc = DoorArcPrimitive("door_arc_0001", hinge_point=(10.0, 0.0), origin_far_point=(30.0, 0.0), width=20.0)
        svg = arc.to_svg()
        assert "<path" in svg
        assert "#dc5a5a" in svg
        assert 'fill="none"' in svg

    def test_save_svg_creates_file(self, tmp_path):
        out = tmp_path / "out.svg"
        save_svg("<svg></svg>", out)
        assert out.exists()


# ---------------------------------------------------------------------------
# debug.py / metrics (SS17 items 33-34)
# ---------------------------------------------------------------------------


class TestDebugAndMetrics:
    def test_metrics_records_rejected_components_and_validation(self):
        from src.vectorization.debug import build_metrics
        from src.vectorization.graph_types import RejectedEvidence

        rejected = [RejectedEvidence(kind="wall_component_too_small", reason="x", class_name="wall")]
        issues = [ValidationIssue(rule="odd_window_point_count", message="x")]
        metrics = build_metrics(
            image_name="sample.png", components={"wall": []}, rejected_evidence=rejected,
            points=[], edges=[], validation_issues=issues, scale_info=RESOLVED_SCALE,
        )
        assert metrics["rejected_evidence"]["wall_component_too_small"] == 1
        assert metrics["validation_issues"][0]["rule"] == "odd_window_point_count"

    def test_debug_overlay_includes_rejected_evidence(self):
        from src.vectorization.debug import build_debug_overlay
        from src.vectorization.graph_types import RejectedEvidence

        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rejected = [RejectedEvidence(kind="window_too_narrow", reason="x", bbox=(5, 5, 10, 10))]
        overlay = build_debug_overlay(rgb, [], [], rejected, RESOLVED_SCALE)
        assert overlay.size[0] > 32
        assert overlay.size[1] >= 32


# ---------------------------------------------------------------------------
# run_mask_to_vector - end-to-end integration
# ---------------------------------------------------------------------------


class TestProcessSingleIntegration:
    def _write_image(self, tmp_path, rgb):
        from pathlib import Path

        from PIL import Image as PILImage

        path = Path(tmp_path) / "sample_000_prediction.png"
        PILImage.fromarray(rgb).save(str(path))
        return path

    def test_produces_all_required_artifacts(self, tmp_path):
        from src.vectorization.run_mask_to_vector import _scale_info_from_config, process_single

        h, w = 80, 80
        rgb = np.full((h, w, 3), CLASS_PALETTE[0], dtype=np.uint8)
        rgb[10:14, 10:60] = CLASS_PALETTE[2]
        rgb[10:60, 10:14] = CLASS_PALETTE[2]
        image_path = self._write_image(tmp_path, rgb)

        config = {"scale": {"explicit_px_to_mm": 10.0}}
        scale_info = _scale_info_from_config(config)
        out_dir = tmp_path / "out"
        result = process_single(image_path, config, scale_info, out_dir, output_filename="vector.svg")

        assert (out_dir / "vector.svg").exists()
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "debug_overlay.png").exists()
        assert len(result.walls) > 0

    def test_final_svg_has_only_wall_window_door_groups(self, tmp_path):
        from src.vectorization.run_mask_to_vector import _scale_info_from_config, process_single

        h, w = 80, 80
        rgb = np.full((h, w, 3), CLASS_PALETTE[0], dtype=np.uint8)
        rgb[10:60, 10:60] = CLASS_PALETTE[1]  # floor evidence present but must be ignored
        rgb[10:14, 10:60] = CLASS_PALETTE[2]
        rgb[10:60, 10:14] = CLASS_PALETTE[2]
        image_path = self._write_image(tmp_path, rgb)

        config = {"scale": {"explicit_px_to_mm": 10.0}}
        scale_info = _scale_info_from_config(config)
        out_dir = tmp_path / "out"
        process_single(image_path, config, scale_info, out_dir, output_filename="vector.svg")

        svg_text = (out_dir / "vector.svg").read_text(encoding="utf-8")
        assert '<g id="floor">' not in svg_text
        for required in ('<g id="wall">',):
            assert required in svg_text

    def test_incompatible_mask_raises_clear_error(self, tmp_path):
        from src.vectorization.run_mask_to_vector import _scale_info_from_config, process_single

        rgb = np.full((32, 32, 3), (200, 80, 80), dtype=np.uint8)
        image_path = self._write_image(tmp_path, rgb)

        config = {}
        scale_info = _scale_info_from_config(config)
        with pytest.raises(IncompatibleMaskError):
            process_single(image_path, config, scale_info, tmp_path / "out")
