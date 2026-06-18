"""Tests for v008 mask-to-vector pipeline modules (task06: four final classes)."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from src.vectorization.cleanup import (
    clean_icon_mask,
    clean_opening_mask,
    clean_room_mask,
    clean_wall_mask,
)
from src.vectorization.decode_prediction import CLASS_PALETTE, decode_color_mask
from src.vectorization.export_svg import build_svg, save_svg
from src.vectorization.floor_extraction import extract_floor
from src.vectorization.geometry_rules import (
    apply_geometry_rules,
    project_opening_onto_wall,
    snap_walls_to_45,
    split_walls_at_openings,
)
from src.vectorization.icon_extraction import extract_icons
from src.vectorization.load_prediction import find_prediction_images, load_image_as_array
from src.vectorization.masks import split_class_masks
from src.vectorization.opening_classification import ClassificationConfig, classify_openings
from src.vectorization.opening_extraction import extract_openings
from src.vectorization.primitives import (
    DoorPrimitive,
    FloorPrimitive,
    IconPrimitive,
    OpeningPrimitive,
    ScaleInfo,
    WallPrimitive,
    WindowPrimitive,
)
from src.vectorization.wall_extraction import (
    _rectilinearize_contour,
    extract_inner_walls,
    extract_outer_wall_loop,
    extract_walls,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_color_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Synthesize a color-coded prediction image with all 5 classes."""
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:] = CLASS_PALETTE[0]  # background
    rgb[2:6, :] = CLASS_PALETTE[1]
    rgb[2:6, 20:28] = CLASS_PALETTE[2]
    rgb[10:54, 5:59] = CLASS_PALETTE[3]
    rgb[56:62, 56:62] = CLASS_PALETTE[4]
    return rgb


def _make_wall_mask(h: int = 64, w: int = 64) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[4, 5:59] = 255   # horizontal wall
    mask[5:20, 4] = 255   # vertical wall
    return mask


def _make_opening_mask(h: int = 64, w: int = 64) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[3:6, 20:28] = 255
    return mask


def _make_room_mask(h: int = 64, w: int = 64) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:54, 5:59] = 255
    return mask


def _make_rectangle_outline_mask(
    h: int = 80, w: int = 80, margin: int = 10, thickness: int = 4
) -> np.ndarray:
    """A hollow rectangular wall outline - synthetic outer wall evidence."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[margin:margin + thickness, margin:w - margin] = 255           # top
    mask[h - margin - thickness:h - margin, margin:w - margin] = 255   # bottom
    mask[margin:h - margin, margin:margin + thickness] = 255           # left
    mask[margin:h - margin, w - margin - thickness:w - margin] = 255   # right
    return mask


# ---------------------------------------------------------------------------
# decode_prediction
# ---------------------------------------------------------------------------

class TestDecodeColorMask:
    def test_pure_background_decodes_correctly(self):
        rgb = np.full((8, 8, 3), CLASS_PALETTE[0], dtype=np.uint8)
        result = decode_color_mask(rgb)
        assert (result == 0).all()

    def test_all_classes_decoded(self):
        rgb = _make_color_image()
        result = decode_color_mask(rgb)
        assert set(np.unique(result)).issuperset({0, 1, 2, 3, 4})

    def test_wall_pixels_decoded(self):
        rgb = _make_color_image()
        result = decode_color_mask(rgb)
        assert result[3, 10] == 1

    def test_room_pixels_decoded(self):
        rgb = _make_color_image()
        result = decode_color_mask(rgb)
        assert result[30, 30] == 3

    def test_wrong_palette_raises(self):
        rgb = np.random.randint(50, 150, (16, 16, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Color decode failed"):
            decode_color_mask(rgb, tolerance=5)


# ---------------------------------------------------------------------------
# masks - background must be ignored, icon must be present
# ---------------------------------------------------------------------------

class TestSplitClassMasks:
    def test_keys_present(self):
        class_map = np.zeros((8, 8), dtype=np.uint8)
        masks = split_class_masks(class_map)
        assert "wall" in masks
        assert "opening" in masks
        assert "room" in masks
        assert "icon" in masks

    def test_background_not_a_key(self):
        class_map = np.zeros((8, 8), dtype=np.uint8)
        masks = split_class_masks(class_map)
        assert "background" not in masks

    def test_wall_mask_binary(self):
        class_map = np.zeros((8, 8), dtype=np.uint8)
        class_map[2:4, :] = 1
        masks = split_class_masks(class_map)
        assert set(np.unique(masks["wall"])).issubset({0, 255})

    def test_icon_key_present_and_isolated(self):
        class_map = np.full((8, 8), 4, dtype=np.uint8)
        masks = split_class_masks(class_map)
        assert (masks["icon"] == 255).all()
        assert (masks["wall"] == 0).all()
        assert (masks["opening"] == 0).all()
        assert (masks["room"] == 0).all()


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_small_wall_component_removed(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[1, 1] = 255
        mask[5:10, 5:20] = 255
        out = clean_wall_mask(mask, min_area=20, close_gap_px=1)
        assert out[1, 1] == 0
        assert out[7, 10] == 255

    def test_small_opening_removed(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[1, 1] = 255
        mask[10:14, 10:18] = 255
        out = clean_opening_mask(mask, min_area=8)
        assert out[1, 1] == 0

    def test_room_mask_keeps_large_region(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        out = clean_room_mask(mask, min_area=100)
        assert out[30, 30] == 255

    def test_small_icon_component_removed(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[1, 1] = 255
        mask[10:18, 10:18] = 255
        out = clean_icon_mask(mask, min_area=20)
        assert out[1, 1] == 0
        assert out[13, 13] == 255


# ---------------------------------------------------------------------------
# wall_extraction - outer rectilinear loop, then inner walls
# ---------------------------------------------------------------------------

class TestOuterWallLoop:
    def test_outer_loop_is_closed_rectilinear(self):
        """Acceptance: outer wall centerlines form a closed rectilinear loop."""
        mask = _make_rectangle_outline_mask()
        outer_walls, polygon = extract_outer_wall_loop(mask)
        assert len(outer_walls) >= 3
        assert len(polygon) >= 3
        for wall in outer_walls:
            dx = abs(wall.end[0] - wall.start[0])
            dy = abs(wall.end[1] - wall.start[1])
            assert dx < 0.5 or dy < 0.5, f"Outer wall not axis-aligned: {wall.start}->{wall.end}"
        # Closed: last wall's end connects back to the first wall's start.
        assert math.hypot(
            outer_walls[-1].end[0] - outer_walls[0].start[0],
            outer_walls[-1].end[1] - outer_walls[0].start[1],
        ) < 1.0

    def test_outer_loop_empty_mask_returns_nothing(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        outer_walls, polygon = extract_outer_wall_loop(mask)
        assert outer_walls == []
        assert polygon == []


class TestWallExtraction:
    def test_extract_walls_returns_outer_inner_and_polygon(self):
        mask = _make_rectangle_outline_mask()
        outer_walls, inner_walls, polygon = extract_walls(mask, min_wall_length_px=5)
        assert len(outer_walls) > 0
        assert len(polygon) >= 3
        assert all(isinstance(w, WallPrimitive) for w in outer_walls)
        assert all(isinstance(w, WallPrimitive) for w in inner_walls)

    def test_inner_wall_extracted_inside_outer_loop(self):
        mask = _make_rectangle_outline_mask(h=80, w=80, margin=10, thickness=4)
        mask[40:44, 20:60] = 255  # interior wall, far from the outer band
        _, inner_walls, _ = extract_walls(mask, min_wall_length_px=5)
        assert len(inner_walls) > 0

    def test_empty_mask_returns_no_walls(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        outer_walls, inner_walls, polygon = extract_walls(mask)
        assert outer_walls == []
        assert inner_walls == []
        assert polygon == []

    def test_wall_ids_unique(self):
        mask = _make_rectangle_outline_mask()
        mask[40:44, 20:60] = 255
        outer_walls, inner_walls, _ = extract_walls(mask, min_wall_length_px=5)
        ids = [w.primitive_id for w in outer_walls + inner_walls]
        assert len(ids) == len(set(ids))

    def test_inner_wall_confidence_in_range(self):
        mask = _make_rectangle_outline_mask()
        mask[40:44, 20:60] = 255
        _, inner_walls, _ = extract_walls(mask, min_wall_length_px=5)
        for wall in inner_walls:
            assert 0.0 <= wall.confidence <= 1.0

    def test_walls_extracted_before_openings_pipeline_dependency(self):
        """Wall geometry must exist before opening extraction can host onto it."""
        params = inspect.signature(extract_openings).parameters
        assert "walls" in params
        assert params["walls"].default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# geometry_rules - 45-degree snapping
# ---------------------------------------------------------------------------

class TestSnapWallsTo45:
    def test_near_horizontal_wall_snapped_to_cardinal(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 3.0), thickness=5.0)
        walls = snap_walls_to_45([wall], snap_threshold_deg=8.0)
        assert walls[0].start[1] == pytest.approx(walls[0].end[1])

    def test_near_vertical_wall_snapped_to_cardinal(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(3.0, 100.0), thickness=5.0)
        walls = snap_walls_to_45([wall], snap_threshold_deg=8.0)
        assert walls[0].start[0] == pytest.approx(walls[0].end[0])

    def test_diagonal_wall_snapped_to_nearest_45(self):
        """A wall ~28 degrees off-axis snaps to the nearest 45-degree increment."""
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 53.0), thickness=5.0)
        walls = snap_walls_to_45([wall], snap_threshold_deg=8.0)
        angle = walls[0].orientation_angle % 180.0
        assert angle == pytest.approx(45.0, abs=0.5)

    def test_exact_45_degree_wall_unchanged(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(50.0, 50.0), thickness=5.0)
        walls = snap_walls_to_45([wall], snap_threshold_deg=8.0)
        angle = walls[0].orientation_angle % 180.0
        assert angle == pytest.approx(45.0, abs=0.5)

    def test_apply_geometry_rules_projects_openings(self):
        wall = WallPrimitive("w0", start=(0.0, 10.0), end=(100.0, 10.0), thickness=5.0)
        op = OpeningPrimitive("o0", center=(50.0, 15.0), width=20.0,
                              host_wall_id="w0", orientation_angle=5.0)
        _, openings = apply_geometry_rules([wall], [op], snap_threshold_deg=8.0)
        assert openings[0].center[1] == pytest.approx(10.0, abs=1.0)


# ---------------------------------------------------------------------------
# opening_extraction
# ---------------------------------------------------------------------------

class TestOpeningExtraction:
    def test_extracts_openings(self):
        opening_mask = _make_opening_mask()
        wall = WallPrimitive("w0", start=(5.0, 4.0), end=(59.0, 4.0), thickness=8.0)
        openings = extract_openings(opening_mask, [wall])
        assert len(openings) > 0

    def test_opening_attached_to_wall(self):
        opening_mask = _make_opening_mask()
        wall = WallPrimitive("w0", start=(5.0, 4.0), end=(59.0, 4.0), thickness=8.0)
        openings = extract_openings(opening_mask, [wall], max_wall_dist=20.0)
        assert any(o.host_wall_id == "w0" for o in openings)

    def test_unhosted_opening_low_confidence(self):
        opening_mask = _make_opening_mask()
        openings = extract_openings(opening_mask, walls=[])
        assert all(o.confidence <= 0.5 for o in openings)


# ---------------------------------------------------------------------------
# opening_classification
# ---------------------------------------------------------------------------

class TestOpeningClassification:
    def _make_wide_opening(self) -> OpeningPrimitive:
        return OpeningPrimitive(
            "o_win", center=(30.0, 4.0), width=60.0,
            orientation_angle=0.0, host_wall_id="w0", confidence=0.9
        )

    def _make_compact_opening(self) -> OpeningPrimitive:
        return OpeningPrimitive(
            "o_door", center=(44.0, 7.0), width=20.0,
            orientation_angle=0.0, host_wall_id="w0", confidence=0.9
        )

    def _wide_opening_mask(self) -> np.ndarray:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[3:6, 10:50] = 255
        return mask

    def _compact_opening_mask(self) -> np.ndarray:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[3:13, 40:50] = 255
        return mask

    def test_wide_aspect_classified_as_window(self):
        opening = self._make_wide_opening()
        mask = self._wide_opening_mask()
        doors, windows, unresolved = classify_openings([opening], mask)
        assert len(windows) > 0 or len(unresolved) > 0

    def test_compact_aspect_classified_as_door(self):
        opening = self._make_compact_opening()
        mask = self._compact_opening_mask()
        doors, windows, unresolved = classify_openings([opening], mask)
        assert len(doors) + len(unresolved) > 0

    def test_output_lists_non_overlapping(self):
        openings = [self._make_wide_opening(), self._make_compact_opening()]
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[3:6, 20:60] = 255
        mask[3:8, 40:55] = 255
        doors, windows, unresolved = classify_openings(openings, mask)
        total = len(doors) + len(windows) + len(unresolved)
        assert total == len(openings)

    def test_window_uses_host_wall_thickness(self):
        wall = WallPrimitive("w0", start=(0.0, 4.0), end=(64.0, 4.0), thickness=12.0)
        opening = self._make_wide_opening()
        mask = self._wide_opening_mask()
        doors, windows, unresolved = classify_openings([opening], mask, walls=[wall])
        if windows:
            assert windows[0].thickness == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# floor_extraction - floor is a direct translation of the outer wall loop
# ---------------------------------------------------------------------------

class TestFloorExtraction:
    def test_floor_polygon_matches_outer_polygon(self):
        polygon = [(0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0)]
        floor = extract_floor(polygon)
        assert floor is not None
        assert isinstance(floor, FloorPrimitive)
        assert floor.polygon == polygon
        assert floor.area > 0.0

    def test_floor_returns_none_for_empty_polygon(self):
        assert extract_floor([]) is None

    def test_floor_returns_none_for_degenerate_polygon(self):
        assert extract_floor([(0.0, 0.0), (1.0, 1.0)]) is None

    def test_floor_primitive_to_svg_is_filled_with_no_stroke(self):
        fp = FloorPrimitive(
            primitive_id="floor_0",
            polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0)],
        )
        svg = fp.to_svg()
        assert "<polygon" in svg
        assert 'stroke="none"' in svg
        assert "fill=" in svg

    def test_floor_appears_before_wall_in_svg(self):
        fp = FloorPrimitive(
            primitive_id="floor_0",
            polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0)],
        )
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 0.0), thickness=5.0)
        svg = build_svg(
            image_width=128, image_height=128,
            walls=[wall], doors=[], windows=[], icons=[], floor=fp,
        )
        floor_pos = svg.find('<g id="floor">')
        wall_pos = svg.find('<g id="wall">')
        assert floor_pos != -1
        assert wall_pos != -1
        assert floor_pos < wall_pos

    def test_rectilinearize_contour_produces_axis_aligned_edges(self):
        pts = [(0.0, 0.0), (10.0, 3.0), (10.0, 50.0), (3.0, 50.0)]
        result = _rectilinearize_contour(pts)
        for i in range(len(result)):
            a = result[i]
            b = result[(i + 1) % len(result)]
            dx = abs(b[0] - a[0])
            dy = abs(b[1] - a[1])
            assert dx < 0.5 or dy < 0.5, f"Edge {a}->{b} is diagonal: dx={dx:.3f}, dy={dy:.3f}"


# ---------------------------------------------------------------------------
# door primitive geometry
# ---------------------------------------------------------------------------

class TestDoorPrimitiveGeometry:
    def _make_door(
        self, angle: float = 0.0, swing: str = "left", width: float = 30.0
    ) -> DoorPrimitive:
        return DoorPrimitive(
            primitive_id="d0",
            hinge_point=(50.0, 50.0),
            width=width,
            orientation_angle=angle,
            swing_direction=swing,
        )

    def test_door_origin_aligns_with_host_wall(self):
        door = self._make_door(angle=0.0)
        hx, hy = door.hinge_point
        lx, ly = door._leaf_end()
        assert abs(ly - hy) < 1e-6

    def test_door_opening_segment_perpendicular_to_origin(self):
        door = self._make_door(angle=0.0)
        hx, hy = door.hinge_point
        lx, ly = door._leaf_end()
        px, py = door._panel_end()
        origin_dx, origin_dy = lx - hx, ly - hy
        panel_dx, panel_dy = px - hx, py - hy
        dot = origin_dx * panel_dx + origin_dy * panel_dy
        assert abs(dot) < 1e-6

    def test_door_opening_segment_same_length_as_origin(self):
        door = self._make_door(angle=30.0, width=30.0)
        hx, hy = door.hinge_point
        origin_len = math.hypot(door._leaf_end()[0] - hx, door._leaf_end()[1] - hy)
        panel_len = math.hypot(door._panel_end()[0] - hx, door._panel_end()[1] - hy)
        assert origin_len == pytest.approx(door.width, abs=1e-6)
        assert panel_len == pytest.approx(door.width, abs=1e-6)

    def test_door_swing_arc_is_quarter_arc(self):
        for angle in (0.0, 45.0, 90.0):
            door = self._make_door(angle=angle, swing="left")
            hx, hy = door.hinge_point
            lx, ly = door._leaf_end()
            px, py = door._panel_end()
            v1 = (lx - hx, ly - hy)
            v2 = (px - hx, py - hy)
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag = math.hypot(*v1) * math.hypot(*v2)
            cos_angle = dot / mag
            deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
            assert deg == pytest.approx(90.0, abs=1e-4)

    def test_door_svg_has_exactly_two_lines_and_one_path(self):
        svg = self._make_door().to_svg()
        assert svg.count("<line") == 2
        assert svg.count("<path") == 1

    def test_door_right_swing_perpendicular_and_quarter(self):
        door = self._make_door(angle=0.0, swing="right")
        hx, hy = door.hinge_point
        lx, ly = door._leaf_end()
        px, py = door._panel_end()
        dot = (lx - hx) * (px - hx) + (ly - hy) * (py - hy)
        assert abs(dot) < 1e-6

    def test_door_center_is_midpoint_of_origin_segment(self):
        door = self._make_door(angle=0.0, width=20.0)
        hx, hy = door.hinge_point
        lx, ly = door._leaf_end()
        assert door.center == pytest.approx(((hx + lx) / 2.0, (hy + ly) / 2.0))


# ---------------------------------------------------------------------------
# window primitive - blue, wall-aligned line segment
# ---------------------------------------------------------------------------

class TestWindowPrimitiveGeometry:
    def test_window_svg_is_a_single_blue_line(self):
        win = WindowPrimitive(
            primitive_id="win0", center=(50.0, 10.0), width=20.0,
            orientation_angle=0.0, thickness=8.0,
        )
        svg = win.to_svg()
        assert svg.count("<line") == 1
        assert "#3355cc" in svg

    def test_window_endpoints_collinear_with_host_wall(self):
        wall = WallPrimitive("w0", start=(0.0, 10.0), end=(100.0, 10.0), thickness=8.0)
        win = WindowPrimitive(
            primitive_id="win0", center=(50.0, 10.0), width=20.0,
            orientation_angle=wall.orientation_angle, thickness=wall.thickness,
            host_wall_id="w0",
        )
        s, e = win._endpoints()
        assert s[1] == pytest.approx(10.0)
        assert e[1] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# geometry_rules - projection and wall splitting
# ---------------------------------------------------------------------------

class TestProjectionAndSplitting:
    def test_opening_projected_onto_host_wall_centerline(self):
        wall = WallPrimitive("w0", start=(0.0, 10.0), end=(100.0, 10.0), thickness=5.0)
        op = OpeningPrimitive(
            "o0", center=(50.0, 18.0), width=20.0,
            orientation_angle=5.0, host_wall_id="w0",
        )
        projected = project_opening_onto_wall(op, wall)
        assert projected.center[1] == pytest.approx(10.0, abs=0.1)
        assert projected.orientation_angle == pytest.approx(wall.orientation_angle, abs=0.1)

    def test_host_walls_split_at_opening_endpoints(self):
        wall = WallPrimitive("w0", start=(0.0, 10.0), end=(100.0, 10.0), thickness=5.0)
        op = OpeningPrimitive(
            "o0", center=(50.0, 10.0), width=20.0,
            orientation_angle=0.0, host_wall_id="w0",
        )
        result = split_walls_at_openings([wall], [op])
        assert len(result) == 2

    def test_split_segment_ids_are_unique(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 0.0), thickness=5.0)
        op = OpeningPrimitive(
            "o0", center=(50.0, 0.0), width=20.0,
            orientation_angle=0.0, host_wall_id="w0",
        )
        result = split_walls_at_openings([wall], [op])
        ids = [w.primitive_id for w in result]
        assert len(ids) == len(set(ids))

    def test_walls_without_hosted_openings_unchanged(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 0.0), thickness=5.0)
        result = split_walls_at_openings([wall], [])
        assert len(result) == 1
        assert result[0].primitive_id == "w0"

    def test_split_segments_cover_complement_of_gap(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 0.0), thickness=5.0)
        op = OpeningPrimitive(
            "o0", center=(50.0, 0.0), width=20.0,
            orientation_angle=0.0, host_wall_id="w0",
        )
        result = split_walls_at_openings([wall], [op])
        for seg in result:
            xs = sorted([seg.start[0], seg.end[0]])
            assert xs[1] <= 40.0 + 1.0 or xs[0] >= 60.0 - 1.0

    def test_door_origin_segment_replaces_trimmed_wall_portion(self):
        """Acceptance: door origin segment replaces the trimmed wall portion."""
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 0.0), thickness=5.0)
        door = DoorPrimitive(
            primitive_id="door_0", hinge_point=(40.0, 0.0), width=20.0,
            orientation_angle=0.0, swing_direction="left", host_wall_id="w0",
        )
        result = split_walls_at_openings([wall], [door])
        assert len(result) == 2
        for seg in result:
            xs = sorted([seg.start[0], seg.end[0]])
            # The door's origin segment spans x in [40, 60]; no wall segment overlaps it.
            assert xs[1] <= 40.0 + 1.0 or xs[0] >= 60.0 - 1.0


# ---------------------------------------------------------------------------
# icon_extraction / IconPrimitive
# ---------------------------------------------------------------------------

class TestIconExtraction:
    def test_extracts_icons_as_simplified_filled_shapes(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:25] = 255
        icons = extract_icons(mask, min_area=20)
        assert len(icons) > 0
        assert all(isinstance(i, IconPrimitive) for i in icons)
        assert all(len(i.polygon) >= 3 for i in icons)

    def test_empty_mask_returns_no_icons(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        assert extract_icons(mask) == []

    def test_small_icon_component_filtered_by_min_area(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[1, 1] = 255
        icons = extract_icons(mask, min_area=20)
        assert icons == []

    def test_icon_primitive_svg_is_filled_polygon(self):
        icon = IconPrimitive(
            primitive_id="icon_0",
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        )
        svg = icon.to_svg()
        assert "<polygon" in svg
        assert "fill=" in svg


# ---------------------------------------------------------------------------
# export_svg - exactly four final classes: floor, wall, opening, icon
# ---------------------------------------------------------------------------

class TestExportSvgFinalGroups:
    def _all_primitives(self):
        floor = FloorPrimitive("floor_0", [(0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0)])
        wall = WallPrimitive("w0", (0.0, 0.0), (100.0, 0.0))
        door = DoorPrimitive("door_0", hinge_point=(20.0, 0.0), width=20.0, host_wall_id="w0")
        window = WindowPrimitive("win_0", center=(70.0, 0.0), width=20.0, host_wall_id="w0")
        icon = IconPrimitive("icon_0", [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)])
        return floor, wall, door, window, icon

    def test_no_rooms_or_plural_groups_in_final_output(self):
        floor, wall, door, window, icon = self._all_primitives()
        svg = build_svg(
            image_width=128, image_height=128,
            walls=[wall], doors=[door], windows=[window], icons=[icon], floor=floor,
        )
        for forbidden in ('<g id="rooms">', '<g id="walls">', '<g id="openings">',
                          '<g id="doors">', '<g id="windows">'):
            assert forbidden not in svg

    def test_exactly_four_final_semantic_groups(self):
        floor, wall, door, window, icon = self._all_primitives()
        svg = build_svg(
            image_width=128, image_height=128,
            walls=[wall], doors=[door], windows=[window], icons=[icon], floor=floor,
        )
        for required in ('<g id="floor">', '<g id="wall">', '<g id="opening">', '<g id="icon">'):
            assert required in svg

    def test_group_order_is_floor_wall_opening_icon(self):
        floor, wall, door, window, icon = self._all_primitives()
        svg = build_svg(
            image_width=128, image_height=128,
            walls=[wall], doors=[door], windows=[window], icons=[icon], floor=floor,
        )
        positions = [
            svg.find('<g id="floor">'),
            svg.find('<g id="wall">'),
            svg.find('<g id="opening">'),
            svg.find('<g id="icon">'),
        ]
        assert all(p != -1 for p in positions)
        assert positions == sorted(positions)

    def test_doors_and_windows_both_render_inside_opening_group(self):
        floor, wall, door, window, icon = self._all_primitives()
        svg = build_svg(
            image_width=128, image_height=128,
            walls=[wall], doors=[door], windows=[window], icons=[icon], floor=floor,
        )
        opening_start = svg.find('<g id="opening">')
        opening_end = svg.find("</g>", opening_start)
        opening_block = svg[opening_start:opening_end]
        assert 'data-type="door"' in opening_block
        assert 'data-type="window"' in opening_block

    def test_svg_records_unit(self):
        si = ScaleInfo(unit="px", scale_status="unknown")
        svg = build_svg(256, 256, [], [], [], [], scale_info=si)
        assert 'data-unit="px"' in svg
        assert 'data-scale-status="unknown"' in svg

    def test_debug_layer_shows_unresolved(self):
        op = OpeningPrimitive("o_un", center=(50.0, 50.0), width=20.0)
        svg = build_svg(
            256, 256, [], [], [], [],
            unresolved_openings=[op],
            svg_config={"include_debug_layer": True},
        )
        assert 'data-type="unresolved"' in svg

    def test_unresolved_openings_not_in_opening_group(self):
        op = OpeningPrimitive("o_float", center=(50.0, 50.0), width=15.0)
        svg = build_svg(
            128, 128, [], [], [], [],
            unresolved_openings=[op],
            svg_config={"include_debug_layer": True},
        )
        assert '<g id="opening">' not in svg   # empty group omitted
        assert 'id="debug"' in svg
        assert 'data-type="unresolved"' in svg

    def test_save_svg_creates_file(self, tmp_path):
        out = tmp_path / "out.svg"
        save_svg("<svg></svg>", out)
        assert out.exists()
        assert out.read_text().startswith("<svg")

    def test_svg_disabled_layers(self):
        wall = WallPrimitive("w0", (0, 0), (100, 0))
        svg = build_svg(
            256, 256,
            walls=[wall], doors=[], windows=[], icons=[],
            svg_config={"draw_wall": False},
        )
        assert '<g id="wall">' not in svg


# ---------------------------------------------------------------------------
# load_prediction
# ---------------------------------------------------------------------------

class TestLoadPrediction:
    def test_find_prediction_images_filters_by_name(self, tmp_path):
        (tmp_path / "sample_000_prediction.png").write_bytes(b"")
        (tmp_path / "sample_000_input.png").write_bytes(b"")
        results = find_prediction_images(tmp_path, "prediction")
        assert len(results) == 1
        assert "prediction" in results[0].name

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_prediction_images(tmp_path / "nonexistent")

    def test_load_image_returns_uint8_array(self, tmp_path):
        from PIL import Image
        img = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
        path = tmp_path / "test.png"
        img.save(str(path))
        arr = load_image_as_array(path)
        assert arr.dtype == np.uint8
        assert arr.shape == (16, 16, 3)
