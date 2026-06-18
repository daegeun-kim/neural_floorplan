"""Tests for v007 component primitives."""

from __future__ import annotations

import math

import pytest

from src.vectorization.primitives import (
    DoorPrimitive,
    OpeningPrimitive,
    RoomPrimitive,
    ScaleInfo,
    WallPrimitive,
    WindowPrimitive,
)


# ---------------------------------------------------------------------------
# WallPrimitive
# ---------------------------------------------------------------------------

class TestWallPrimitive:
    def test_svg_contains_line_tag(self):
        wall = WallPrimitive("w1", start=(0.0, 0.0), end=(100.0, 0.0), thickness=8.0)
        svg = wall.to_svg()
        assert "<line" in svg
        assert 'id="w1"' in svg

    def test_bounds_include_thickness(self):
        wall = WallPrimitive("w1", start=(0.0, 0.0), end=(100.0, 0.0), thickness=10.0)
        xmin, ymin, xmax, ymax = wall.bounds()
        assert xmin < 0
        assert ymin < 0
        assert xmax > 100
        assert ymax > 0

    def test_orientation_angle_horizontal(self):
        wall = WallPrimitive("w1", start=(0.0, 0.0), end=(100.0, 0.0))
        assert wall.orientation_angle == pytest.approx(0.0)

    def test_orientation_angle_vertical(self):
        wall = WallPrimitive("w1", start=(0.0, 0.0), end=(0.0, 100.0))
        assert wall.orientation_angle == pytest.approx(90.0)

    def test_length(self):
        wall = WallPrimitive("w1", start=(0.0, 0.0), end=(30.0, 40.0))
        assert wall.length == pytest.approx(50.0)

    def test_center(self):
        wall = WallPrimitive("w1", start=(0.0, 0.0), end=(100.0, 0.0))
        assert wall.center == pytest.approx((50.0, 0.0))

    def test_transform_translate(self):
        wall = WallPrimitive("w1", start=(0.0, 0.0), end=(100.0, 0.0))
        wall.transform(dx=10.0, dy=20.0)
        assert wall.start[0] == pytest.approx(10.0)
        assert wall.start[1] == pytest.approx(20.0)
        assert wall.end[0] == pytest.approx(110.0)
        assert wall.end[1] == pytest.approx(20.0)

    def test_scale_info_defaults(self):
        wall = WallPrimitive("w1", start=(0.0, 0.0), end=(10.0, 0.0))
        assert wall.scale_info.unit == "px"
        assert wall.scale_info.scale_status == "unknown"


# ---------------------------------------------------------------------------
# OpeningPrimitive
# ---------------------------------------------------------------------------

class TestOpeningPrimitive:
    def test_svg_contains_line_tag(self):
        op = OpeningPrimitive("o1", center=(50.0, 50.0), width=20.0)
        svg = op.to_svg()
        assert "<line" in svg
        assert 'id="o1"' in svg

    def test_start_end_symmetric(self):
        op = OpeningPrimitive("o1", center=(50.0, 0.0), width=20.0, orientation_angle=0.0)
        s, e = op.start, op.end
        assert s[0] == pytest.approx(40.0)
        assert e[0] == pytest.approx(60.0)
        assert s[1] == pytest.approx(0.0)
        assert e[1] == pytest.approx(0.0)

    def test_unhosted_opening_no_wall_id(self):
        op = OpeningPrimitive("o1", center=(0.0, 0.0), width=10.0)
        assert op.host_wall_id is None

    def test_opening_type_default(self):
        op = OpeningPrimitive("o1", center=(0.0, 0.0), width=10.0)
        assert op.opening_type == "generic"

    def test_bounds(self):
        op = OpeningPrimitive("o1", center=(50.0, 50.0), width=30.0, orientation_angle=0.0)
        xmin, ymin, xmax, ymax = op.bounds()
        assert xmax - xmin == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# DoorPrimitive
# ---------------------------------------------------------------------------

class TestDoorPrimitive:
    def test_svg_contains_door_group(self):
        door = DoorPrimitive("d1", hinge_point=(0.0, 0.0), width=90.0)
        svg = door.to_svg()
        assert '<g id="d1"' in svg
        assert 'data-type="door"' in svg

    def test_svg_contains_leaf_and_arc(self):
        door = DoorPrimitive("d1", hinge_point=(0.0, 0.0), width=90.0)
        svg = door.to_svg()
        assert "<line" in svg
        assert "<path" in svg

    def test_leaf_end_horizontal_left_swing(self):
        door = DoorPrimitive("d1", hinge_point=(0.0, 0.0), width=90.0,
                             orientation_angle=0.0, swing_direction="left")
        leaf = door._leaf_end()
        assert leaf == pytest.approx((90.0, 0.0))

    def test_leaf_end_horizontal_right_swing(self):
        door = DoorPrimitive("d1", hinge_point=(0.0, 0.0), width=90.0,
                             orientation_angle=0.0, swing_direction="right")
        leaf = door._leaf_end()
        assert leaf == pytest.approx((-90.0, 0.0))

    def test_bounds_radius(self):
        door = DoorPrimitive("d1", hinge_point=(50.0, 50.0), width=90.0)
        xmin, ymin, xmax, ymax = door.bounds()
        assert xmin == pytest.approx(-40.0)
        assert xmax == pytest.approx(140.0)

    def test_transform_translate(self):
        door = DoorPrimitive("d1", hinge_point=(0.0, 0.0), width=90.0)
        door.transform(dx=10.0, dy=20.0)
        assert door.hinge_point == pytest.approx((10.0, 20.0))


# ---------------------------------------------------------------------------
# WindowPrimitive
# ---------------------------------------------------------------------------

class TestWindowPrimitive:
    """Window is a blue wall-centerline-style line (task06), not a polygon box."""

    def test_svg_is_a_blue_line(self):
        win = WindowPrimitive("win1", center=(50.0, 50.0), width=120.0)
        svg = win.to_svg()
        assert '<line id="win1"' in svg
        assert 'data-type="window"' in svg
        assert "#3355cc" in svg

    def test_endpoints_symmetric(self):
        win = WindowPrimitive("win1", center=(50.0, 0.0), width=100.0, orientation_angle=0.0)
        s, e = win._endpoints()
        assert s[0] == pytest.approx(0.0)
        assert e[0] == pytest.approx(100.0)

    def test_bounds_include_thickness(self):
        win = WindowPrimitive(
            "win1", center=(50.0, 50.0), width=100.0, orientation_angle=0.0, thickness=8.0
        )
        xmin, ymin, xmax, ymax = win.bounds()
        assert ymax - ymin == pytest.approx(win.thickness)

    def test_transform_width_scale(self):
        win = WindowPrimitive("win1", center=(0.0, 0.0), width=100.0)
        win.transform(sx=2.0)
        assert win.width == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# RoomPrimitive
# ---------------------------------------------------------------------------

class TestRoomPrimitive:
    def test_svg_contains_polygon(self):
        room = RoomPrimitive("r1", polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])
        svg = room.to_svg()
        assert "<polygon" in svg
        assert 'id="r1"' in svg

    def test_area_square(self):
        room = RoomPrimitive("r1", polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        assert room.area == pytest.approx(100.0)

    def test_area_triangle(self):
        room = RoomPrimitive("r1", polygon=[(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)])
        assert room.area == pytest.approx(50.0)

    def test_centroid_square(self):
        room = RoomPrimitive("r1", polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        cx, cy = room.centroid
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)

    def test_empty_polygon_no_svg(self):
        room = RoomPrimitive("r1", polygon=[])
        assert room.to_svg() == ""

    def test_bounds(self):
        room = RoomPrimitive("r1", polygon=[(0.0, 0.0), (50.0, 0.0), (50.0, 80.0), (0.0, 80.0)])
        xmin, ymin, xmax, ymax = room.bounds()
        assert xmin == pytest.approx(0.0)
        assert ymax == pytest.approx(80.0)

    def test_transform_translate(self):
        room = RoomPrimitive("r1", polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
        room.transform(dx=5.0, dy=5.0)
        assert room.polygon[0] == pytest.approx((5.0, 5.0))

    def test_icon_primitive_added_in_task06(self):
        from src.vectorization.primitives import __all__ as exports
        assert "IconPrimitive" in exports


# ---------------------------------------------------------------------------
# ScaleInfo
# ---------------------------------------------------------------------------

class TestScaleInfo:
    def test_defaults(self):
        si = ScaleInfo()
        assert si.unit == "px"
        assert si.scale_status == "unknown"
        assert si.scale_factor == pytest.approx(1.0)

    def test_custom(self):
        si = ScaleInfo(scale_factor=0.01, unit="m", scale_status="resolved")
        assert si.unit == "m"
