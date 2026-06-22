"""Tests for v007 component primitives (active 7-class run3 scheme)."""

from __future__ import annotations

import math

import pytest

from src.vectorization.primitives import (
    DoorArcPrimitive,
    DoorLeafPrimitive,
    DoorOriginPrimitive,
    FloorPrimitive,
    OpeningPrimitive,
    OuterWallLoopPrimitive,
    ScaleInfo,
    WallPrimitive,
    WindowPrimitive,
    resolve_scale,
    snap_to_module_mm,
)
from src.vectorization.primitives.scale import DOOR_MODULES_MM, WALL_MODULES_MM


def _svg_arc_center(
    x1: float, y1: float, x2: float, y2: float, rx: float, ry: float, large_arc: int, sweep: int
) -> tuple[float, float]:
    """W3C SVG 1.1 Appendix F.6.5 endpoint-to-center conversion (no rotation).

    Used to independently verify (from the raw SVG path string, not from
    DoorArcPrimitive's own state) which of the two valid circle centers a
    given arc command actually renders with.
    """
    x1p = (x1 - x2) / 2.0
    y1p = (y1 - y2) / 2.0
    sign = -1 if large_arc == sweep else 1
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = sign * math.sqrt(max(num / den, 0.0))
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    return cxp + (x1 + x2) / 2.0, cyp + (y1 + y2) / 2.0


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

    def test_wall_type_default_unknown(self):
        wall = WallPrimitive("w1", start=(0.0, 0.0), end=(10.0, 0.0))
        assert wall.wall_type == "unknown"

    def test_wall_type_tagged_in_svg(self):
        wall = WallPrimitive("w1", start=(0.0, 0.0), end=(10.0, 0.0), wall_type="outer")
        assert 'data-wall-type="outer"' in wall.to_svg()

    def test_thickness_mm_defaults_to_none(self):
        wall = WallPrimitive("w1", start=(0.0, 0.0), end=(10.0, 0.0))
        assert wall.thickness_mm is None

    def test_shared_base_fields(self):
        wall = WallPrimitive(
            "w1", start=(0.0, 0.0), end=(10.0, 0.0), source_class_ids=[2],
        )
        assert wall.kind == "WallPrimitive"
        assert wall.source_class_ids == [2]


# ---------------------------------------------------------------------------
# OuterWallLoopPrimitive
# ---------------------------------------------------------------------------

class TestOuterWallLoopPrimitive:
    def test_is_closed_for_a_rectangle(self):
        loop = OuterWallLoopPrimitive(
            "loop1", centerline=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        )
        assert loop.is_closed()

    def test_is_not_closed_for_too_few_points(self):
        loop = OuterWallLoopPrimitive("loop1", centerline=[(0.0, 0.0), (100.0, 0.0)])
        assert not loop.is_closed()

    def test_is_not_closed_for_degenerate_line(self):
        loop = OuterWallLoopPrimitive(
            "loop1", centerline=[(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)]
        )
        assert not loop.is_closed()

    def test_to_svg_is_a_polygon_with_no_fill(self):
        loop = OuterWallLoopPrimitive(
            "loop1", centerline=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        )
        svg = loop.to_svg()
        assert "<polygon" in svg
        assert 'fill="none"' in svg

    def test_bounds(self):
        loop = OuterWallLoopPrimitive(
            "loop1", centerline=[(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)]
        )
        xmin, ymin, xmax, ymax = loop.bounds()
        assert (xmin, ymin, xmax, ymax) == pytest.approx((0.0, 0.0, 100.0, 50.0))


# ---------------------------------------------------------------------------
# OpeningPrimitive (debug-only marker)
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
# DoorOriginPrimitive / DoorLeafPrimitive / DoorArcPrimitive
# ---------------------------------------------------------------------------

class TestDoorOriginPrimitive:
    def test_svg_is_a_purple_thin_line(self):
        # task09 supersedes task08's polygon decision for door origin: it
        # must be a thin symbolic SVG line, not a closed filled polygon.
        origin = DoorOriginPrimitive("do1", center=(50.0, 0.0), width=90.0, orientation_angle=0.0)
        svg = origin.to_svg()
        assert "<line" in svg
        assert "<path" not in svg
        assert 'stroke="#a046b4"' in svg
        assert 'data-type="door_origin"' in svg

    def test_start_end_symmetric_like_window(self):
        origin = DoorOriginPrimitive("do1", center=(50.0, 0.0), width=90.0, orientation_angle=0.0)
        s, e = origin.start, origin.end
        assert s == pytest.approx((5.0, 0.0))
        assert e == pytest.approx((95.0, 0.0))

    def test_width_mm_defaults_to_none(self):
        origin = DoorOriginPrimitive("do1", center=(0.0, 0.0), width=90.0)
        assert origin.width_mm is None


class TestDoorLeafPrimitive:
    def test_leaf_end_horizontal_left_swing(self):
        leaf = DoorLeafPrimitive(
            "dl1", hinge_point=(0.0, 0.0), width=90.0, orientation_angle=0.0, swing_direction="left"
        )
        assert leaf.leaf_end == pytest.approx((0.0, 90.0))

    def test_leaf_end_horizontal_right_swing(self):
        leaf = DoorLeafPrimitive(
            "dl1", hinge_point=(0.0, 0.0), width=90.0, orientation_angle=0.0, swing_direction="right"
        )
        assert leaf.leaf_end == pytest.approx((0.0, -90.0))

    def test_leaf_is_perpendicular_to_origin_direction(self):
        # Origin runs along the wall (orientation_angle=0 -> along x-axis).
        # The leaf must point perpendicular to that (along y-axis here).
        leaf = DoorLeafPrimitive("dl1", hinge_point=(10.0, 10.0), width=80.0, orientation_angle=0.0)
        hx, hy = leaf.hinge_point
        ex, ey = leaf.leaf_end
        wall_dir = (1.0, 0.0)
        leaf_dir = (ex - hx, ey - hy)
        norm = math.hypot(*leaf_dir)
        leaf_dir = (leaf_dir[0] / norm, leaf_dir[1] / norm)
        dot = wall_dir[0] * leaf_dir[0] + wall_dir[1] * leaf_dir[1]
        assert dot == pytest.approx(0.0, abs=1e-9)

    def test_svg_is_an_orange_thin_line(self):
        # task09 supersedes task08's polygon decision for door leaf: it must
        # be a thin symbolic SVG line, not a closed filled polygon.
        leaf = DoorLeafPrimitive("dl1", hinge_point=(0.0, 0.0), width=80.0)
        svg = leaf.to_svg()
        assert "<line" in svg
        assert "<path" not in svg
        assert 'stroke="#eb8c50"' in svg
        assert 'data-type="door_leaf"' in svg

    def test_transform_translate(self):
        leaf = DoorLeafPrimitive("dl1", hinge_point=(0.0, 0.0), width=80.0)
        leaf.transform(dx=10.0, dy=20.0)
        assert leaf.hinge_point == pytest.approx((10.0, 20.0))


class TestDoorArcPrimitive:
    def test_svg_is_a_red_stroked_path(self):
        # task08 allows the arc to stay a stroked (not filled) primitive,
        # unlike origin/leaf, but it must render red.
        arc = DoorArcPrimitive(
            "da1", hinge_point=(0.0, 0.0), origin_far_point=(90.0, 0.0), width=90.0
        )
        svg = arc.to_svg()
        assert "<path" in svg
        assert 'stroke="#dc5a5a"' in svg
        assert 'data-type="door_arc"' in svg

    def test_arc_center_is_hinge_point_for_any_orientation(self):
        # The SVG arc command lets the renderer pick either of two valid
        # circle centers for a given radius/endpoints; the rendered sweep
        # flag must always select the one centered on hinge_point, for any
        # wall orientation (this was the "reversed arc" bug).
        import re

        for orientation in (0.0, 30.0, 90.0, 137.0, 200.0):
            for swing in ("left", "right"):
                arc = DoorArcPrimitive(
                    "da1", hinge_point=(20.0, 30.0), origin_far_point=(
                        20.0 + 40.0 * math.cos(math.radians(orientation)),
                        30.0 + 40.0 * math.sin(math.radians(orientation)),
                    ),
                    width=40.0, orientation_angle=orientation, swing_direction=swing,
                )
                svg = arc.to_svg()
                match = re.search(
                    r"M ([\d.\-]+) ([\d.\-]+) A ([\d.\-]+) ([\d.\-]+) 0 0 (\d) "
                    r"([\d.\-]+) ([\d.\-]+)", svg
                )
                assert match is not None
                ox, oy, rx, ry = (float(match.group(i)) for i in (1, 2, 3, 4))
                sweep = int(match.group(5))
                ex, ey = float(match.group(6)), float(match.group(7))
                cx, cy = _svg_arc_center(ox, oy, ex, ey, rx, ry, 0, sweep)
                # Tolerance accounts for the SVG path's 2-decimal coordinate
                # rounding, not just floating-point error.
                assert cx == pytest.approx(arc.hinge_point[0], abs=0.05)
                assert cy == pytest.approx(arc.hinge_point[1], abs=0.05)

    def test_arc_spans_90_degrees(self):
        arc = DoorArcPrimitive(
            "da1", hinge_point=(0.0, 0.0), origin_far_point=(90.0, 0.0), width=90.0,
            orientation_angle=0.0, swing_direction="left",
        )
        hx, hy = arc.hinge_point
        ox, oy = arc.origin_far_point
        ex, ey = arc.leaf_end
        v1 = (ox - hx, oy - hy)
        v2 = (ex - hx, ey - hy)
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
        assert angle_deg == pytest.approx(90.0, abs=1e-6)

    def test_bounds_radius(self):
        arc = DoorArcPrimitive(
            "da1", hinge_point=(50.0, 50.0), origin_far_point=(140.0, 50.0), width=90.0
        )
        xmin, ymin, xmax, ymax = arc.bounds()
        assert xmin == pytest.approx(-40.0)
        assert xmax == pytest.approx(140.0)


# ---------------------------------------------------------------------------
# WindowPrimitive
# ---------------------------------------------------------------------------

class TestWindowPrimitive:
    """Window is a blue closed polygon replacing a wall segment (task08)."""

    def test_svg_is_a_blue_closed_polygon(self):
        win = WindowPrimitive("win1", center=(50.0, 50.0), width=120.0)
        svg = win.to_svg()
        assert "<path" in svg
        assert "<line" not in svg
        assert "stroke-width" not in svg
        assert 'id="win1"' in svg
        assert 'data-type="window"' in svg
        assert "#3c78dc" in svg

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

    def test_width_mm_defaults_to_none(self):
        win = WindowPrimitive("win1", center=(0.0, 0.0), width=100.0)
        assert win.width_mm is None


# ---------------------------------------------------------------------------
# FloorPrimitive
# ---------------------------------------------------------------------------

class TestFloorPrimitive:
    def test_svg_contains_polygon(self):
        floor = FloorPrimitive("f1", polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])
        svg = floor.to_svg()
        assert "<polygon" in svg
        assert 'id="f1"' in svg

    def test_fill_color_is_pure_white(self):
        # task08: floor must render pure white, not the CNN debug palette's
        # off-white, to read as a clean architectural drawing.
        assert FloorPrimitive.FILL_COLOR == "#ffffff"

    def test_area_square(self):
        floor = FloorPrimitive("f1", polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        assert floor.area == pytest.approx(100.0)

    def test_bounds(self):
        floor = FloorPrimitive("f1", polygon=[(0.0, 0.0), (50.0, 0.0), (50.0, 80.0), (0.0, 80.0)])
        xmin, ymin, xmax, ymax = floor.bounds()
        assert xmin == pytest.approx(0.0)
        assert ymax == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# ScaleInfo / resolve_scale / snap_to_module_mm
# ---------------------------------------------------------------------------

class TestScaleInfo:
    def test_defaults(self):
        si = ScaleInfo()
        assert si.unit == "px"
        assert si.scale_status == "unknown"
        assert si.px_to_mm is None
        assert si.confidence == pytest.approx(0.0)

    def test_custom(self):
        si = ScaleInfo(unit="mm", px_to_mm=2.5, scale_status="resolved", confidence=1.0)
        assert si.unit == "mm"
        assert si.px_to_mm == pytest.approx(2.5)


class TestResolveScale:
    def test_explicit_scale_is_always_resolved(self):
        si = resolve_scale(explicit_px_to_mm=3.0)
        assert si.scale_status == "resolved"
        assert si.scale_source == "explicit_metadata"
        assert si.px_to_mm == pytest.approx(3.0)
        assert si.confidence == pytest.approx(1.0)

    def test_consistent_door_widths_estimate_a_scale(self):
        # Three doors all ~30 px wide, consistent with an 900mm module
        # at px_to_mm ~ 30 mm/px.
        si = resolve_scale(door_origin_lengths_px=[30.0, 29.0, 31.0])
        assert si.unit == "mm"
        assert si.scale_status == "estimated"
        assert si.px_to_mm is not None
        assert si.confidence >= 0.70

    def test_insufficient_evidence_falls_back_to_pixels(self):
        si = resolve_scale()
        assert si.unit == "px"
        assert si.scale_status == "unknown"
        assert si.px_to_mm is None

    def test_conflicting_door_and_wall_evidence_falls_back_to_pixels(self):
        # Doors imply one scale, walls imply a wildly different one.
        si = resolve_scale(
            door_origin_lengths_px=[30.0, 29.0, 31.0],
            wall_thickness_px=[2.0, 2.0, 2.0],
        )
        assert si.unit == "px"
        assert si.scale_status == "unknown"
        assert si.scale_source == "scale_conflict_door_vs_wall"

    def test_wall_thickness_alone_can_estimate_scale(self):
        si = resolve_scale(wall_thickness_px=[7.0, 7.0, 7.0])
        assert si.unit == "mm"
        assert si.scale_status == "estimated"
        assert si.scale_source == "wall_thickness_clustering"


class TestSnapToModuleMm:
    def test_returns_none_when_scale_unknown(self):
        si = ScaleInfo()
        mm, px = snap_to_module_mm(30.0, si, DOOR_MODULES_MM)
        assert mm is None
        assert px == pytest.approx(30.0)

    def test_returns_none_when_confidence_below_threshold(self):
        si = ScaleInfo(unit="mm", px_to_mm=20.0, scale_status="estimated", confidence=0.2)
        mm, _ = snap_to_module_mm(30.0, si, DOOR_MODULES_MM, min_confidence_for_metric=0.70)
        assert mm is None

    def test_snaps_to_nearest_door_module_when_confident(self):
        si = ScaleInfo(unit="mm", px_to_mm=20.0, scale_status="resolved", confidence=1.0)
        # 30px * 20mm/px = 600mm -> exact module.
        mm, px = snap_to_module_mm(30.0, si, DOOR_MODULES_MM)
        assert mm == pytest.approx(600.0)
        assert px == pytest.approx(30.0)

    def test_snaps_to_nearest_wall_module_when_confident(self):
        si = ScaleInfo(unit="mm", px_to_mm=20.0, scale_status="resolved", confidence=1.0)
        # 5px * 20mm/px = 100mm -> exact module.
        mm, _ = snap_to_module_mm(5.0, si, WALL_MODULES_MM)
        assert mm == pytest.approx(100.0)
