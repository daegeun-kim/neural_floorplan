"""Tests for the v008 strict 7-class mask-to-vector pipeline (run3)."""

from __future__ import annotations

import inspect
import math
import re

import numpy as np
import pytest

from src.vectorization.cleanup import (
    clean_door_arc_mask,
    clean_door_leaf_mask,
    clean_door_origin_mask,
    clean_floor_mask,
    clean_wall_mask,
    clean_window_mask,
)
from src.vectorization.decode_prediction import (
    CLASS_PALETTE,
    IncompatibleMaskError,
    decode_class_id_mask,
    decode_color_mask,
)
from src.vectorization.door_extraction import extract_doors, raw_door_origin_lengths_px
from src.vectorization.export_svg import build_svg, save_svg
from src.vectorization.floor_extraction import extract_floor
from src.vectorization.geometry_rules import (
    nearest_wall,
    project_opening_onto_wall,
    project_pixels_onto_wall,
    select_host_wall_for_opening,
    snap_walls_to_45,
    split_walls_at_openings,
)
from src.vectorization.load_prediction import find_prediction_images, load_image_as_array
from src.vectorization.masks import split_class_masks
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
)
from src.vectorization.wall_extraction import (
    _erase_outer_wall_band,
    _rectilinearize_contour,
    extract_outer_wall_loop,
    extract_walls,
)
from src.vectorization.wall_geometry import (
    merge_connected_chains,
    segments_to_polygon,
    snap_inner_endpoints_to_outer_wall_mm,
)
from src.vectorization.window_extraction import extract_windows

RESOLVED_SCALE = ScaleInfo(unit="mm", px_to_mm=10.0, scale_status="resolved", confidence=1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_color_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Synthesize a color-coded prediction image with all 7 classes."""
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:] = CLASS_PALETTE[0]  # background
    rgb[10:54, 5:59] = CLASS_PALETTE[1]   # floor
    rgb[2:6, :] = CLASS_PALETTE[2]        # wall
    rgb[2:6, 20:28] = CLASS_PALETTE[3]    # window
    rgb[56:60, 10:14] = CLASS_PALETTE[4]  # door_arc
    rgb[56:60, 20:24] = CLASS_PALETTE[5]  # door_leaf
    rgb[56:60, 30:34] = CLASS_PALETTE[6]  # door_origin
    return rgb


def _make_wall_mask(h: int = 64, w: int = 64) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[4, 5:59] = 255   # horizontal wall
    mask[5:20, 4] = 255   # vertical wall
    return mask


def _make_window_mask(h: int = 64, w: int = 64) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[3:6, 15:35] = 255  # 20px wide along the wall - a real window, not noise
    return mask


def _make_floor_mask(h: int = 64, w: int = 64) -> np.ndarray:
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
        assert set(np.unique(result)).issuperset({0, 1, 2, 3, 4, 5, 6})

    def test_wall_pixels_decoded(self):
        rgb = _make_color_image()
        result = decode_color_mask(rgb)
        assert result[3, 10] == 2

    def test_floor_pixels_decoded(self):
        rgb = _make_color_image()
        result = decode_color_mask(rgb)
        assert result[30, 30] == 1

    def test_door_origin_pixels_decoded(self):
        rgb = _make_color_image()
        result = decode_color_mask(rgb)
        assert result[57, 31] == 6

    def test_wrong_palette_raises_incompatible_mask_error(self):
        rgb = np.random.randint(50, 150, (16, 16, 3), dtype=np.uint8)
        with pytest.raises(IncompatibleMaskError, match="7-class run3 palette"):
            decode_color_mask(rgb, tolerance=5)

    def test_retired_5class_palette_raises(self):
        # The retired 5-class "opening" color (200, 80, 80) does not exist in
        # the active 7-class palette and must be rejected, not silently mapped.
        rgb = np.full((16, 16, 3), (200, 80, 80), dtype=np.uint8)
        with pytest.raises(IncompatibleMaskError):
            decode_color_mask(rgb, tolerance=5)


class TestDecodeClassIdMask:
    def test_valid_range_passes_through(self):
        mask = np.array([[0, 1, 2], [3, 4, 6]], dtype=np.uint8)
        result = decode_class_id_mask(mask)
        assert (result == mask).all()

    def test_value_above_max_class_id_raises(self):
        mask = np.array([[0, 1], [2, 99]], dtype=np.uint8)
        with pytest.raises(IncompatibleMaskError):
            decode_class_id_mask(mask)

    def test_non_2d_input_raises(self):
        mask = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(IncompatibleMaskError):
            decode_class_id_mask(mask)


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

class TestSplitClassMasks:
    def test_keys_present(self):
        class_map = np.zeros((8, 8), dtype=np.uint8)
        masks = split_class_masks(class_map)
        for key in ("floor", "wall", "window", "door_arc", "door_leaf", "door_origin"):
            assert key in masks

    def test_background_not_a_key(self):
        class_map = np.zeros((8, 8), dtype=np.uint8)
        masks = split_class_masks(class_map)
        assert "background" not in masks

    def test_no_icon_or_room_keys(self):
        class_map = np.zeros((8, 8), dtype=np.uint8)
        masks = split_class_masks(class_map)
        assert "icon" not in masks
        assert "room" not in masks
        assert "opening" not in masks

    def test_wall_mask_binary(self):
        class_map = np.zeros((8, 8), dtype=np.uint8)
        class_map[2:4, :] = 2
        masks = split_class_masks(class_map)
        assert set(np.unique(masks["wall"])).issubset({0, 255})

    def test_door_origin_isolated_from_other_classes(self):
        class_map = np.full((8, 8), 6, dtype=np.uint8)
        masks = split_class_masks(class_map)
        assert (masks["door_origin"] == 255).all()
        assert (masks["wall"] == 0).all()
        assert (masks["window"] == 0).all()


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

    def test_small_window_component_removed(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[1, 1] = 255
        mask[10:14, 10:18] = 255
        out = clean_window_mask(mask, min_area=8)
        assert out[1, 1] == 0

    def test_floor_mask_keeps_large_region(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        out = clean_floor_mask(mask, min_area=100)
        assert out[30, 30] == 255

    def test_door_origin_cleanup_does_not_close_gaps(self):
        # Two short collinear strokes with a real gap between them must stay
        # separate - closing them would corrupt the measured door width.
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10, 5:8] = 255
        mask[10, 20:23] = 255
        out = clean_door_origin_mask(mask, min_area=2)
        assert out[10, 12] == 0

    def test_door_leaf_small_noise_removed(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[1, 1] = 255
        out = clean_door_leaf_mask(mask, min_area=4)
        assert out[1, 1] == 0

    def test_door_arc_cleanup_fills_small_gaps(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        out = clean_door_arc_mask(mask, min_area=4)
        assert out[15, 15] == 255


# ---------------------------------------------------------------------------
# wall_extraction - outer rectilinear loop, then inner walls
# ---------------------------------------------------------------------------

class TestOuterWallLoop:
    def test_outer_loop_is_closed_rectilinear(self):
        mask = _make_rectangle_outline_mask()
        outer_walls, polygon, outer_loop = extract_outer_wall_loop(mask)
        assert len(outer_walls) >= 3
        assert len(polygon) >= 3
        for wall in outer_walls:
            dx = abs(wall.end[0] - wall.start[0])
            dy = abs(wall.end[1] - wall.start[1])
            assert dx < 0.5 or dy < 0.5, f"Outer wall not axis-aligned: {wall.start}->{wall.end}"
            assert wall.wall_type == "outer"
        assert isinstance(outer_loop, OuterWallLoopPrimitive)
        assert outer_loop.is_closed()

    def test_outer_loop_empty_mask_returns_nothing(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        outer_walls, polygon, outer_loop = extract_outer_wall_loop(mask)
        assert outer_walls == []
        assert polygon == []
        assert outer_loop is None



class TestWallExtraction:
    def test_extract_walls_returns_outer_inner_polygon_and_loop(self):
        mask = _make_rectangle_outline_mask()
        outer_walls, inner_walls, polygon, outer_loop = extract_walls(mask, min_wall_length_px=5)
        assert len(outer_walls) > 0
        assert len(polygon) >= 3
        assert all(isinstance(w, WallPrimitive) for w in outer_walls)
        assert all(isinstance(w, WallPrimitive) for w in inner_walls)
        assert isinstance(outer_loop, OuterWallLoopPrimitive)

    def test_inner_wall_extracted_inside_outer_loop(self):
        mask = _make_rectangle_outline_mask(h=80, w=80, margin=10, thickness=4)
        mask[40:44, 20:60] = 255  # interior wall, far from the outer band
        _, inner_walls, _, _ = extract_walls(mask, min_wall_length_px=5)
        assert len(inner_walls) > 0
        assert all(w.wall_type == "inner" for w in inner_walls)

    def test_opening_evidence_mask_bridges_outer_loop_gaps(self):
        # task08: the outer loop must be bridged by wall/opening evidence
        # (window/door masks), never by floor evidence.
        mask = _make_rectangle_outline_mask()
        opening_evidence_mask = np.zeros_like(mask)
        opening_evidence_mask[10:14, 35:45] = 255  # bridges a gap in the wall band
        outer_walls, _, polygon, _ = extract_walls(
            mask, opening_evidence_mask=opening_evidence_mask, min_wall_length_px=5
        )
        assert len(outer_walls) > 0

    def test_extract_walls_rejects_floor_mask_kwarg(self):
        # The outer loop must never be derived from the floor/background
        # border - extract_walls no longer accepts a floor_mask parameter.
        mask = _make_rectangle_outline_mask()
        floor_mask = np.zeros_like(mask)
        floor_mask[15:65, 15:65] = 255
        with pytest.raises(TypeError):
            extract_walls(mask, floor_mask=floor_mask, min_wall_length_px=5)

    def test_empty_mask_returns_no_walls(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        outer_walls, inner_walls, polygon, outer_loop = extract_walls(mask)
        assert outer_walls == []
        assert inner_walls == []
        assert polygon == []
        assert outer_loop is None

    def test_wall_ids_unique(self):
        mask = _make_rectangle_outline_mask()
        mask[40:44, 20:60] = 255
        outer_walls, inner_walls, _, _ = extract_walls(mask, min_wall_length_px=5)
        ids = [w.primitive_id for w in outer_walls + inner_walls]
        assert len(ids) == len(set(ids))

    def test_inner_wall_confidence_in_range(self):
        mask = _make_rectangle_outline_mask()
        mask[40:44, 20:60] = 255
        _, inner_walls, _, _ = extract_walls(mask, min_wall_length_px=5)
        for wall in inner_walls:
            assert 0.0 <= wall.confidence <= 1.0

    def test_walls_extracted_before_window_extraction_pipeline_dependency(self):
        """Wall geometry must exist before window extraction can host onto it."""
        params = inspect.signature(extract_windows).parameters
        assert "walls" in params
        assert params["walls"].default is inspect.Parameter.empty


class TestEraseOuterWallBand:
    def test_band_pixels_removed_but_bulge_outside_band_survives(self):
        # task10: a connected wall blob that's part of the outer wall but
        # bulges well past the synthetic erase-band thickness must now
        # survive outside the band - removing the *whole* connected
        # component (the retired _erase_claimed_wall_components behavior)
        # was exactly the task10 bug: it erased real interior walls that
        # happen to touch the exterior wall in the source mask.
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[10:14, 0:80] = 255          # the strip the band directly traces
        mask[10:40, 35:45] = 255         # a bulge, still one connected blob

        outer_polygon = [(0.0, 12.0), (80.0, 12.0)]
        remainder = _erase_outer_wall_band(mask, outer_polygon, thickness=4.0)

        assert remainder[10:14, 0:80].sum() == 0      # band strip removed
        assert remainder[25:40, 35:45].sum() > 0       # bulge far from the band survives

    def test_untouched_component_is_preserved(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[10:14, 0:80] = 255          # claimed by the band
        mask[50:54, 10:60] = 255         # a separate, untouched inner wall

        outer_polygon = [(0.0, 12.0), (80.0, 12.0)]
        remainder = _erase_outer_wall_band(mask, outer_polygon, thickness=4.0)

        assert remainder[10:14, 0:80].sum() == 0
        assert remainder[50:54, 10:60].sum() > 0

    def test_no_outer_polygon_returns_mask_unchanged(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:8, 5:15] = 255
        remainder = _erase_outer_wall_band(mask, [], thickness=4.0)
        assert np.array_equal(remainder, mask)


class TestInnerWallRecovery:
    def test_inner_wall_touching_outer_wall_is_preserved(self):
        # task10: the core bug - an interior wall fused to the outer wall in
        # one connected component must survive (only the outer band itself
        # is erased), not get wiped out along with the outer loop.
        mask = _make_rectangle_outline_mask(h=80, w=80, margin=10, thickness=4)
        mask[10:50, 40:44] = 255  # interior wall starting right at the outer band
        _, inner_walls, _, _ = extract_walls(mask, min_wall_length_px=5)
        assert len(inner_walls) > 0

    def test_inner_wall_bridges_door_origin_gap(self):
        # An interior wall with a doorway gap (no wall pixels at the door)
        # is recovered as one longer wall when door_origin_mask bridges the
        # gap - door_origin (purple) pixels are unioned into the inner-wall
        # candidate mask (task10 clarification), the same way opening
        # evidence already bridges gaps for the outer loop.
        mask = _make_rectangle_outline_mask(h=80, w=80, margin=10, thickness=4)
        mask[40:44, 20:35] = 255   # interior wall, left segment
        mask[40:44, 45:60] = 255   # interior wall, right segment (gap 35-45)
        door_origin_mask = np.zeros_like(mask)
        door_origin_mask[40:44, 35:45] = 255  # bridges the doorway gap

        _, inner_walls_no_bridge, _, _ = extract_walls(mask, min_wall_length_px=5)
        _, inner_walls_bridged, _, _ = extract_walls(
            mask, min_wall_length_px=5, door_origin_mask=door_origin_mask
        )
        max_len_no_bridge = max((w.length for w in inner_walls_no_bridge), default=0.0)
        max_len_bridged = max((w.length for w in inner_walls_bridged), default=0.0)
        assert max_len_bridged > max_len_no_bridge


# ---------------------------------------------------------------------------
# geometry_rules - 45-degree snapping, hosting, splitting
# ---------------------------------------------------------------------------

class TestSnapWallsTo45:
    def test_near_horizontal_wall_snapped_to_cardinal(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 3.0), thickness=5.0)
        walls = snap_walls_to_45([wall])
        assert walls[0].start[1] == pytest.approx(walls[0].end[1])

    def test_near_vertical_wall_snapped_to_cardinal(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(3.0, 100.0), thickness=5.0)
        walls = snap_walls_to_45([wall])
        assert walls[0].start[0] == pytest.approx(walls[0].end[0])

    def test_explicit_diagonal_wall_snapped_to_nearest_45(self):
        # ~40 degrees - within the strict diagonal_snap_deg=10 window of the
        # exact 45-degree line, so this counts as "explicit" diagonal evidence.
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 84.0), thickness=5.0)
        walls = snap_walls_to_45([wall])
        angle = walls[0].orientation_angle % 180.0
        assert angle == pytest.approx(45.0, abs=0.5)

    def test_ambiguous_angle_defaults_to_orthogonal_not_diagonal(self):
        # task09: ~28 degrees off horizontal is closer to 45 than to 0, but
        # not within the strict diagonal_snap_deg window - ambiguous evidence
        # must default to orthogonal, not the mathematically-nearer diagonal.
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 53.0), thickness=5.0)
        walls = snap_walls_to_45([wall])
        angle = walls[0].orientation_angle % 180.0
        assert angle == pytest.approx(0.0, abs=0.5)

    def test_exact_45_degree_wall_unchanged(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(50.0, 50.0), thickness=5.0)
        walls = snap_walls_to_45([wall])
        angle = walls[0].orientation_angle % 180.0
        assert angle == pytest.approx(45.0, abs=0.5)


class TestMergeConnectedChains:
    def test_joins_segments_sharing_an_endpoint(self):
        segs = [((0.0, 0.0), (50.0, 0.0)), ((50.0, 0.0), (50.0, 50.0))]
        chains = merge_connected_chains(segs)
        assert len(chains) == 1
        assert list(chains[0].coords) == [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)]

    def test_absorbs_small_floating_point_drift(self):
        segs = [((0.0, 0.0), (50.0, 0.0)), ((50.0001, -0.0002), (50.0, 50.0))]
        chains = merge_connected_chains(segs, tol=1.0)
        assert len(chains) == 1

    def test_stops_at_three_way_junction(self):
        segs = [
            ((0.0, 0.0), (50.0, 0.0)),
            ((50.0, 0.0), (50.0, 50.0)),
            ((50.0, 0.0), (100.0, 0.0)),
        ]
        chains = merge_connected_chains(segs)
        # A LineString cannot represent a branch - the junction point keeps
        # at least two of the three segments as separate chains.
        assert len(chains) >= 2

    def test_leaves_disconnected_segments_separate(self):
        segs = [((0.0, 0.0), (50.0, 0.0)), ((200.0, 200.0), (250.0, 200.0))]
        chains = merge_connected_chains(segs)
        assert len(chains) == 2


class TestSegmentsToPolygon:
    def test_l_shaped_chain_produces_one_clean_mitred_polygon(self):
        # Two perpendicular segments sharing an endpoint should buffer into
        # ONE polygon with a sharp mitred inner/outer corner, not two
        # independently-capped rectangles with a seam at the joint.
        segs = [((0.0, 0.0), (50.0, 0.0)), ((50.0, 0.0), (50.0, 50.0))]
        geom = segments_to_polygon(segs, half_width_px=8.0)
        assert geom is not None
        assert geom.geom_type == "Polygon"
        # A clean L-shaped buffer has exactly 6 corners (7 coords incl. the
        # closing repeat) - a seamed/duplicated-cap union would have more.
        assert len(list(geom.exterior.coords)) == 7

    def test_disconnected_segments_produce_separate_polygons(self):
        segs = [((0.0, 0.0), (50.0, 0.0)), ((500.0, 500.0), (550.0, 500.0))]
        geom = segments_to_polygon(segs, half_width_px=8.0)
        assert geom.geom_type == "MultiPolygon"
        assert len(geom.geoms) == 2

    def test_empty_segments_returns_none(self):
        assert segments_to_polygon([], half_width_px=8.0) is None


class TestNearestWall:
    def test_finds_closest_wall_within_max_dist(self):
        wall = WallPrimitive("w0", start=(0.0, 4.0), end=(100.0, 4.0), thickness=5.0)
        found = nearest_wall((50.0, 6.0), [wall], max_dist=20.0)
        assert found is wall

    def test_returns_none_when_too_far(self):
        wall = WallPrimitive("w0", start=(0.0, 4.0), end=(100.0, 4.0), thickness=5.0)
        assert nearest_wall((50.0, 100.0), [wall], max_dist=10.0) is None

    def test_returns_none_for_empty_wall_list(self):
        assert nearest_wall((0.0, 0.0), []) is None


class TestProjectPixelsOntoWall:
    def test_extent_matches_pixel_spread_along_wall(self):
        wall = WallPrimitive("w0", start=(0.0, 4.0), end=(100.0, 4.0), thickness=5.0)
        pixel_coords = np.array([[40.0, 3.0], [41.0, 4.0], [60.0, 5.0], [59.0, 4.0]])
        center, width, t_min, t_max = project_pixels_onto_wall(pixel_coords, wall)
        assert width == pytest.approx(20.0, abs=0.5)
        assert center[0] == pytest.approx(50.0, abs=0.5)


class TestSelectHostWallForOpening:
    def test_single_candidate_matches_nearest_wall(self):
        wall = WallPrimitive("w0", start=(0.0, 4.0), end=(100.0, 4.0), thickness=5.0)
        pixel_coords = np.array([[40.0, 3.0], [41.0, 4.0], [42.0, 5.0]])
        found = select_host_wall_for_opening(pixel_coords, [wall], max_dist=20.0)
        assert found is wall

    def test_corner_ambiguous_opening_picks_higher_probability_wall(self):
        # Two walls meeting at a corner; the opening evidence hugs the
        # horizontal wall's centerline and is oriented along it, even though
        # the vertical wall is also within max_dist of the centroid - the
        # horizontal wall must win the tie-break.
        horiz = WallPrimitive("wh", start=(0.0, 0.0), end=(100.0, 0.0), thickness=8.0)
        vert = WallPrimitive("wv", start=(0.0, 0.0), end=(0.0, 100.0), thickness=8.0)
        xs = np.arange(5.0, 25.0)
        pixel_coords = np.column_stack([xs, np.full_like(xs, 1.0)])
        found = select_host_wall_for_opening(
            pixel_coords, [horiz, vert], max_dist=20.0, corner_ambiguity_px=20.0
        )
        assert found is horiz

    def test_no_walls_within_max_dist_returns_none(self):
        wall = WallPrimitive("w0", start=(0.0, 4.0), end=(100.0, 4.0), thickness=5.0)
        pixel_coords = np.array([[40.0, 100.0]])
        assert select_host_wall_for_opening(pixel_coords, [wall], max_dist=10.0) is None


class TestInnerWallOuterAttachment:
    def test_endpoint_within_threshold_snaps_to_outer_wall(self):
        outer = [WallPrimitive("wo0", start=(0.0, 0.0), end=(1000.0, 0.0), thickness=8.0)]
        inner = [WallPrimitive("wi0", start=(500.0, 40.0), end=(500.0, 200.0), thickness=8.0)]
        snapped = snap_inner_endpoints_to_outer_wall_mm(inner, outer, RESOLVED_SCALE, threshold_mm=500.0)
        assert inner[0].start[1] == pytest.approx(0.0, abs=1e-6)
        assert "wi0" in snapped

    def test_endpoint_beyond_threshold_unchanged(self):
        outer = [WallPrimitive("wo0", start=(0.0, 0.0), end=(1000.0, 0.0), thickness=8.0)]
        inner = [WallPrimitive("wi0", start=(500.0, 8000.0), end=(500.0, 9000.0), thickness=8.0)]
        original_start = inner[0].start
        snapped = snap_inner_endpoints_to_outer_wall_mm(inner, outer, RESOLVED_SCALE, threshold_mm=500.0)
        assert inner[0].start == original_start
        assert snapped == {}

    def test_unresolved_scale_raises(self):
        outer = [WallPrimitive("wo0", start=(0.0, 0.0), end=(1000.0, 0.0), thickness=8.0)]
        inner = [WallPrimitive("wi0", start=(500.0, 40.0), end=(500.0, 200.0), thickness=8.0)]
        with pytest.raises(ValueError):
            snap_inner_endpoints_to_outer_wall_mm(inner, outer, ScaleInfo(), threshold_mm=500.0)


# ---------------------------------------------------------------------------
# window_extraction
# ---------------------------------------------------------------------------

class TestWindowExtraction:
    def test_extracts_and_hosts_window(self):
        # window_mask spans x=[15,35) -> width_px=20, well over the 300mm
        # minimum at RESOLVED_SCALE's 10mm/px (200mm)... so use a scale where
        # 20px clears 300mm: px_to_mm=20.0 -> 400mm.
        window_mask = _make_window_mask()
        wall = WallPrimitive("w0", start=(5.0, 4.0), end=(59.0, 4.0), thickness=8.0)
        si = ScaleInfo(unit="mm", px_to_mm=20.0, scale_status="resolved", confidence=1.0)
        windows, unresolved = extract_windows(window_mask, [wall], max_wall_dist=20.0, scale_info=si)
        assert len(windows) == 1
        assert windows[0].host_wall_id == "w0"
        assert unresolved == []

    def test_unhosted_window_marked_unresolved(self):
        window_mask = _make_window_mask()
        windows, unresolved = extract_windows(window_mask, walls=[])
        assert windows == []
        assert len(unresolved) == 1
        assert unresolved[0].opening_type == "unresolved_window"

    def test_too_narrow_window_evidence_is_unresolved_not_a_real_window(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[3, 30] = 255  # 1px wide - noise, not a real window
        wall = WallPrimitive("w0", start=(0.0, 4.0), end=(64.0, 4.0), thickness=8.0)
        si = ScaleInfo(unit="mm", px_to_mm=20.0, scale_status="resolved", confidence=1.0)
        windows, unresolved = extract_windows(
            mask, [wall], min_area=1, min_hosted_width_px=10.0, scale_info=si
        )
        assert windows == []
        assert len(unresolved) == 1

    def test_window_unresolved_scale_blocked_when_scale_unknown(self):
        # task10: window min-width is a real-world-mm rule with no pixel
        # fallback - an unresolved scale must block the window, not silently
        # produce a hosted window with width_mm=None.
        window_mask = _make_window_mask()
        wall = WallPrimitive("w0", start=(5.0, 4.0), end=(59.0, 4.0), thickness=8.0)
        windows, unresolved = extract_windows(window_mask, [wall], scale_info=ScaleInfo())
        assert windows == []
        assert len(unresolved) == 1
        assert unresolved[0].opening_type == "unresolved_window_scale_blocked"

    def test_width_mm_set_when_scale_resolved_with_confidence(self):
        window_mask = _make_window_mask()
        wall = WallPrimitive("w0", start=(5.0, 4.0), end=(59.0, 4.0), thickness=8.0)
        si = ScaleInfo(unit="mm", px_to_mm=100.0, scale_status="resolved", confidence=1.0)
        windows, _ = extract_windows(window_mask, [wall], scale_info=si)
        assert windows[0].width_mm is not None

    def test_window_thickness_is_half_the_host_wall_thickness(self):
        # task09: window total width is 100mm vs the wall's 200mm - exactly
        # half, regardless of the wall's measured px thickness.
        window_mask = _make_window_mask()
        wall = WallPrimitive("w0", start=(5.0, 4.0), end=(59.0, 4.0), thickness=16.0)
        si = ScaleInfo(unit="mm", px_to_mm=20.0, scale_status="resolved", confidence=1.0)
        windows, _ = extract_windows(window_mask, [wall], scale_info=si)
        assert windows[0].thickness == pytest.approx(8.0)

    def test_window_below_300mm_minimum_is_unresolved(self):
        # task10: window minimum hosted width is 300mm (architectural scale).
        window_mask = _make_window_mask()  # 20px wide
        wall = WallPrimitive("w0", start=(5.0, 4.0), end=(59.0, 4.0), thickness=8.0)
        si = ScaleInfo(unit="mm", px_to_mm=5.0, scale_status="resolved", confidence=1.0)  # 20px -> 100mm
        windows, unresolved = extract_windows(window_mask, [wall], scale_info=si)
        assert windows == []
        assert len(unresolved) == 1
        assert unresolved[0].opening_type == "unresolved_window_too_narrow_mm"

    def test_window_at_or_above_300mm_minimum_is_accepted(self):
        window_mask = _make_window_mask()  # 20px wide
        wall = WallPrimitive("w0", start=(5.0, 4.0), end=(59.0, 4.0), thickness=8.0)
        si = ScaleInfo(unit="mm", px_to_mm=20.0, scale_status="resolved", confidence=1.0)  # 20px -> 400mm
        windows, unresolved = extract_windows(window_mask, [wall], scale_info=si)
        assert len(windows) == 1
        assert unresolved == []


# ---------------------------------------------------------------------------
# door_extraction
# ---------------------------------------------------------------------------

class TestDoorExtraction:
    """task10: doors are arc-group (red) led - every fixture below includes a
    real door_arc component, since an empty door_arc_mask now means zero
    doors regardless of door_origin/door_leaf evidence."""

    MASK_SHAPE = (64, 100)

    def _host_wall(self) -> WallPrimitive:
        return WallPrimitive("w0", start=(0.0, 10.0), end=(100.0, 10.0), thickness=5.0)

    def _origin_mask(self) -> np.ndarray:
        mask = np.zeros(self.MASK_SHAPE, dtype=np.uint8)
        mask[9:12, 40:60] = 255  # door_origin component spanning x in [40, 60)
        return mask

    def _leaf_mask_near_hinge(self) -> np.ndarray:
        mask = np.zeros(self.MASK_SHAPE, dtype=np.uint8)
        mask[9:12, 38:44] = 255  # overlaps the origin's near end -> orange/purple intersection
        return mask

    def _arc_mask_near_hinge(self) -> np.ndarray:
        mask = np.zeros(self.MASK_SHAPE, dtype=np.uint8)
        mask[12:24, 36:56] = 255  # swing wedge below the wall, near the hinge end
        return mask

    def test_extracts_origin_leaf_and_arc(self):
        wall = self._host_wall()
        origins, leaves, arcs, unresolved = extract_doors(
            self._origin_mask(), self._leaf_mask_near_hinge(), self._arc_mask_near_hinge(),
            [wall], scale_info=RESOLVED_SCALE,
        )
        assert len(origins) == 1
        assert len(leaves) == 1
        assert len(arcs) == 1
        assert origins[0].host_wall_id == "w0"
        assert origins[0].width_mm in (700.0, 900.0)

    def test_hinge_chosen_near_orange_purple_intersection(self):
        wall = self._host_wall()
        origins, leaves, arcs, unresolved = extract_doors(
            self._origin_mask(), self._leaf_mask_near_hinge(), self._arc_mask_near_hinge(),
            [wall], scale_info=RESOLVED_SCALE,
        )
        hx, _hy = leaves[0].hinge_point
        assert abs(hx - 40.0) < abs(hx - 60.0)

    def test_swing_side_biased_toward_more_evidence(self):
        wall = self._host_wall()
        origins, leaves, arcs, unresolved = extract_doors(
            self._origin_mask(), self._leaf_mask_near_hinge(), self._arc_mask_near_hinge(),
            [wall], scale_info=RESOLVED_SCALE,
        )
        leaf = leaves[0]
        _hx, hy = leaf.hinge_point
        _lx, ly = leaf.leaf_end
        assert ly > hy  # leaf swings toward the side with more evidence (larger y, below the wall)

    def test_arc_origin_far_point_is_module_snapped_distance_from_hinge(self):
        wall = self._host_wall()
        origins, leaves, arcs, unresolved = extract_doors(
            self._origin_mask(), self._leaf_mask_near_hinge(), self._arc_mask_near_hinge(),
            [wall], scale_info=RESOLVED_SCALE,
        )
        arc = arcs[0]
        far = arc.origin_far_point
        hinge = arc.hinge_point
        dist = math.hypot(far[0] - hinge[0], far[1] - hinge[1])
        assert dist == pytest.approx(origins[0].width_mm / RESOLVED_SCALE.px_to_mm, abs=1e-3)
        assert far != hinge

    def test_no_door_arc_means_no_door(self):
        # task10: red door_arc connected components are the sole standard for
        # door count/location - no arc group means no door, even with
        # plenty of door_origin/door_leaf evidence nearby.
        wall = self._host_wall()
        origins, leaves, arcs, unresolved = extract_doors(
            self._origin_mask(), self._leaf_mask_near_hinge(),
            np.zeros(self.MASK_SHAPE, dtype=np.uint8), [wall], scale_info=RESOLVED_SCALE,
        )
        assert origins == []
        assert leaves == []
        assert arcs == []

    def test_door_origin_without_matching_arc_never_creates_a_door(self):
        wall = self._host_wall()
        # Arc evidence far away from the origin/leaf evidence - no pairing,
        # and no provisional host wall within reach either.
        arc_mask = np.zeros(self.MASK_SHAPE, dtype=np.uint8)
        arc_mask[60:63, 90:96] = 255
        origins, leaves, arcs, unresolved = extract_doors(
            self._origin_mask(), np.zeros(self.MASK_SHAPE, dtype=np.uint8), arc_mask,
            [wall], scale_info=RESOLVED_SCALE,
        )
        assert origins == []
        assert any(o.opening_type == "unresolved_door_origin" for o in unresolved)

    def test_unhosted_door_origin_marked_unresolved_when_no_walls(self):
        origins, leaves, arcs, unresolved = extract_doors(
            self._origin_mask(), np.zeros(self.MASK_SHAPE, dtype=np.uint8),
            self._arc_mask_near_hinge(), walls=[], scale_info=RESOLVED_SCALE,
        )
        assert origins == []
        assert any(o.opening_type == "unresolved_door_origin" for o in unresolved)

    def test_too_narrow_origin_evidence_is_unresolved(self):
        wall = self._host_wall()
        tiny_origin_mask = np.zeros(self.MASK_SHAPE, dtype=np.uint8)
        tiny_origin_mask[9:12, 41:43] = 255  # 2px wide - effectively at the hinge, no real width
        origins, leaves, arcs, unresolved = extract_doors(
            tiny_origin_mask, self._leaf_mask_near_hinge(), self._arc_mask_near_hinge(),
            [wall], min_hosted_width_px=10.0, scale_info=RESOLVED_SCALE,
        )
        assert origins == []
        assert any(o.opening_type == "unresolved_door_too_narrow" for o in unresolved)


class TestRawDoorOriginLengths:
    def test_measures_long_axis_length_per_component(self):
        mask = np.zeros((64, 100), dtype=np.uint8)
        mask[9:12, 40:60] = 255  # 20px long, 3px thick
        lengths = raw_door_origin_lengths_px(mask)
        assert len(lengths) == 1
        assert lengths[0] == pytest.approx(20.0, abs=1.0)

    def test_empty_mask_returns_empty_list(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        assert raw_door_origin_lengths_px(mask) == []


class TestDoorHingeDetection:
    def test_hinge_prefers_orange_purple_intersection_when_present(self):
        wall = WallPrimitive("w0", start=(0.0, 10.0), end=(100.0, 10.0), thickness=5.0)
        origin_mask = np.zeros((64, 100), dtype=np.uint8)
        origin_mask[9:12, 40:60] = 255
        leaf_mask = np.zeros((64, 100), dtype=np.uint8)
        leaf_mask[9:12, 38:44] = 255  # intersects the origin's near end
        arc_mask = np.zeros((64, 100), dtype=np.uint8)
        arc_mask[12:24, 36:56] = 255
        origins, leaves, arcs, unresolved = extract_doors(
            origin_mask, leaf_mask, arc_mask, [wall], scale_info=RESOLVED_SCALE,
        )
        assert len(leaves) == 1
        hx, hy = leaves[0].hinge_point
        assert 38.0 <= hx <= 46.0  # the intersection sits near x in [40, 44]
        assert hy == pytest.approx(10.0, abs=0.5)

    def test_hinge_falls_back_to_arc_geometry_when_no_intersection(self):
        wall = WallPrimitive("w0", start=(0.0, 10.0), end=(100.0, 10.0), thickness=5.0)
        origin_mask = np.zeros((64, 100), dtype=np.uint8)
        origin_mask[9:12, 40:60] = 255
        leaf_mask = np.zeros((64, 100), dtype=np.uint8)  # no overlap possible -> no intersection
        arc_mask = np.zeros((64, 100), dtype=np.uint8)
        arc_mask[12:24, 36:56] = 255
        origins, leaves, arcs, unresolved = extract_doors(
            origin_mask, leaf_mask, arc_mask, [wall], scale_info=RESOLVED_SCALE,
        )
        assert len(leaves) == 1  # still resolves a door via the arc-geometry fallback
        hx, hy = leaves[0].hinge_point
        assert hy == pytest.approx(10.0, abs=0.5)  # snapped onto the host wall

    def test_hinge_inference_disabled_leaves_evidence_unresolved(self):
        wall = WallPrimitive("w0", start=(0.0, 10.0), end=(100.0, 10.0), thickness=5.0)
        origin_mask = np.zeros((64, 100), dtype=np.uint8)
        origin_mask[9:12, 40:60] = 255
        leaf_mask = np.zeros((64, 100), dtype=np.uint8)
        arc_mask = np.zeros((64, 100), dtype=np.uint8)
        arc_mask[12:24, 36:56] = 255
        origins, leaves, arcs, unresolved = extract_doors(
            origin_mask, leaf_mask, arc_mask, [wall],
            scale_info=RESOLVED_SCALE, hinge_arc_inference_enabled=False,
        )
        assert origins == []
        assert any(o.opening_type == "unresolved_door_arc" for o in unresolved)

    def test_hinge_snaps_to_nearest_of_two_in_range_walls(self):
        # outer/inner both within hinge_snap_to_wall_max_dist_px - the closer
        # one (inner, at y=50) must win, not just whichever is in `walls` first.
        outer = WallPrimitive("wo", start=(0.0, 20.0), end=(100.0, 20.0), thickness=5.0)
        inner = WallPrimitive("wi", start=(0.0, 50.0), end=(100.0, 50.0), thickness=5.0)
        origin_mask = np.zeros((64, 100), dtype=np.uint8)
        origin_mask[49:52, 40:60] = 255
        leaf_mask = np.zeros((64, 100), dtype=np.uint8)
        leaf_mask[49:52, 38:44] = 255
        arc_mask = np.zeros((64, 100), dtype=np.uint8)
        arc_mask[40:49, 36:56] = 255
        origins, leaves, arcs, unresolved = extract_doors(
            origin_mask, leaf_mask, arc_mask, [outer, inner], scale_info=RESOLVED_SCALE,
        )
        assert len(origins) == 1
        assert origins[0].host_wall_id == "wi"


class TestDoorPairing:
    def test_unpaired_hinge_without_purple_evidence_stays_debug_only(self):
        wall = WallPrimitive("w0", start=(0.0, 10.0), end=(100.0, 10.0), thickness=5.0)
        leaf_mask = np.zeros((64, 100), dtype=np.uint8)
        leaf_mask[9:12, 38:44] = 255
        arc_mask = np.zeros((64, 100), dtype=np.uint8)
        arc_mask[12:24, 36:56] = 255
        # No door_origin evidence anywhere - an orange hinge is found/
        # inferred, but no purple far-point partner exists to pair with it.
        origin_mask = np.zeros((64, 100), dtype=np.uint8)
        origins, leaves, arcs, unresolved = extract_doors(
            origin_mask, leaf_mask, arc_mask, [wall], scale_info=RESOLVED_SCALE,
        )
        assert origins == []
        assert leaves == []
        assert any(o.opening_type == "unresolved_door_hinge" for o in unresolved)


class TestDoorModuleSnap:
    def _fixture(self):
        wall = WallPrimitive("w0", start=(0.0, 10.0), end=(100.0, 10.0), thickness=5.0)
        origin_mask = np.zeros((64, 100), dtype=np.uint8)
        origin_mask[9:12, 40:60] = 255
        leaf_mask = np.zeros((64, 100), dtype=np.uint8)
        leaf_mask[9:12, 38:44] = 255
        arc_mask = np.zeros((64, 100), dtype=np.uint8)
        arc_mask[12:24, 36:56] = 255
        return wall, origin_mask, leaf_mask, arc_mask

    def test_width_snaps_to_700_or_900_when_scale_resolved(self):
        wall, origin_mask, leaf_mask, arc_mask = self._fixture()
        origins, leaves, arcs, unresolved = extract_doors(
            origin_mask, leaf_mask, arc_mask, [wall], scale_info=RESOLVED_SCALE,
        )
        assert origins[0].width_mm in (700.0, 900.0)
        assert leaves[0].width == pytest.approx(origins[0].width_mm / RESOLVED_SCALE.px_to_mm)

    def test_600_and_800_are_not_valid_modules(self):
        wall, origin_mask, leaf_mask, arc_mask = self._fixture()
        origins, leaves, arcs, unresolved = extract_doors(
            origin_mask, leaf_mask, arc_mask, [wall], scale_info=RESOLVED_SCALE,
            door_width_modules_mm=(700.0, 900.0),
        )
        assert origins[0].width_mm not in (600.0, 800.0)


class TestScaleBlockedBehavior:
    def test_door_scale_unresolved_blocks_door_generation(self):
        wall = WallPrimitive("w0", start=(0.0, 10.0), end=(100.0, 10.0), thickness=5.0)
        origin_mask = np.zeros((64, 100), dtype=np.uint8)
        origin_mask[9:12, 40:60] = 255
        leaf_mask = np.zeros((64, 100), dtype=np.uint8)
        leaf_mask[9:12, 38:44] = 255
        arc_mask = np.zeros((64, 100), dtype=np.uint8)
        arc_mask[12:24, 36:56] = 255
        origins, leaves, arcs, unresolved = extract_doors(
            origin_mask, leaf_mask, arc_mask, [wall], scale_info=ScaleInfo(),
        )
        assert origins == []
        assert any(o.opening_type == "unresolved_door_scale_blocked" for o in unresolved)

    def test_window_scale_unresolved_blocks_window_generation(self):
        window_mask = _make_window_mask()
        wall = WallPrimitive("w0", start=(5.0, 4.0), end=(59.0, 4.0), thickness=8.0)
        windows, unresolved = extract_windows(window_mask, [wall], scale_info=ScaleInfo())
        assert windows == []
        assert unresolved[0].opening_type == "unresolved_window_scale_blocked"

    def test_inner_wall_attach_scale_unresolved_raises(self):
        outer = [WallPrimitive("wo0", start=(0.0, 0.0), end=(1000.0, 0.0), thickness=8.0)]
        inner = [WallPrimitive("wi0", start=(500.0, 40.0), end=(500.0, 200.0), thickness=8.0)]
        with pytest.raises(ValueError):
            snap_inner_endpoints_to_outer_wall_mm(inner, outer, ScaleInfo(), threshold_mm=500.0)


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
            walls=[wall], windows=[], door_origins=[], door_leaves=[], door_arcs=[], floor=fp,
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
# door primitive geometry (origin/leaf/arc)
# ---------------------------------------------------------------------------

class TestDoorPrimitiveGeometry:
    def test_leaf_is_perpendicular_to_wall_direction(self):
        leaf = DoorLeafPrimitive("dl0", hinge_point=(50.0, 50.0), width=30.0, orientation_angle=0.0)
        hx, hy = leaf.hinge_point
        lx, ly = leaf.leaf_end
        assert abs(lx - hx) < 1e-6  # purely vertical displacement for a horizontal wall

    def test_leaf_length_equals_width(self):
        leaf = DoorLeafPrimitive("dl0", hinge_point=(50.0, 50.0), width=30.0, orientation_angle=30.0)
        hx, hy = leaf.hinge_point
        lx, ly = leaf.leaf_end
        assert math.hypot(lx - hx, ly - hy) == pytest.approx(30.0, abs=1e-6)

    def test_arc_spans_quarter_circle(self):
        for angle in (0.0, 45.0, 90.0):
            arc = DoorArcPrimitive(
                "da0", hinge_point=(50.0, 50.0), origin_far_point=(50.0 + 30 * math.cos(math.radians(angle)),
                                                                    50.0 + 30 * math.sin(math.radians(angle))),
                width=30.0, orientation_angle=angle, swing_direction="left",
            )
            hx, hy = arc.hinge_point
            ox, oy = arc.origin_far_point
            ex, ey = arc.leaf_end
            v1, v2 = (ox - hx, oy - hy), (ex - hx, ey - hy)
            cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
            deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
            assert deg == pytest.approx(90.0, abs=1e-4)

    def test_origin_replaces_wall_segment_shape_matches_window(self):
        origin = DoorOriginPrimitive("do0", center=(50.0, 10.0), width=20.0, orientation_angle=0.0)
        assert origin.start == pytest.approx((40.0, 10.0))
        assert origin.end == pytest.approx((60.0, 10.0))

    def test_svg_two_thin_lines_and_one_stroked_arc(self):
        # task09 supersedes task08: origin and leaf are thin symbolic SVG
        # lines again (not closed filled polygons); the arc stays a stroked
        # path, as it always has been.
        origin = DoorOriginPrimitive("do0", center=(50.0, 0.0), width=20.0)
        leaf = DoorLeafPrimitive("dl0", hinge_point=(40.0, 0.0), width=20.0)
        arc = DoorArcPrimitive("da0", hinge_point=(40.0, 0.0), origin_far_point=(60.0, 0.0), width=20.0)
        combined = origin.to_svg() + leaf.to_svg() + arc.to_svg()
        assert combined.count("<line") == 2
        assert combined.count("<path") == 1
        assert combined.count('fill="none"') == 1  # only the arc is unfilled


# ---------------------------------------------------------------------------
# window primitive - blue, wall-aligned line segment
# ---------------------------------------------------------------------------

class TestWindowPrimitiveGeometry:
    def test_window_svg_is_a_single_blue_closed_polygon(self):
        win = WindowPrimitive(
            primitive_id="win0", center=(50.0, 10.0), width=20.0,
            orientation_angle=0.0, thickness=8.0,
        )
        svg = win.to_svg()
        assert svg.count("<line") == 0
        assert svg.count("<path") == 1
        assert "#3c78dc" in svg

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

    def test_window_host_walls_split_at_endpoints(self):
        wall = WallPrimitive("w0", start=(0.0, 10.0), end=(100.0, 10.0), thickness=5.0)
        window = WindowPrimitive("win0", center=(50.0, 10.0), width=20.0,
                                  orientation_angle=0.0, host_wall_id="w0")
        result = split_walls_at_openings([wall], [window])
        assert len(result) == 2

    def test_split_segment_ids_are_unique(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 0.0), thickness=5.0)
        window = WindowPrimitive("win0", center=(50.0, 0.0), width=20.0,
                                  orientation_angle=0.0, host_wall_id="w0")
        result = split_walls_at_openings([wall], [window])
        ids = [w.primitive_id for w in result]
        assert len(ids) == len(set(ids))

    def test_walls_without_hosted_openings_unchanged(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 0.0), thickness=5.0)
        result = split_walls_at_openings([wall], [])
        assert len(result) == 1
        assert result[0].primitive_id == "w0"

    def test_split_segments_cover_complement_of_gap(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 0.0), thickness=5.0)
        window = WindowPrimitive("win0", center=(50.0, 0.0), width=20.0,
                                  orientation_angle=0.0, host_wall_id="w0")
        result = split_walls_at_openings([wall], [window])
        for seg in result:
            xs = sorted([seg.start[0], seg.end[0]])
            assert xs[1] <= 40.0 + 1.0 or xs[0] >= 60.0 - 1.0

    def test_door_origin_segment_replaces_trimmed_wall_portion(self):
        """Acceptance: door origin segment replaces the trimmed wall portion."""
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 0.0), thickness=5.0)
        origin = DoorOriginPrimitive(
            "door_origin_0", center=(50.0, 0.0), width=20.0,
            orientation_angle=0.0, host_wall_id="w0",
        )
        result = split_walls_at_openings([wall], [origin])
        assert len(result) == 2
        for seg in result:
            xs = sorted([seg.start[0], seg.end[0]])
            assert xs[1] <= 40.0 + 1.0 or xs[0] >= 60.0 - 1.0


# ---------------------------------------------------------------------------
# export_svg - floor, wall, window, door, debug
# ---------------------------------------------------------------------------

class TestExportSvgFinalGroups:
    def _all_primitives(self):
        floor = FloorPrimitive("floor_0", [(0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0)])
        wall = WallPrimitive("w0", (0.0, 0.0), (100.0, 0.0))
        origin = DoorOriginPrimitive("door_origin_0001", center=(20.0, 0.0), width=20.0, host_wall_id="w0")
        leaf = DoorLeafPrimitive("door_leaf_0001", hinge_point=(10.0, 0.0), width=20.0, host_wall_id="w0")
        arc = DoorArcPrimitive("door_arc_0001", hinge_point=(10.0, 0.0), origin_far_point=(30.0, 0.0),
                                width=20.0, host_wall_id="w0")
        window = WindowPrimitive("win_0", center=(70.0, 0.0), width=20.0, host_wall_id="w0")
        return floor, wall, origin, leaf, arc, window

    def test_no_retired_class_groups_in_final_output(self):
        floor, wall, origin, leaf, arc, window = self._all_primitives()
        svg = build_svg(
            image_width=128, image_height=128,
            walls=[wall], windows=[window],
            door_origins=[origin], door_leaves=[leaf], door_arcs=[arc], floor=floor,
        )
        for forbidden in ('<g id="rooms">', '<g id="walls">', '<g id="opening">',
                          '<g id="icon">', '<g id="room">'):
            assert forbidden not in svg

    def test_exactly_four_final_semantic_groups(self):
        floor, wall, origin, leaf, arc, window = self._all_primitives()
        svg = build_svg(
            image_width=128, image_height=128,
            walls=[wall], windows=[window],
            door_origins=[origin], door_leaves=[leaf], door_arcs=[arc], floor=floor,
        )
        for required in ('<g id="floor">', '<g id="wall">', '<g id="window">', '<g id="door">'):
            assert required in svg

    def test_group_order_is_floor_wall_window_door(self):
        floor, wall, origin, leaf, arc, window = self._all_primitives()
        svg = build_svg(
            image_width=128, image_height=128,
            walls=[wall], windows=[window],
            door_origins=[origin], door_leaves=[leaf], door_arcs=[arc], floor=floor,
        )
        positions = [
            svg.find('<g id="floor">'),
            svg.find('<g id="wall">'),
            svg.find('<g id="window">'),
            svg.find('<g id="door">'),
        ]
        assert all(p != -1 for p in positions)
        assert positions == sorted(positions)

    def test_door_group_contains_origin_leaf_and_arc(self):
        floor, wall, origin, leaf, arc, window = self._all_primitives()
        svg = build_svg(
            image_width=128, image_height=128,
            walls=[wall], windows=[window],
            door_origins=[origin], door_leaves=[leaf], door_arcs=[arc], floor=floor,
        )
        door_start = svg.find('<g id="door">')
        door_end = svg.find("\n  </g>", door_start)
        door_block = svg[door_start:door_end]
        assert 'data-type="door_origin"' in door_block
        assert 'data-type="door_leaf"' in door_block
        assert 'data-type="door_arc"' in door_block

    def test_svg_records_unit_and_scale_metadata(self):
        si = ScaleInfo(unit="mm", px_to_mm=2.5, scale_status="estimated",
                        scale_source="door_origin_width_clustering", confidence=0.8)
        svg = build_svg(256, 256, [], [], [], [], [], scale_info=si)
        assert 'data-unit="mm"' in svg
        assert 'data-scale-status="estimated"' in svg
        assert 'data-px-to-mm="2.5"' in svg
        assert 'data-scale-source="door_origin_width_clustering"' in svg

    def test_build_svg_has_no_unresolved_or_debug_parameter(self):
        # task08: debug/unresolved evidence must never reach the final SVG -
        # build_svg no longer accepts an `unresolved` argument at all.
        with pytest.raises(TypeError):
            build_svg(
                256, 256, [], [], [], [], [],
                unresolved=[OpeningPrimitive("o_un", center=(50.0, 50.0), width=20.0)],
            )

    def test_no_debug_group_in_final_svg(self):
        floor, wall, origin, leaf, arc, window = self._all_primitives()
        svg = build_svg(
            128, 128,
            walls=[wall], windows=[window],
            door_origins=[origin], door_leaves=[leaf], door_arcs=[arc], floor=floor,
        )
        assert 'id="debug"' not in svg
        assert "unresolved" not in svg
        assert "dasharray" not in svg
        assert "#ff8800" not in svg

    def test_save_svg_creates_file(self, tmp_path):
        out = tmp_path / "out.svg"
        save_svg("<svg></svg>", out)
        assert out.exists()
        assert out.read_text().startswith("<svg")

    def test_svg_disabled_layers(self):
        wall = WallPrimitive("w0", (0, 0), (100, 0))
        svg = build_svg(
            256, 256,
            walls=[wall], windows=[], door_origins=[], door_leaves=[], door_arcs=[],
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


# ---------------------------------------------------------------------------
# run_mask_to_vector - end-to-end integration on a synthetic prediction image
# ---------------------------------------------------------------------------

class TestProcessSingleIntegration:
    def _write_synthetic_prediction(self, tmp_path):
        from pathlib import Path
        from PIL import Image as PILImage

        rgb = _make_color_image(h=80, w=120)
        path = tmp_path / "sample_000_prediction.png"
        PILImage.fromarray(rgb).save(str(path))
        return Path(path)

    def test_process_single_produces_all_required_artifacts(self, tmp_path):
        from src.vectorization.run_mask_to_vector import _scale_info_from_config, process_single

        image_path = self._write_synthetic_prediction(tmp_path)
        config = {
            "cleanup": {"close_wall_gap_px": 1},
            "walls": {"min_wall_length_px": 3},
        }
        scale_info = _scale_info_from_config(config)
        out_dir = tmp_path / "out"
        process_single(image_path, config, scale_info, out_dir, output_filename="vector.svg")

        assert (out_dir / "vector.svg").exists()
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "debug_overlay.png").exists()

        svg_text = (out_dir / "vector.svg").read_text(encoding="utf-8")
        assert "<svg" in svg_text
        assert 'data-scale-status=' in svg_text

    def test_metrics_json_records_scale_and_counts(self, tmp_path):
        import json

        from src.vectorization.run_mask_to_vector import _scale_info_from_config, process_single

        image_path = self._write_synthetic_prediction(tmp_path)
        config = {}
        scale_info = _scale_info_from_config(config)
        out_dir = tmp_path / "out"
        process_single(image_path, config, scale_info, out_dir, output_filename="vector.svg")

        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        assert "walls" in metrics
        assert "scale" in metrics
        assert metrics["scale"]["scale_status"] in ("resolved", "estimated", "unknown")

    def test_outer_wall_not_derived_from_floor_evidence(self, tmp_path):
        # task08: the CNN's floor class is unreliable, so the outer wall
        # envelope must never be traced from the floor/background border.
        # Build a sample where floor evidence spills out far beyond the
        # actual wall rectangle (mimicking floor over-segmentation) and
        # confirm the rendered wall polygon still hugs the small wall
        # rectangle, not the inflated floor region.
        from pathlib import Path

        from PIL import Image as PILImage

        h, w = 100, 100
        rgb = np.full((h, w, 3), CLASS_PALETTE[0], dtype=np.uint8)
        rgb[5:95, 5:95] = CLASS_PALETTE[1]  # floor spills across almost the whole image
        # A small wall rectangle, much smaller than the floor region above.
        rgb[20:24, 20:80] = CLASS_PALETTE[2]
        rgb[76:80, 20:80] = CLASS_PALETTE[2]
        rgb[20:80, 20:24] = CLASS_PALETTE[2]
        rgb[20:80, 76:80] = CLASS_PALETTE[2]

        path = Path(tmp_path) / "sample_000_prediction.png"
        PILImage.fromarray(rgb).save(str(path))

        from src.vectorization.run_mask_to_vector import _scale_info_from_config, process_single

        config = {"walls": {"min_wall_length_px": 3}}
        scale_info = _scale_info_from_config(config)
        out_dir = tmp_path / "out"
        process_single(path, config, scale_info, out_dir, output_filename="vector.svg")

        svg_text = (out_dir / "vector.svg").read_text(encoding="utf-8")
        wall_start = svg_text.find('<g id="wall">')
        wall_end = svg_text.find("</g>", wall_start)
        wall_block = svg_text[wall_start:wall_end]

        path_d = wall_block.split('<path d="')[1].split('" fill=')[0]
        coords = [float(v) for v in re.findall(r"-?\d+\.\d+", path_d)]
        xs, ys = coords[0::2], coords[1::2]

        # The wall polygon must stay close to the 20-80 wall rectangle, not
        # balloon out to the 5-95 floor region.
        assert min(xs) >= 9
        assert max(xs) <= 91
        assert min(ys) >= 9
        assert max(ys) <= 91

    def test_incompatible_mask_raises_clear_error(self, tmp_path):
        from PIL import Image as PILImage

        from src.vectorization.run_mask_to_vector import _scale_info_from_config, process_single

        # Retired 5-class "opening" color - must be rejected, not silently vectorized.
        rgb = np.full((32, 32, 3), (200, 80, 80), dtype=np.uint8)
        path = tmp_path / "old_5class_prediction.png"
        PILImage.fromarray(rgb).save(str(path))

        config = {}
        scale_info = _scale_info_from_config(config)
        with pytest.raises(IncompatibleMaskError):
            process_single(path, config, scale_info, tmp_path / "out")
