"""Tests for v008 mask-to-vector pipeline modules."""

from __future__ import annotations

import numpy as np
import pytest

from src.vectorization.cleanup import clean_opening_mask, clean_room_mask, clean_wall_mask
from src.vectorization.decode_prediction import CLASS_PALETTE, decode_color_mask
from src.vectorization.export_svg import build_svg, save_svg
from src.vectorization.geometry_rules import apply_geometry_rules, snap_walls_to_cardinal
from src.vectorization.load_prediction import find_prediction_images, load_image_as_array
from src.vectorization.masks import split_class_masks
from src.vectorization.opening_classification import ClassificationConfig, classify_openings
from src.vectorization.opening_extraction import extract_openings
from src.vectorization.primitives import (
    OpeningPrimitive,
    ScaleInfo,
    WallPrimitive,
)
from src.vectorization.room_extraction import extract_rooms
from src.vectorization.wall_extraction import extract_walls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_color_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Synthesize a color-coded prediction image with all 5 classes."""
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:] = CLASS_PALETTE[0]  # background
    # wall: top row band
    rgb[2:6, :] = CLASS_PALETTE[1]
    # opening: small patch
    rgb[2:6, 20:28] = CLASS_PALETTE[2]
    # room: large center block
    rgb[10:54, 5:59] = CLASS_PALETTE[3]
    # icon: small corner
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
        # top wall band (rows 2-5, cols away from opening)
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
# masks
# ---------------------------------------------------------------------------

class TestSplitClassMasks:
    def test_keys_present(self):
        class_map = np.zeros((8, 8), dtype=np.uint8)
        masks = split_class_masks(class_map)
        assert "wall" in masks
        assert "opening" in masks
        assert "room" in masks

    def test_wall_mask_binary(self):
        class_map = np.zeros((8, 8), dtype=np.uint8)
        class_map[2:4, :] = 1
        masks = split_class_masks(class_map)
        assert set(np.unique(masks["wall"])).issubset({0, 255})

    def test_icon_not_in_masks(self):
        class_map = np.full((8, 8), 4, dtype=np.uint8)
        masks = split_class_masks(class_map)
        assert (masks["wall"] == 0).all()
        assert (masks["opening"] == 0).all()
        assert (masks["room"] == 0).all()


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_small_wall_component_removed(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[1, 1] = 255   # 1-pixel blob → below min_area
        mask[5:10, 5:20] = 255  # larger component
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


# ---------------------------------------------------------------------------
# wall_extraction
# ---------------------------------------------------------------------------

class TestWallExtraction:
    def test_returns_wall_primitives(self):
        mask = _make_wall_mask()
        walls = extract_walls(mask, min_wall_length_px=5)
        assert len(walls) > 0
        assert all(isinstance(w, WallPrimitive) for w in walls)

    def test_empty_mask_returns_no_walls(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        walls = extract_walls(mask)
        assert walls == []

    def test_wall_ids_unique(self):
        mask = _make_wall_mask()
        walls = extract_walls(mask)
        ids = [w.primitive_id for w in walls]
        assert len(ids) == len(set(ids))

    def test_wall_confidence_in_range(self):
        mask = _make_wall_mask()
        for wall in extract_walls(mask):
            assert 0.0 <= wall.confidence <= 1.0


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
        mask[3:6, 10:50] = 255  # 4 rows × 40 cols → aspect ~10
        return mask

    def _compact_opening_mask(self) -> np.ndarray:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[3:13, 40:50] = 255  # roughly 10×10 square → aspect ~1.0
        return mask

    def test_wide_aspect_classified_as_window(self):
        opening = self._make_wide_opening()
        mask = self._wide_opening_mask()
        doors, windows, unresolved = classify_openings([opening], mask)
        assert len(windows) > 0 or len(unresolved) > 0  # may be window or unresolved

    def test_compact_aspect_classified_as_door(self):
        opening = self._make_compact_opening()
        mask = self._compact_opening_mask()
        doors, windows, unresolved = classify_openings([opening], mask)
        # compact block → door or unresolved
        assert len(doors) + len(unresolved) > 0

    def test_output_lists_non_overlapping(self):
        openings = [self._make_wide_opening(), self._make_compact_opening()]
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[3:6, 20:60] = 255
        mask[3:8, 40:55] = 255
        doors, windows, unresolved = classify_openings(openings, mask)
        total = len(doors) + len(windows) + len(unresolved)
        assert total == len(openings)


# ---------------------------------------------------------------------------
# room_extraction
# ---------------------------------------------------------------------------

class TestRoomExtraction:
    def test_extracts_rooms(self):
        mask = _make_room_mask()
        rooms = extract_rooms(mask, min_area=50)
        assert len(rooms) > 0

    def test_room_has_valid_polygon(self):
        mask = _make_room_mask()
        rooms = extract_rooms(mask, min_area=50)
        assert all(len(r.polygon) >= 3 for r in rooms)

    def test_empty_mask_returns_no_rooms(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        assert extract_rooms(mask) == []

    def test_room_confidence_in_range(self):
        mask = _make_room_mask()
        for room in extract_rooms(mask):
            assert 0.0 <= room.confidence <= 1.0


# ---------------------------------------------------------------------------
# geometry_rules
# ---------------------------------------------------------------------------

class TestGeometryRules:
    def test_near_horizontal_wall_snapped(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(100.0, 3.0), thickness=5.0)
        walls = snap_walls_to_cardinal([wall], snap_threshold_deg=8.0)
        assert walls[0].start[1] == pytest.approx(walls[0].end[1])

    def test_near_vertical_wall_snapped(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(3.0, 100.0), thickness=5.0)
        walls = snap_walls_to_cardinal([wall], snap_threshold_deg=8.0)
        assert walls[0].start[0] == pytest.approx(walls[0].end[0])

    def test_diagonal_wall_not_snapped(self):
        wall = WallPrimitive("w0", start=(0.0, 0.0), end=(50.0, 50.0), thickness=5.0)
        walls = snap_walls_to_cardinal([wall], snap_threshold_deg=8.0)
        start = walls[0].start
        end = walls[0].end
        assert start[1] != pytest.approx(end[1], abs=1.0)

    def test_apply_geometry_rules_projects_openings(self):
        wall = WallPrimitive("w0", start=(0.0, 10.0), end=(100.0, 10.0), thickness=5.0)
        op = OpeningPrimitive("o0", center=(50.0, 15.0), width=20.0,
                              host_wall_id="w0", orientation_angle=5.0)
        _, openings = apply_geometry_rules([wall], [op], snap_threshold_deg=8.0)
        assert openings[0].center[1] == pytest.approx(10.0, abs=1.0)


# ---------------------------------------------------------------------------
# export_svg
# ---------------------------------------------------------------------------

class TestExportSvg:
    def test_svg_has_correct_groups(self):
        from src.vectorization.primitives import RoomPrimitive
        svg = build_svg(
            image_width=256, image_height=256,
            walls=[WallPrimitive("w0", (0, 0), (100, 0))],
            openings=[],
            doors=[],
            windows=[],
            rooms=[RoomPrimitive("r0", [(0, 0), (50, 0), (50, 50), (0, 50)])],
        )
        assert '<g id="walls">' in svg
        assert '<g id="rooms">' in svg

    def test_svg_records_unit(self):
        si = ScaleInfo(unit="px", scale_status="unknown")
        svg = build_svg(256, 256, [], [], [], [], [], scale_info=si)
        assert 'data-unit="px"' in svg
        assert 'data-scale-status="unknown"' in svg

    def test_debug_layer_shows_unresolved(self):
        op = OpeningPrimitive("o_un", center=(50.0, 50.0), width=20.0)
        svg = build_svg(
            256, 256, [], [], [], [], [],
            unresolved_openings=[op],
            svg_config={"include_debug_layer": True},
        )
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
            walls=[wall], openings=[], doors=[], windows=[], rooms=[],
            svg_config={"draw_walls": False},
        )
        assert '<g id="walls">' not in svg


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
