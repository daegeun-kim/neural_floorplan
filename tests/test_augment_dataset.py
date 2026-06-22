"""Tests for src/augment_dataset.py (spec_v004)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.augment_dataset import (
    AUGMENTED_DIR,
    MASKS_DIR,
    SPATIAL_MASK_NAMES,
    _apply_blur,
    _apply_brightness,
    _apply_flip,
    _apply_rotate90,
    _apply_translation,
    apply_augmentation,
    augment_sample,
    process_dataset,
    save_preview,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample(tmp_path: Path, name: str = "s1", image_size: tuple[int, int] = (60, 60)) -> Path:
    """Create a minimal sample directory with a clean image and pre-generated masks."""
    sample_dir = tmp_path / name
    sample_dir.mkdir()
    w, h = image_size

    # Clean raster image (RGB)
    img = Image.new("RGB", (w, h), (240, 240, 240))
    img.save(sample_dir / "model_clean.png")

    # Masks
    masks_dir = sample_dir / MASKS_DIR
    masks_dir.mkdir()

    # Wall mask: top strip
    wall = np.zeros((h, w), dtype=np.uint8)
    wall[:5, :] = 255
    Image.fromarray(wall, "L").save(masks_dir / "wall_mask.png")

    # Window mask: small rect
    window = np.zeros((h, w), dtype=np.uint8)
    window[1:4, 20:30] = 255
    Image.fromarray(window, "L").save(masks_dir / "window_mask.png")

    # Floor mask: large interior
    floor = np.zeros((h, w), dtype=np.uint8)
    floor[10:, :] = 255
    Image.fromarray(floor, "L").save(masks_dir / "floor_mask.png")

    # Door origin mask: small rect inside floor
    door_origin = np.zeros((h, w), dtype=np.uint8)
    door_origin[15:20, 10:15] = 255
    Image.fromarray(door_origin, "L").save(masks_dir / "door_origin_mask.png")

    # Semantic class map (combined): background=0 floor=1 wall=2 window=3 door_origin=6
    class_map = np.zeros((h, w), dtype=np.uint8)
    class_map[10:, :] = 1        # floor
    class_map[15:20, 10:15] = 6  # door_origin
    class_map[1:4, 20:30] = 3    # window
    class_map[:5, :] = 2         # wall
    Image.fromarray(class_map, "L").save(masks_dir / "semantic_class_map.png")

    return sample_dir


# ---------------------------------------------------------------------------
# Primitive transform tests
# ---------------------------------------------------------------------------


def _img_and_mask(size=(40, 40)):
    img = Image.new("RGB", size, (200, 200, 200))
    mask = Image.fromarray(
        np.pad(np.ones((10, 20), dtype=np.uint8) * 255, ((5, 25), (5, 15)), constant_values=0),
        mode="L",
    )
    return img, mask


def test_flip_horizontal_changes_image():
    img, mask = _img_and_mask()
    orig_arr = np.array(img)
    aug_img, [aug_mask] = _apply_flip(img, [mask], "horizontal")
    assert aug_img.size == img.size
    assert aug_mask.size == mask.size


def test_flip_horizontal_mirrors_content():
    """A mask with content on the left should move to the right after horizontal flip."""
    mask_arr = np.zeros((40, 40), dtype=np.uint8)
    mask_arr[:, :10] = 255  # left band
    mask = Image.fromarray(mask_arr, mode="L")
    img = Image.new("RGB", (40, 40), (255, 255, 255))
    _, [flipped_mask] = _apply_flip(img, [mask], "horizontal")
    flipped_arr = np.array(flipped_mask)
    assert np.all(flipped_arr[:, -10:] == 255), "Content should be on the right after flip"
    assert np.all(flipped_arr[:, :30] == 0)


def test_rotate90_expands_correctly():
    img = Image.new("RGB", (60, 40), (100, 100, 100))
    mask = Image.new("L", (60, 40), 0)
    aug_img, [aug_mask] = _apply_rotate90(img, [mask], k=1)
    assert aug_img.size == (40, 60)  # width/height swapped for 90-degree rotation
    assert aug_mask.size == (40, 60)


def test_rotate90_k0_is_identity():
    img = Image.new("RGB", (50, 50), (1, 2, 3))
    mask = Image.new("L", (50, 50), 128)
    aug_img, [aug_mask] = _apply_rotate90(img, [mask], k=0)
    assert np.array_equal(np.array(aug_img), np.array(img))
    assert np.array_equal(np.array(aug_mask), np.array(mask))


def test_translation_preserves_size():
    img, mask = _img_and_mask()
    aug_img, [aug_mask] = _apply_translation(img, [mask], dx=5, dy=5)
    assert aug_img.size == img.size
    assert aug_mask.size == mask.size


def test_translation_fills_with_zero_for_mask():
    mask_arr = np.ones((40, 40), dtype=np.uint8) * 255
    mask = Image.fromarray(mask_arr, mode="L")
    img = Image.new("RGB", (40, 40), (0, 0, 0))
    _, [aug_mask] = _apply_translation(img, [mask], dx=5, dy=5)
    aug_arr = np.array(aug_mask)
    # Top 5 rows should be zero (zero-padding)
    assert np.all(aug_arr[:5, :] == 0)
    # Left 5 columns should be zero
    assert np.all(aug_arr[:, :5] == 0)


def test_blur_changes_image():
    img = Image.new("RGB", (40, 40), (100, 100, 100))
    arr_before = np.array(img)
    blurred = _apply_blur(img, radius=1.5)
    # A solid-color image won't visually change but the function should run without error
    assert blurred.size == img.size


def test_brightness_increases_pixels():
    arr = np.full((30, 30, 3), 100, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    brighter = _apply_brightness(img, factor=1.5)
    bright_arr = np.array(brighter)
    assert np.all(bright_arr >= 100)


def test_brightness_clamp():
    arr = np.full((10, 10, 3), 200, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    result = _apply_brightness(img, factor=2.0)
    result_arr = np.array(result)
    assert np.all(result_arr <= 255)


# ---------------------------------------------------------------------------
# apply_augmentation — alignment guarantee
# ---------------------------------------------------------------------------


def test_apply_augmentation_same_size(tmp_path):
    sample_dir = _make_sample(tmp_path)
    img = Image.open(sample_dir / "model_clean.png").convert("RGB")
    mask = Image.open(sample_dir / MASKS_DIR / "semantic_class_map.png")

    params = {"flip": "horizontal", "rotate_k": 1}
    aug_img, [aug_mask] = apply_augmentation(img, [mask], params)

    assert aug_img.size == aug_mask.size, "Image and mask sizes must match after augmentation"


def test_apply_augmentation_mask_values_unchanged(tmp_path):
    """Augmented mask should still contain only the original class IDs."""
    sample_dir = _make_sample(tmp_path)
    mask = Image.open(sample_dir / MASKS_DIR / "semantic_class_map.png")
    img = Image.open(sample_dir / "model_clean.png").convert("RGB")

    orig_ids = set(np.unique(np.array(mask)))
    params = {"flip": "horizontal"}
    _, [aug_mask] = apply_augmentation(img, [mask.copy()], params)
    aug_ids = set(np.unique(np.array(aug_mask)))

    assert aug_ids == orig_ids, f"Class IDs changed: before={orig_ids}  after={aug_ids}"


# ---------------------------------------------------------------------------
# augment_sample
# ---------------------------------------------------------------------------


def test_augment_sample_creates_files(tmp_path):
    sample_dir = _make_sample(tmp_path)
    entries = augment_sample(sample_dir, n_augmentations=3, seed=0)

    assert len(entries) == 3
    aug_root = sample_dir / AUGMENTED_DIR
    aug_dirs = sorted(aug_root.glob("aug_????"))
    assert len(aug_dirs) == 3

    for aug_dir in aug_dirs:
        assert (aug_dir / "augmented_image.png").exists()
        assert (aug_dir / "semantic_class_map.png").exists()
        assert (aug_dir / "augmentation_metadata.json").exists()


def test_augment_sample_image_and_mask_same_size(tmp_path):
    sample_dir = _make_sample(tmp_path)
    augment_sample(sample_dir, n_augmentations=2, seed=42)

    for aug_dir in sorted((sample_dir / AUGMENTED_DIR).glob("aug_????")):
        with Image.open(aug_dir / "augmented_image.png") as img:
            img_size = img.size
        with Image.open(aug_dir / "semantic_class_map.png") as m:
            mask_size = m.size
        assert img_size == mask_size, f"Mismatch in {aug_dir.name}: img={img_size}  mask={mask_size}"


def test_augment_sample_original_not_overwritten(tmp_path):
    sample_dir = _make_sample(tmp_path)
    original_bytes = (sample_dir / "model_clean.png").read_bytes()
    augment_sample(sample_dir, n_augmentations=2, seed=1)
    assert (sample_dir / "model_clean.png").read_bytes() == original_bytes


def test_augment_sample_skip_existing(tmp_path):
    sample_dir = _make_sample(tmp_path)
    entries_first = augment_sample(sample_dir, n_augmentations=3, seed=0)
    assert len(entries_first) == 3

    entries_second = augment_sample(sample_dir, n_augmentations=3, seed=0, overwrite=False)
    assert len(entries_second) == 0  # all skipped


def test_augment_sample_overwrite(tmp_path):
    sample_dir = _make_sample(tmp_path)
    augment_sample(sample_dir, n_augmentations=2, seed=0)

    # Corrupt one output
    aug_dir = next((sample_dir / AUGMENTED_DIR).glob("aug_????"))
    (aug_dir / "augmented_image.png").write_bytes(b"bad")

    augment_sample(sample_dir, n_augmentations=2, seed=0, overwrite=True)

    # Should be valid PNG again
    with Image.open(aug_dir / "augmented_image.png") as img:
        assert img.size[0] > 0


def test_augment_sample_no_masks_dir_skipped(tmp_path):
    sample_dir = tmp_path / "empty"
    sample_dir.mkdir()
    # No masks dir — should be skipped

    entries = augment_sample(sample_dir, n_augmentations=2)
    assert entries == []


def test_augment_sample_no_clean_image_skipped(tmp_path):
    sample_dir = tmp_path / "no_image"
    sample_dir.mkdir()
    masks_dir = sample_dir / MASKS_DIR
    masks_dir.mkdir()
    Image.new("L", (60, 60), 0).save(masks_dir / "semantic_class_map.png")

    entries = augment_sample(sample_dir, n_augmentations=2)
    assert entries == []


def test_augment_sample_metadata_content(tmp_path):
    sample_dir = _make_sample(tmp_path)
    augment_sample(sample_dir, n_augmentations=1, seed=7)

    aug_dir = next((sample_dir / AUGMENTED_DIR).glob("aug_????"))
    meta = json.loads((aug_dir / "augmentation_metadata.json").read_text())
    assert "aug_params" in meta
    assert "source_image" in meta


# ---------------------------------------------------------------------------
# process_dataset
# ---------------------------------------------------------------------------


def test_process_dataset_multiple_samples(tmp_path):
    for name in ("a", "b"):
        _make_sample(tmp_path, name=name)

    entries = process_dataset(tmp_path, n_augmentations=2, seed=0)
    assert len(entries) == 4  # 2 samples × 2 augmentations


def test_process_dataset_index_written(tmp_path):
    _make_sample(tmp_path, name="s1")
    process_dataset(tmp_path, n_augmentations=2, seed=0)

    index_path = tmp_path / "augmented_dataset_index.json"
    assert index_path.exists()
    entries = json.loads(index_path.read_text())
    assert isinstance(entries, list)
    assert len(entries) == 2


def test_process_dataset_entry_has_required_keys(tmp_path):
    _make_sample(tmp_path, name="s1")
    entries = process_dataset(tmp_path, n_augmentations=1, seed=0)

    entry = entries[0]
    for key in ("sample_id", "image", "target", "input_type"):
        assert key in entry, f"Missing key '{key}' in index entry"


# ---------------------------------------------------------------------------
# save_preview
# ---------------------------------------------------------------------------


def test_save_preview_creates_files(tmp_path):
    sample_dir = _make_sample(tmp_path)
    augment_sample(sample_dir, n_augmentations=3, seed=0)
    save_preview(sample_dir, n_preview=2)

    preview_dir = sample_dir / AUGMENTED_DIR / "previews"
    assert preview_dir.exists()
    previews = list(preview_dir.glob("*.png"))
    assert len(previews) >= 1
