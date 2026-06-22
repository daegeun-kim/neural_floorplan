"""Offline sketch-style augmentation for floorplan training data (spec_v004).

Augmentation is applied identically to the input image and all mask files so
pixel-level alignment is preserved.  Pixel-level variations (blur, brightness)
are applied to the image only — never to semantic masks.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

MASKS_DIR = "masks"
AUGMENTED_DIR = "augmented"

# Mask files that must receive the same spatial transform as the image.
SPATIAL_MASK_NAMES = [
    "wall_mask.png",
    "floor_mask.png",
    "window_mask.png",
    "door_origin_mask.png",
    "door_arc_mask.png",
    "door_leaf_mask.png",
    "semantic_class_map.png",
]

# Valid source image names (per spec_v004).
CLEAN_IMAGE_NAMES = ["model_clean.png", "model_clean01.png"]


# ---------------------------------------------------------------------------
# Individual augmentation primitives
# ---------------------------------------------------------------------------


def _apply_flip(img: Image.Image, masks: list[Image.Image], flip: str) -> tuple[Image.Image, list[Image.Image]]:
    """horizontal or vertical flip."""
    if flip == "horizontal":
        method = Image.FLIP_LEFT_RIGHT
    elif flip == "vertical":
        method = Image.FLIP_TOP_BOTTOM
    else:
        return img, masks
    return img.transpose(method), [m.transpose(method) for m in masks]


def _apply_rotate90(img: Image.Image, masks: list[Image.Image], k: int) -> tuple[Image.Image, list[Image.Image]]:
    """Rotate by k * 90 degrees counter-clockwise."""
    if k == 0:
        return img, masks
    angle = k * 90
    rotated_img = img.rotate(angle, expand=True)
    # Use NEAREST for masks to avoid interpolating class IDs
    rotated_masks = [m.rotate(angle, expand=True, resample=Image.NEAREST) for m in masks]
    return rotated_img, rotated_masks


def _apply_translation(
    img: Image.Image, masks: list[Image.Image], dx: int, dy: int
) -> tuple[Image.Image, list[Image.Image]]:
    """Translate by (dx, dy) pixels, padding with white / zero."""
    def _shift_image(im: Image.Image, fill: int | tuple) -> Image.Image:
        shifted = Image.new(im.mode, im.size, fill)
        shifted.paste(im, (dx, dy))
        return shifted

    new_img = _shift_image(img, 255 if img.mode == "L" else (255, 255, 255))
    new_masks = [_shift_image(m, 0) for m in masks]
    return new_img, new_masks


def _apply_blur(img: Image.Image, radius: float) -> Image.Image:
    """Gaussian blur — image only."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Scale pixel brightness by factor.  Clamped to [0, 255]."""
    arr = np.array(img, dtype=np.float32) * factor
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=img.mode)


# ---------------------------------------------------------------------------
# Augmentation config and application
# ---------------------------------------------------------------------------


def _random_aug_params(rng: random.Random, cfg: dict) -> dict:
    """Sample a random set of augmentation parameters."""
    params: dict = {}

    # Spatial
    if cfg.get("horizontal_flip") and rng.random() < 0.5:
        params["flip"] = "horizontal"
    elif cfg.get("vertical_flip") and rng.random() < 0.5:
        params["flip"] = "vertical"

    if cfg.get("rotate_90"):
        params["rotate_k"] = rng.choice([0, 1, 2, 3])

    max_tx = cfg.get("max_translate_px", 0)
    if max_tx > 0:
        params["dx"] = rng.randint(-max_tx, max_tx)
        params["dy"] = rng.randint(-max_tx, max_tx)

    # Pixel-level (image only)
    max_blur = cfg.get("max_blur_radius", 0.0)
    if max_blur > 0 and rng.random() < 0.5:
        params["blur_radius"] = rng.uniform(0.3, max_blur)

    brightness_range = cfg.get("brightness_range", [1.0, 1.0])
    lo, hi = brightness_range
    if abs(hi - lo) > 1e-6:
        params["brightness"] = rng.uniform(lo, hi)

    return params


def apply_augmentation(
    img: Image.Image,
    masks: list[Image.Image],
    params: dict,
) -> tuple[Image.Image, list[Image.Image]]:
    """Apply a parameter dict to image and masks.  Returns (aug_img, aug_masks)."""
    if "flip" in params:
        img, masks = _apply_flip(img, masks, params["flip"])

    k = params.get("rotate_k", 0)
    if k:
        img, masks = _apply_rotate90(img, masks, k)

    dx, dy = params.get("dx", 0), params.get("dy", 0)
    if dx or dy:
        img, masks = _apply_translation(img, masks, dx, dy)

    # Pixel-level — image only
    if "blur_radius" in params:
        img = _apply_blur(img, params["blur_radius"])

    if "brightness" in params:
        img = _apply_brightness(img, params["brightness"])

    return img, masks


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotate_90": True,
    "max_translate_px": 10,
    "max_blur_radius": 1.0,
    "brightness_range": [0.85, 1.15],
}


# ---------------------------------------------------------------------------
# Per-sample augmentation
# ---------------------------------------------------------------------------


def augment_sample(
    sample_dir: Path,
    n_augmentations: int = 5,
    output_dir_name: str = AUGMENTED_DIR,
    masks_dir_name: str = MASKS_DIR,
    overwrite: bool = False,
    config: dict | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Generate *n_augmentations* augmented pairs for one sample.

    Returns a list of dataset-index entries for the generated pairs.
    """
    cfg = config or DEFAULT_CONFIG
    rng = random.Random(seed)

    masks_dir = sample_dir / masks_dir_name
    if not masks_dir.exists():
        logger.warning("No masks dir in %s — skipping", sample_dir)
        return []

    semantic_map_path = masks_dir / "semantic_class_map.png"
    if not semantic_map_path.exists():
        logger.warning("No semantic_class_map.png in %s — skipping", sample_dir)
        return []

    # Collect available source images
    source_images = [
        sample_dir / name for name in CLEAN_IMAGE_NAMES if (sample_dir / name).exists()
    ]
    if not source_images:
        logger.warning("No clean image found in %s — skipping", sample_dir)
        return []

    # Load masks that need spatial transform
    mask_images: dict[str, Image.Image] = {}
    for mask_name in SPATIAL_MASK_NAMES:
        p = masks_dir / mask_name
        if p.exists():
            mask_images[mask_name] = Image.open(p).copy()

    out_dir = sample_dir / output_dir_name
    out_dir.mkdir(exist_ok=True)

    entries: list[dict] = []

    # Distribute augmentations across source images
    aug_idx = 0
    for src_img_path in source_images:
        with Image.open(src_img_path) as img_raw:
            src_img = img_raw.convert("RGB").copy()

        n = n_augmentations // len(source_images) + (
            1 if aug_idx < n_augmentations % len(source_images) else 0
        )

        for _ in range(n):
            aug_dir = out_dir / f"aug_{aug_idx:04d}"
            aug_image_path = aug_dir / "augmented_image.png"

            if aug_image_path.exists() and not overwrite:
                aug_idx += 1
                continue

            aug_dir.mkdir(exist_ok=True)
            params = _random_aug_params(rng, cfg)

            # Apply spatial transforms to image and all masks together
            masks_list = [mask_images[k].copy() for k in SPATIAL_MASK_NAMES if k in mask_images]
            aug_img, aug_masks = apply_augmentation(src_img.copy(), masks_list, params)

            aug_img.save(aug_image_path)

            # Save each augmented mask under its original name
            for mask_name, aug_mask in zip(
                [k for k in SPATIAL_MASK_NAMES if k in mask_images], aug_masks
            ):
                aug_mask.save(aug_dir / mask_name)

            # Save augmentation metadata
            meta = {
                "source_image": str(src_img_path.relative_to(sample_dir.parent)),
                "aug_params": params,
            }
            with open(aug_dir / "augmentation_metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            entries.append(
                {
                    "sample_id": f"{sample_dir.name}_aug_{aug_idx:04d}",
                    "image": str((aug_dir / "augmented_image.png").relative_to(sample_dir.parent)),
                    "target": str((aug_dir / "semantic_class_map.png").relative_to(sample_dir.parent)),
                    "input_type": "augmented",
                }
            )
            aug_idx += 1

    return entries


def save_preview(
    sample_dir: Path,
    n_preview: int = 3,
    masks_dir_name: str = MASKS_DIR,
    output_dir_name: str = AUGMENTED_DIR,
    seed: int = 0,
) -> None:
    """Export side-by-side preview images for manual inspection."""
    aug_root = sample_dir / output_dir_name
    if not aug_root.exists():
        logger.warning("No augmented dir in %s", sample_dir)
        return

    preview_dir = aug_root / "previews"
    preview_dir.mkdir(exist_ok=True)

    aug_dirs = sorted(aug_root.glob("aug_????"))[:n_preview]
    original_img_path = next(
        (sample_dir / n for n in CLEAN_IMAGE_NAMES if (sample_dir / n).exists()), None
    )
    original_mask_path = sample_dir / masks_dir_name / "semantic_class_map.png"

    for aug_dir in aug_dirs:
        aug_img_path = aug_dir / "augmented_image.png"
        aug_mask_path = aug_dir / "semantic_class_map.png"

        if not aug_img_path.exists() or not aug_mask_path.exists():
            continue

        panels = []
        for p in (original_img_path, aug_img_path):
            if p and p.exists():
                panels.append(Image.open(p).convert("RGB"))
        for p in (original_mask_path, aug_mask_path):
            if p and p.exists():
                m = Image.open(p).convert("L")
                # Scale 0..4 → 0..255 for visibility
                arr = np.array(m, dtype=np.uint8) * (255 // 4)
                panels.append(Image.fromarray(arr).convert("RGB"))

        if not panels:
            continue

        target_h = max(p.height for p in panels)
        resized = [p.resize((int(p.width * target_h / p.height), target_h)) for p in panels]
        total_w = sum(p.width for p in resized)
        strip = Image.new("RGB", (total_w, target_h), (200, 200, 200))
        x = 0
        for panel in resized:
            strip.paste(panel, (x, 0))
            x += panel.width

        strip.save(preview_dir / f"{aug_dir.name}_preview.png")
        logger.info("Preview saved: %s", preview_dir / f"{aug_dir.name}_preview.png")


def process_dataset(
    root_dir: Path,
    n_augmentations: int = 5,
    output_dir_name: str = AUGMENTED_DIR,
    masks_dir_name: str = MASKS_DIR,
    overwrite: bool = False,
    preview: bool = False,
    config: dict | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Augment all samples in root_dir and return the combined dataset index."""
    sample_dirs = sorted(d for d in root_dir.iterdir() if d.is_dir())
    all_entries: list[dict] = []

    for i, sample_dir in enumerate(sample_dirs, 1):
        logger.info("[%d/%d] %s", i, len(sample_dirs), sample_dir.name)
        entries = augment_sample(
            sample_dir,
            n_augmentations=n_augmentations,
            output_dir_name=output_dir_name,
            masks_dir_name=masks_dir_name,
            overwrite=overwrite,
            config=config,
            seed=seed,
        )
        all_entries.extend(entries)

        if preview and entries:
            save_preview(sample_dir, masks_dir_name=masks_dir_name, output_dir_name=output_dir_name)

    index_path = root_dir / "augmented_dataset_index.json"
    with open(index_path, "w") as f:
        json.dump(all_entries, f, indent=2)
    logger.info("Dataset index saved: %s  (%d entries)", index_path, len(all_entries))

    return all_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment floorplan dataset (spec_v004).")
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--n-augmentations", type=int, default=5, metavar="N")
    parser.add_argument("--output-dir-name", default=AUGMENTED_DIR, metavar="NAME")
    parser.add_argument("--masks-dir-name", default=MASKS_DIR, metavar="NAME")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Save preview strip images.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    process_dataset(
        args.root_dir,
        n_augmentations=args.n_augmentations,
        output_dir_name=args.output_dir_name,
        masks_dir_name=args.masks_dir_name,
        overwrite=args.overwrite,
        preview=args.preview,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
