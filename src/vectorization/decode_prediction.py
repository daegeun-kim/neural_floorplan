"""Decode a run3 prediction image (RGB preview or class-ID PNG) into a class-ID mask."""

from __future__ import annotations

import numpy as np

# Must match _CLASS_COLORS in src/train_segmentation.py and CLASS_IDS/DEBUG_COLORS
# in src/generate_semantic_masks.py exactly. Duplicated (not imported) so this
# module does not pull in generate_semantic_masks.py's cairosvg/lxml dependency
# chain just to decode a color-coded preview image.
CLASS_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (200, 200, 200),  # background
    1: (245, 240, 232),  # floor
    2: (30, 30, 30),     # wall
    3: (60, 120, 220),   # window
    4: (220, 90, 90),    # door_arc
    5: (235, 140, 80),   # door_leaf
    6: (160, 70, 180),   # door_origin
}

PALETTE_ARRAY = np.array(list(CLASS_PALETTE.values()), dtype=np.int32)
PALETTE_IDS = np.array(list(CLASS_PALETTE.keys()), dtype=np.int32)

_MAX_CLASS_ID = max(CLASS_PALETTE.keys())


class IncompatibleMaskError(ValueError):
    """Raised when an input mask does not match the active 7-class run3 scheme."""


def decode_color_mask(rgb: np.ndarray, tolerance: int = 20) -> np.ndarray:
    """Convert an RGB array to a 2-D class-ID mask.

    Each pixel is matched to the closest palette entry within `tolerance`.
    Raises IncompatibleMaskError if many pixels remain unmatched - this is the
    expected outcome for a retired 5-class mask, since its palette differs
    from the active 7-class run3 palette.
    """
    h, w = rgb.shape[:2]
    flat = rgb.reshape(-1, 3).astype(np.int32)

    # Broadcast: (N, 1, 3) vs (1, C, 3) -> (N, C)
    diff = flat[:, None, :] - PALETTE_ARRAY[None, :, :]
    dist = np.sum(diff ** 2, axis=2)
    best = np.argmin(dist, axis=1)
    min_dist = dist[np.arange(len(flat)), best]

    class_ids = np.where(min_dist <= tolerance ** 2, PALETTE_IDS[best], -1)
    unmatched_frac = (class_ids == -1).mean()
    if unmatched_frac > 0.05:
        raise IncompatibleMaskError(
            f"Color decode failed: {unmatched_frac:.1%} of pixels unmatched against "
            "the active 7-class run3 palette (background/floor/wall/window/"
            "door_arc/door_leaf/door_origin). This usually means the image was "
            "produced by a retired 5-class model (background/wall/opening/room/icon) "
            "or another incompatible palette."
        )

    # Assign unmatched pixels to background
    class_ids[class_ids == -1] = 0
    return class_ids.reshape(h, w).astype(np.uint8)


def decode_class_id_mask(mask: np.ndarray) -> np.ndarray:
    """Validate and pass through an already class-ID-encoded single-channel mask.

    Raises IncompatibleMaskError if any value falls outside the active
    0..6 (background..door_origin) range.
    """
    if mask.ndim != 2:
        raise IncompatibleMaskError(
            f"Expected a single-channel class-ID mask, got shape {mask.shape}."
        )
    max_value = int(mask.max()) if mask.size else 0
    if max_value > _MAX_CLASS_ID:
        raise IncompatibleMaskError(
            f"Class-ID mask contains value {max_value}, which is outside the "
            f"active run3 range 0..{_MAX_CLASS_ID}. This input is incompatible "
            "with the 7-class run3 scheme."
        )
    return mask.astype(np.uint8)
