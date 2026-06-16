"""Decode a color-coded RGB prediction image into a class-ID mask."""

from __future__ import annotations

import numpy as np

# Must match _CLASS_COLORS in src/train_segmentation.py
CLASS_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (200, 200, 200),  # background
    1: (30,  30,  30),   # wall
    2: (200, 80,  80),   # opening
    3: (80,  160, 220),  # room
    4: (80,  200, 100),  # icon
}

PALETTE_ARRAY = np.array(list(CLASS_PALETTE.values()), dtype=np.int32)
PALETTE_IDS = np.array(list(CLASS_PALETTE.keys()), dtype=np.int32)


def decode_color_mask(rgb: np.ndarray, tolerance: int = 20) -> np.ndarray:
    """Convert an RGB array to a 2-D class-ID mask.

    Each pixel is matched to the closest palette entry within `tolerance`.
    Raises ValueError if many pixels remain unmatched (suggesting a wrong palette).
    """
    h, w = rgb.shape[:2]
    flat = rgb.reshape(-1, 3).astype(np.int32)

    # Broadcast: (N, 1, 3) vs (1, C, 3) → (N, C)
    diff = flat[:, None, :] - PALETTE_ARRAY[None, :, :]
    dist = np.sum(diff ** 2, axis=2)
    best = np.argmin(dist, axis=1)
    min_dist = dist[np.arange(len(flat)), best]

    class_ids = np.where(min_dist <= tolerance ** 2, PALETTE_IDS[best], -1)
    unmatched_frac = (class_ids == -1).mean()
    if unmatched_frac > 0.05:
        raise ValueError(
            f"Color decode failed: {unmatched_frac:.1%} of pixels unmatched. "
            "Check that the image uses the expected prediction color palette."
        )

    # Assign unmatched pixels to background
    class_ids[class_ids == -1] = 0
    return class_ids.reshape(h, w).astype(np.uint8)
