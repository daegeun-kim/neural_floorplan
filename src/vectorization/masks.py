"""Split a class-ID mask into per-class binary masks."""

from __future__ import annotations

import numpy as np


CLASS_BACKGROUND = 0
CLASS_WALL = 1
CLASS_OPENING = 2
CLASS_ROOM = 3
CLASS_ICON = 4


def split_class_masks(class_map: np.ndarray) -> dict[str, np.ndarray]:
    """Return binary uint8 masks (255 = present, 0 = absent) for each active class."""
    return {
        "wall":    (class_map == CLASS_WALL).astype(np.uint8) * 255,
        "opening": (class_map == CLASS_OPENING).astype(np.uint8) * 255,
        "room":    (class_map == CLASS_ROOM).astype(np.uint8) * 255,
    }
