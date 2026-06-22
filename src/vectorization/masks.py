"""Split a class-ID mask into per-class binary masks (active 7-class run3 scheme)."""

from __future__ import annotations

import numpy as np

# Must match CLASS_IDS in src/generate_semantic_masks.py exactly (duplicated,
# not imported - see decode_prediction.py for why).
CLASS_BACKGROUND = 0
CLASS_FLOOR = 1
CLASS_WALL = 2
CLASS_WINDOW = 3
CLASS_DOOR_ARC = 4
CLASS_DOOR_LEAF = 5
CLASS_DOOR_ORIGIN = 6


def split_class_masks(class_map: np.ndarray) -> dict[str, np.ndarray]:
    """Return binary uint8 masks (255 = present, 0 = absent) for each active class."""
    return {
        "floor": (class_map == CLASS_FLOOR).astype(np.uint8) * 255,
        "wall": (class_map == CLASS_WALL).astype(np.uint8) * 255,
        "window": (class_map == CLASS_WINDOW).astype(np.uint8) * 255,
        "door_arc": (class_map == CLASS_DOOR_ARC).astype(np.uint8) * 255,
        "door_leaf": (class_map == CLASS_DOOR_LEAF).astype(np.uint8) * 255,
        "door_origin": (class_map == CLASS_DOOR_ORIGIN).astype(np.uint8) * 255,
    }
