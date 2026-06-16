"""Lightweight morphological cleanup for each class mask."""

from __future__ import annotations

import cv2
import numpy as np


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def clean_wall_mask(
    mask: np.ndarray,
    min_area: int = 20,
    close_gap_px: int = 3,
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_gap_px, close_gap_px))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = _remove_small_components(closed, min_area)
    return cleaned


def clean_opening_mask(
    mask: np.ndarray,
    min_area: int = 8,
) -> np.ndarray:
    return _remove_small_components(mask, min_area)


def clean_room_mask(
    mask: np.ndarray,
    min_area: int = 100,
    close_gap_px: int = 5,
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_gap_px, close_gap_px))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = _remove_small_components(closed, min_area)
    return cleaned
