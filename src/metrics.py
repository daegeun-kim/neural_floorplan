"""Segmentation metrics for floorplan training (spec_v005)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core IoU / accuracy metrics (kept for backward compatibility)
# ---------------------------------------------------------------------------


def compute_iou_per_class(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> list[float]:
    """Per-class IoU over flat prediction/target tensors.

    Returns:
        List of length *num_classes*; NaN for classes absent from both tensors.
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    if ignore_index >= 0:
        valid = targets != ignore_index
        preds = preds[valid]
        targets = targets[valid]

    ious: list[float] = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        tgt_cls = targets == cls
        intersection = (pred_cls & tgt_cls).sum().item()
        union = (pred_cls | tgt_cls).sum().item()
        ious.append(float("nan") if union == 0 else intersection / union)

    return ious


def compute_miou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> float:
    """Mean IoU over non-NaN classes (includes background)."""
    ious = compute_iou_per_class(preds, targets, num_classes, ignore_index)
    valid = [v for v in ious if not (v != v)]
    return float(sum(valid) / len(valid)) if valid else float("nan")


def compute_pixel_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
) -> float:
    """Fraction of correctly classified pixels (all classes)."""
    preds = preds.view(-1)
    targets = targets.view(-1)

    if ignore_index >= 0:
        valid = targets != ignore_index
        preds = preds[valid]
        targets = targets[valid]

    if targets.numel() == 0:
        return float("nan")
    return float((preds == targets).float().mean().item())


# ---------------------------------------------------------------------------
# Foreground-only metrics (spec_v005 §16)
# ---------------------------------------------------------------------------


def compute_foreground_miou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    background_class: int = 0,
) -> float:
    """Mean IoU over foreground classes only (excludes background_class)."""
    ious = compute_iou_per_class(preds, targets, num_classes)
    fg_ious = [
        v for i, v in enumerate(ious)
        if i != background_class and not (v != v)
    ]
    return float(sum(fg_ious) / len(fg_ious)) if fg_ious else float("nan")


def compute_foreground_pixel_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    background_class: int = 0,
) -> float:
    """Pixel accuracy restricted to foreground pixels (target != background)."""
    preds = preds.view(-1)
    targets = targets.view(-1)

    fg_mask = targets != background_class
    preds = preds[fg_mask]
    targets = targets[fg_mask]

    if targets.numel() == 0:
        return float("nan")
    return float((preds == targets).float().mean().item())


# ---------------------------------------------------------------------------
# Boundary F1 (spec_v005 §16)
# ---------------------------------------------------------------------------


def _binary_erode_np(mask: np.ndarray) -> np.ndarray:
    """3×3 binary erosion using pure numpy (no cv2/scipy dependency)."""
    m = mask.astype(bool)
    h, w = m.shape
    p = np.pad(m, 1, constant_values=False)
    return (
        p[0:h, 0:w] & p[0:h, 1:w+1] & p[0:h, 2:w+2] &
        p[1:h+1, 0:w] & p[1:h+1, 1:w+1] & p[1:h+1, 2:w+2] &
        p[2:h+2, 0:w] & p[2:h+2, 1:w+1] & p[2:h+2, 2:w+2]
    )


def _binary_dilate_np(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a (2r+1)×(2r+1) structuring element using pure numpy."""
    if radius <= 0:
        return mask.astype(bool)
    m = mask.astype(bool)
    h, w = m.shape
    p = np.pad(m, radius, constant_values=False)
    result = np.zeros((h, w), dtype=bool)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            result |= p[dy:dy + h, dx:dx + w]
    return result


def _extract_boundary_np(binary_mask: np.ndarray) -> np.ndarray:
    """boundary = mask XOR erode(mask, 3×3 kernel)."""
    return binary_mask.astype(bool) ^ _binary_erode_np(binary_mask)


def _dilate_boundary_np(boundary: np.ndarray, tolerance_px: int) -> np.ndarray:
    """Dilate boundary pixels by tolerance_px for approximate matching."""
    return _binary_dilate_np(boundary, tolerance_px)


class BoundaryF1Accumulator:
    """Accumulates per-sample boundary precision/recall numerators+denominators."""

    def __init__(self, class_id: int, tolerance_px: int = 2) -> None:
        self.class_id = class_id
        self.tolerance_px = tolerance_px
        self._tp_p = 0  # predicted boundary pixels within tol of target boundary
        self._n_p  = 0  # total predicted boundary pixels
        self._tp_r = 0  # target boundary pixels within tol of predicted boundary
        self._n_r  = 0  # total target boundary pixels

    def update(self, pred_hw: np.ndarray, target_hw: np.ndarray) -> None:
        """Add one sample (2-D class-ID arrays, HW)."""
        pred_bin   = (pred_hw   == self.class_id)
        target_bin = (target_hw == self.class_id)

        pred_boundary   = _extract_boundary_np(pred_bin)
        target_boundary = _extract_boundary_np(target_bin)

        if not pred_boundary.any() and not target_boundary.any():
            return  # both empty → no contribution

        pred_boundary_dilated   = _dilate_boundary_np(pred_boundary,   self.tolerance_px)
        target_boundary_dilated = _dilate_boundary_np(target_boundary, self.tolerance_px)

        self._tp_p += int((pred_boundary & target_boundary_dilated).sum())
        self._n_p  += int(pred_boundary.sum())
        self._tp_r += int((target_boundary & pred_boundary_dilated).sum())
        self._n_r  += int(target_boundary.sum())

    def compute(self) -> float:
        """Return boundary F1 over all accumulated samples."""
        if self._n_p == 0 and self._n_r == 0:
            return float("nan")
        precision = self._tp_p / (self._n_p  + 1e-8)
        recall    = self._tp_r / (self._n_r  + 1e-8)
        denom     = precision + recall
        if denom < 1e-8:
            return 0.0
        return float(2.0 * precision * recall / denom)


# ---------------------------------------------------------------------------
# Vector-ready score (spec_v005 §18)
# ---------------------------------------------------------------------------


def compute_vector_ready_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted composite score from a metrics dict and a weight dict.

    Keys in *weights* must match keys in *metrics*.  NaN entries are skipped
    and the remaining weights are renormalized.
    """
    score     = 0.0
    total_w   = 0.0
    for key, w in weights.items():
        v = metrics.get(key, float("nan"))
        if v != v:  # NaN
            continue
        score   += w * v
        total_w += w
    return float(score / total_w) if total_w > 1e-8 else float("nan")


# ---------------------------------------------------------------------------
# Automatic class-weight computation (spec_v005 §15)
# ---------------------------------------------------------------------------


_DEFAULT_CLASS_NAMES = [
    "background", "floor", "wall", "window", "door_arc", "door_leaf", "door_origin",
]


def compute_class_weights_auto(
    train_index: Path | str,
    dataset_root: Path | str,
    num_classes: int = 7,
    priority_multipliers: list[float] | None = None,
    min_weight: float = 0.1,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Compute inverse-sqrt-frequency class weights from training masks.

    Formula (spec_v005 §15):
        base[c]       = 1 / sqrt(freq[c] + eps)
        normalized[c] = base[c] / mean(base[foreground_classes])
        final[c]      = clip(normalized[c] * priority_multiplier[c], min, max)

    Args:
        train_index:          Path to split JSON index.
        dataset_root:         Dataset root (mask paths are relative to this).
        num_classes:          Number of semantic classes.
        priority_multipliers: Per-class multiplier list (index = class ID).
                              Default (background, floor, wall, window, door_arc,
                              door_leaf, door_origin): [0.50, 0.80, 0.80, 1.20, 1.80, 1.80, 1.80]
        min_weight, max_weight: Clip bounds.

    Returns:
        Float32 tensor of length *num_classes*.
    """
    from PIL import Image

    if priority_multipliers is None:
        priority_multipliers = [0.50, 0.80, 0.80, 1.20, 1.80, 1.80, 1.80]

    dataset_root = Path(dataset_root)

    with open(train_index) as f:
        entries: list[dict] = json.load(f)

    # Deduplicate: F1_scaled.png and model_clean.png share the same mask
    unique_targets = list({e["target"] for e in entries})
    logger.info("Computing class weights from %d unique masks ...", len(unique_targets))

    pixel_counts = np.zeros(num_classes, dtype=np.float64)

    for i, target_rel in enumerate(unique_targets):
        mask_path = dataset_root / target_rel
        try:
            with Image.open(mask_path) as m:
                arr = np.array(m.convert("L"))
            counts = np.bincount(arr.ravel(), minlength=num_classes)[:num_classes]
            pixel_counts += counts.astype(np.float64)
        except Exception as exc:
            logger.warning("Skipping mask %s: %s", mask_path, exc)

        if (i + 1) % 500 == 0:
            logger.info("  [%d / %d] masks scanned", i + 1, len(unique_targets))

    total = pixel_counts.sum() + 1e-10
    freq  = pixel_counts / total

    base = 1.0 / np.sqrt(freq + 1e-6)

    fg_indices = [c for c in range(num_classes) if c != 0]
    fg_mean    = base[fg_indices].mean() + 1e-10
    normalized = base / fg_mean

    mults = list(priority_multipliers) + [1.0] * num_classes
    for c in range(num_classes):
        normalized[c] *= mults[c]

    clipped = np.clip(normalized, min_weight, max_weight)

    names = _DEFAULT_CLASS_NAMES[:num_classes] + [
        f"class_{i}" for i in range(len(_DEFAULT_CLASS_NAMES), num_classes)
    ]
    logger.info(
        "Class weights (%s)",
        " ".join(f"{name}={w:.3f}" for name, w in zip(names, clipped)),
    )
    return torch.tensor(clipped, dtype=torch.float32)
