"""Seven-class segmentation metrics (spec_v006 §4.6). Required secondary.

Reuses ``src.metrics`` rather than reimplementing IoU. This module adds the
aggregation and empty-class accounting the evaluation contract requires: a class
absent from both prediction and reference contributes NaN and is dropped, never
counted as 0.0, and every per-class aggregate reports how many samples fed it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from src.metrics import (
    BoundaryF1Accumulator,
    compute_foreground_miou,
    compute_iou_per_class,
    compute_miou,
    compute_pixel_accuracy,
)

CLASS_NAMES = [
    "background",
    "floor",
    "wall",
    "window",
    "door_arc",
    "door_leaf",
    "door_origin",
]
NUM_CLASSES = len(CLASS_NAMES)
BOUNDARY_F1_CLASSES = (2, 3, 4, 5, 6)  # wall, window, door_arc, door_leaf, door_origin
BOUNDARY_TOLERANCE_PX = 2


@dataclass
class SegmentationAccumulator:
    """Accumulates per-sample IoU and a pooled confusion matrix.

    Both are kept because macro and micro aggregation diverge sharply for rare
    classes such as ``door_origin``; spec_v006 §4.6 requires reporting both.
    """

    per_class_ious: list[list[float]] = field(default_factory=list)
    pixel_accuracies: list[float] = field(default_factory=list)
    mious: list[float] = field(default_factory=list)
    foreground_mious: list[float] = field(default_factory=list)
    confusion: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    )
    # One accumulator per class: BoundaryF1Accumulator is single-class by design.
    boundary: dict[int, BoundaryF1Accumulator] = field(default_factory=dict)
    sample_count: int = 0

    def __post_init__(self) -> None:
        if not self.boundary:
            self.boundary = {
                cid: BoundaryF1Accumulator(class_id=cid, tolerance_px=BOUNDARY_TOLERANCE_PX)
                for cid in BOUNDARY_F1_CLASSES
            }

    def add(self, preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Add one sample; return that sample's metrics."""
        ious = compute_iou_per_class(preds, targets, NUM_CLASSES)
        self.per_class_ious.append(list(ious))

        pixel_acc = compute_pixel_accuracy(preds, targets)
        miou = compute_miou(preds, targets, NUM_CLASSES)
        fg_miou = compute_foreground_miou(preds, targets, NUM_CLASSES)

        self.pixel_accuracies.append(pixel_acc)
        self.mious.append(miou)
        self.foreground_mious.append(fg_miou)

        p = preds.view(-1).to(torch.int64)
        t = targets.view(-1).to(torch.int64)
        valid = (t >= 0) & (t < NUM_CLASSES) & (p >= 0) & (p < NUM_CLASSES)
        flat = (t[valid] * NUM_CLASSES + p[valid]).cpu().numpy()
        self.confusion += np.bincount(flat, minlength=NUM_CLASSES * NUM_CLASSES).reshape(
            NUM_CLASSES, NUM_CLASSES
        )

        # Boundary F1 needs 2-D class-ID arrays, not the flattened tensors above.
        if preds.dim() >= 2:
            pred_hw = preds.detach().cpu().numpy()
            target_hw = targets.detach().cpu().numpy()
            for accumulator in self.boundary.values():
                accumulator.update(pred_hw, target_hw)

        self.sample_count += 1
        sample_metrics = {
            "seg_pixel_accuracy": pixel_acc,
            "seg_miou": miou,
            "seg_foreground_miou": fg_miou,
        }
        for idx, name in enumerate(CLASS_NAMES):
            sample_metrics[f"seg_iou_{name}"] = ious[idx]
        return sample_metrics

    def aggregate(self) -> dict[str, object]:
        """Macro (per-sample averaged) and micro (pixel-pooled) aggregates."""
        macro_per_class: dict[str, float] = {}
        contributing: dict[str, int] = {}
        for idx, name in enumerate(CLASS_NAMES):
            values = [row[idx] for row in self.per_class_ious if not math.isnan(row[idx])]
            contributing[name] = len(values)
            macro_per_class[name] = sum(values) / len(values) if values else float("nan")

        micro_per_class: dict[str, float] = {}
        for idx, name in enumerate(CLASS_NAMES):
            tp = int(self.confusion[idx, idx])
            fn = int(self.confusion[idx, :].sum()) - tp
            fp = int(self.confusion[:, idx].sum()) - tp
            union = tp + fp + fn
            micro_per_class[name] = tp / union if union else float("nan")

        macro_values = [v for v in macro_per_class.values() if not math.isnan(v)]
        macro_fg = [
            v for name, v in macro_per_class.items() if name != "background" and not math.isnan(v)
        ]
        micro_values = [v for v in micro_per_class.values() if not math.isnan(v)]
        micro_fg = [
            v for name, v in micro_per_class.items() if name != "background" and not math.isnan(v)
        ]

        total = int(self.confusion.sum())
        micro_pixel_accuracy = float(np.trace(self.confusion)) / total if total else float("nan")

        boundary_scores = {cid: accumulator.compute() for cid, accumulator in self.boundary.items()}

        return {
            "sample_count": self.sample_count,
            "macro": {
                "per_class_iou": macro_per_class,
                "contributing_samples": contributing,
                # spec_v006 §4.6: the background-inclusion rule travels with the number.
                "miou_background_included": (
                    sum(macro_values) / len(macro_values) if macro_values else float("nan")
                ),
                "foreground_miou": (sum(macro_fg) / len(macro_fg) if macro_fg else float("nan")),
                "pixel_accuracy": _mean(self.pixel_accuracies),
            },
            "micro": {
                "per_class_iou": micro_per_class,
                "miou_background_included": (
                    sum(micro_values) / len(micro_values) if micro_values else float("nan")
                ),
                "foreground_miou": (sum(micro_fg) / len(micro_fg) if micro_fg else float("nan")),
                "pixel_accuracy": micro_pixel_accuracy,
            },
            "boundary_f1": {CLASS_NAMES[cid]: score for cid, score in boundary_scores.items()},
            "boundary_f1_tolerance_px": BOUNDARY_TOLERANCE_PX,
        }


def _mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")
