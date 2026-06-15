"""Loss functions for floorplan segmentation training (spec_v005 §14)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassDiceLoss(nn.Module):
    """Soft multiclass Dice loss from softmax probabilities (spec_v005 §14).

    Background (class 0) is excluded from the Dice average by default because
    background dominance would otherwise swamp the foreground signal.
    """

    def __init__(
        self,
        num_classes: int,
        exclude_background: bool = True,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.exclude_background = exclude_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  ``[B, num_classes, H, W]`` raw logits.
            targets: ``[B, H, W]`` integer class IDs (long).
        Returns:
            Scalar Dice loss averaged over included classes.
        """
        probs = F.softmax(logits, dim=1)
        B, C, H, W = probs.shape
        target_oh = (
            F.one_hot(targets.clamp(0, C - 1), C).permute(0, 3, 1, 2).float()
        )

        start = 1 if self.exclude_background else 0
        scores: list[torch.Tensor] = []
        for c in range(start, C):
            p = probs[:, c].reshape(-1)
            t = target_oh[:, c].reshape(-1)
            inter = (p * t).sum()
            dice = (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
            scores.append(1.0 - dice)

        if not scores:
            return logits.sum() * 0.0
        return torch.stack(scores).mean()


class WeightedCEPlusDice(nn.Module):
    """Weighted CrossEntropyLoss + DiceLoss (spec_v005 §14).

    loss = ce_weight * WeightedCE(logits, targets) + dice_weight * Dice(logits, targets)

    Background is still included in CE (for clean mask quality) but excluded from
    Dice average (to prevent background dominance in shape loss).
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        dice_exclude_background: bool = True,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = MulticlassDiceLoss(
            num_classes=num_classes,
            exclude_background=dice_exclude_background,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.ce_weight * self.ce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )
