"""Checkpoint save / load utilities for segmentation training (spec_v005)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    global_step: int,
    best_metric_value: float,
    best_metric_name: str,
    config: dict,
    class_mapping: dict,
    history: list[dict],
    arch_version: str | None = None,
    class_weights: list[float] | None = None,
    model_version: str | None = None,
    run_name: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric_value": best_metric_value,
        "best_metric_name": best_metric_name,
        "config": config,
        "class_mapping": class_mapping,
        "history": history,
        "arch_version": arch_version,
        "class_weights": class_weights,
        "model_version": model_version,
        "run_name": run_name,
    }
    torch.save(payload, path)
    logger.debug("Checkpoint saved: %s", path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """Load checkpoint, return its metadata dict."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    logger.info("Checkpoint loaded from %s (epoch %d)", path, payload.get("epoch", -1))
    return payload


def resolve_resume_path(checkpoint_dir: Path, resume_from: str) -> Path | None:
    """Return the checkpoint path to resume from, or None to start fresh."""
    if resume_from == "auto":
        latest = checkpoint_dir / "latest.pt"
        if latest.exists():
            return latest
        best = checkpoint_dir / "best.pt"
        if best.exists():
            return best
        return None
    p = Path(resume_from)
    if p.exists():
        return p
    return None
