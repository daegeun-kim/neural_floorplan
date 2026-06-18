"""Segmentation training script — Phase 1: frozen backbone + custom decoder (spec_v005).

Phase-1 workflow:
  1. Build pretrained SegFormer backbone (frozen).
  2. Cache backbone features for every split sample to features/<variant>/.
  3. Train only the FloorplanDecoder using pre-cached features.
  4. Backbone is NOT called during training epochs → fast iteration.

Usage:
    python -m src.train_segmentation --config configs/train_segformer_b0.yaml
    python -m src.train_segmentation --config configs/train_segformer_b0.yaml --debug
    python -m src.train_segmentation --config configs/train_segformer_b0.yaml --overfit 5
    python -m src.train_segmentation --config configs/train_segformer_b0.yaml --resume checkpoints/segformer_b0_v005/latest.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.amp
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset

from src.checkpointing import load_checkpoint, resolve_resume_path, save_checkpoint
from src.dataset import FloorplanDataset
from src.feature_cache import (
    CachedFloorplanDataset,
    compute_config_hash,
    extract_features_for_split,
)
from src.losses import WeightedCEPlusDice
from src.metrics import (
    BoundaryF1Accumulator,
    compute_class_weights_auto,
    compute_foreground_miou,
    compute_foreground_pixel_accuracy,
    compute_iou_per_class,
    compute_miou,
    compute_pixel_accuracy,
    compute_vector_ready_score,
)
from src.models import (
    FloorplanDecoder,
    FloorplanSegModel,
    build_backbone,
    build_decoder,
)

logger = logging.getLogger(__name__)

DEFAULT_CLASS_MAPPING: dict[int, str] = {
    0: "background",
    1: "wall",
    2: "opening",
    3: "room",
    4: "icon",
}

ARCH_VERSION = "v3_weighted_ce_dice"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _normalize_config(cfg: dict) -> dict:
    """Flatten new nested config structure into the flat format used internally.

    Supports both old flat format and new nested format (paths:/image: sections).
    Existing flat keys are never overwritten so old configs keep working.
    """
    paths_cfg = cfg.get("paths", {})
    image_cfg  = cfg.get("image", {})
    flat = dict(cfg)

    if paths_cfg:
        flat.setdefault("dataset_root",      paths_cfg.get("dataset_root", ""))
        flat.setdefault("train_index",       paths_cfg.get("train_index",  "splits/train.json"))
        flat.setdefault("val_index",         paths_cfg.get("val_index",    "splits/val.json"))
        flat.setdefault("debug_train_index", paths_cfg.get("debug_train_index", "splits/debug_train.json"))
        flat.setdefault("debug_val_index",   paths_cfg.get("debug_val_index",   "splits/debug_val.json"))

    if image_cfg:
        flat.setdefault("image_size",  image_cfg.get("image_size",  512))
        flat.setdefault("num_classes", image_cfg.get("num_classes", 5))

    return flat


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device setup (spec_v005 §13)
# ---------------------------------------------------------------------------


def _setup_device(cfg: dict) -> torch.device:
    device_cfg    = cfg.get("device", {})
    require_cuda  = device_cfg.get("require_cuda", False)
    allow_cpu_dbg = device_cfg.get("allow_cpu_debug", True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(
            "Device: %s | CUDA %s | %.1f GB VRAM | AMP=%s",
            name, torch.version.cuda, mem_gb,
            cfg.get("training", {}).get("mixed_precision", False),
        )
    else:
        if require_cuda and not allow_cpu_dbg:
            raise RuntimeError(
                "CUDA is required (device.require_cuda=true) but torch.cuda.is_available() is False. "
                "Ensure the GPU driver and CUDA toolkit are installed, or set allow_cpu_debug=true "
                "for CPU-only smoke tests."
            )
        device = torch.device("cpu")
        logger.warning("CUDA not available — running on CPU (training will be slow).")

    return device


# ---------------------------------------------------------------------------
# Class-weight helpers (spec_v005 §15)
# ---------------------------------------------------------------------------


def _build_class_weights(cfg: dict, ckpt_dir: Path) -> torch.Tensor | None:
    cw_cfg = cfg.get("class_weights", {})
    mode   = cw_cfg.get("mode", "none")

    if mode == "manual":
        vals = cw_cfg.get("values")
        if vals:
            logger.info("Using manual class weights: %s", vals)
            return torch.tensor(vals, dtype=torch.float32)
        return None

    if mode != "auto":
        return None

    # Auto mode: cache to avoid recomputing on every run
    cache_path = ckpt_dir / "class_weights.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        weights = torch.tensor(cached["weights"], dtype=torch.float32)
        logger.info("Loaded cached class weights from %s: %s", cache_path, cached["weights"])
        return weights

    num_classes = cfg["num_classes"]
    mults_cfg   = cw_cfg.get("priority_multipliers", {})
    class_order = [DEFAULT_CLASS_MAPPING[i] for i in range(num_classes)]
    multipliers = [float(mults_cfg.get(name, 1.0)) for name in class_order]

    weights = compute_class_weights_auto(
        train_index  = cfg["train_index"],
        dataset_root = cfg["dataset_root"],
        num_classes  = num_classes,
        priority_multipliers = multipliers,
        min_weight   = float(cw_cfg.get("min_weight", 0.1)),
        max_weight   = float(cw_cfg.get("max_weight", 5.0)),
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"weights": weights.tolist(), "classes": class_order}, f, indent=2)
    logger.info("Class weights saved to %s", cache_path)

    return weights


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------


def _write_temp_index(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f)


def build_cached_loaders(
    cfg: dict,
    backbone: Any,
    device: torch.device,
    debug: bool = False,
    overfit_n: int = 0,
) -> tuple[DataLoader, DataLoader]:
    cache_cfg    = cfg.get("feature_cache", {})
    cache_dir    = Path(cache_cfg.get("cache_dir", "features/segformer_b0"))
    force_rebuild = cache_cfg.get("force_rebuild", False)
    dataset_root = cfg["dataset_root"]
    image_size   = cfg["image_size"]
    bs           = cfg["training"]["batch_size"]

    if overfit_n > 0:
        with open(cfg["train_index"]) as f:
            all_entries = json.load(f)
        overfit_entries = all_entries[:overfit_n]
        temp_index = Path("splits/_overfit_temp.json")
        _write_temp_index(overfit_entries, temp_index)
        extract_features_for_split(
            backbone, temp_index, dataset_root, cache_dir,
            cfg, force_rebuild=force_rebuild, device=device,
        )
        ds = CachedFloorplanDataset(temp_index, dataset_root, cache_dir, image_size)
        bs_actual = min(bs, overfit_n)
        return (
            DataLoader(ds, batch_size=bs_actual, shuffle=True),
            DataLoader(ds, batch_size=bs_actual, shuffle=False),
        )

    if debug:
        train_index = cfg.get("debug_train_index", cfg["train_index"])
        val_index   = cfg.get("debug_val_index",   cfg["val_index"])
    else:
        train_index = cfg["train_index"]
        val_index   = cfg["val_index"]

    extract_features_for_split(
        backbone, train_index, dataset_root, cache_dir,
        cfg, force_rebuild=force_rebuild, device=device,
    )
    extract_features_for_split(
        backbone, val_index, dataset_root, cache_dir,
        cfg, force_rebuild=force_rebuild, device=device,
    )

    train_ds = CachedFloorplanDataset(train_index, dataset_root, cache_dir, image_size)
    val_ds   = CachedFloorplanDataset(val_index,   dataset_root, cache_dir, image_size)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader


def build_preview_loader(
    cfg: dict,
    debug: bool = False,
    overfit_n: int = 0,
    n_samples: int = 4,
) -> DataLoader:
    dataset_root = cfg["dataset_root"]
    image_size   = cfg["image_size"]

    if overfit_n > 0:
        index = cfg["train_index"]
    elif debug:
        index = cfg.get("debug_val_index", cfg["val_index"])
    else:
        index = cfg["val_index"]

    ds = FloorplanDataset(index, dataset_root, image_size, augment=False)
    if len(ds) > n_samples:
        ds = Subset(ds, list(range(n_samples)))
    return DataLoader(ds, batch_size=n_samples, shuffle=False)


def make_preview_loader(
    train_config_path: str | Path,
    n_samples: int = 4,
) -> DataLoader:
    """Build a DataLoader for preview samples from a training YAML config path."""
    cfg = _normalize_config(load_config(train_config_path))
    return build_preview_loader(cfg, n_samples=n_samples)


@torch.no_grad()
def save_sample_artifacts(
    full_model: "FloorplanSegModel",
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    n_samples: int = 4,
) -> list[Path]:
    """Run inference and write input.png + prediction.png into per-sample subdirs.

    Returns a list of the saved prediction.png paths.
    """
    output_dir = Path(output_dir)
    full_model.eval()
    prediction_paths: list[Path] = []

    saved = 0
    for batch in loader:
        if saved >= n_samples:
            break
        images = batch["image"].to(device)
        logits = full_model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()

        for i in range(len(images)):
            if saved >= n_samples:
                break
            img_np = images[i].cpu().permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = (np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8)
            pred_rgb = _mask_to_rgb(preds[i])

            sample_dir = output_dir / f"sample_{saved:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            Image.fromarray(img_np).save(sample_dir / "input.png")
            pred_path = sample_dir / "prediction.png"
            Image.fromarray(pred_rgb).save(pred_path)
            prediction_paths.append(pred_path)
            saved += 1

    return prediction_paths


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    decoder: FloorplanDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    global_step: int,
) -> tuple[float, int]:
    decoder.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        hidden_states = tuple(hs.to(device, non_blocking=True) for hs in batch["hidden_states"])
        masks         = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = decoder(hidden_states)
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = decoder(hidden_states)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        global_step += 1

    return total_loss / max(n_batches, 1), global_step


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    decoder: FloorplanDecoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    vrs_weights: dict[str, float],
    compute_boundary: bool = True,
    boundary_tol: int = 2,
    compute_grouped: bool = True,
) -> dict[str, float]:
    """Validate decoder; compute IoU, boundary F1, vector_ready_score, grouped metrics."""
    decoder.eval()
    total_loss   = 0.0
    all_preds:   list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_input_types: list[str]      = []

    wall_bf1_acc    = BoundaryF1Accumulator(class_id=1, tolerance_px=boundary_tol)
    opening_bf1_acc = BoundaryF1Accumulator(class_id=2, tolerance_px=boundary_tol)

    # Per-input-type accumulators
    group_preds:   dict[str, list[torch.Tensor]] = {}
    group_targets: dict[str, list[torch.Tensor]] = {}
    group_wall_bf1:    dict[str, BoundaryF1Accumulator] = {}
    group_opening_bf1: dict[str, BoundaryF1Accumulator] = {}

    for batch in loader:
        hidden_states = tuple(hs.to(device, non_blocking=True) for hs in batch["hidden_states"])
        masks         = batch["mask"].to(device, non_blocking=True)

        logits = decoder(hidden_states)
        loss   = criterion(logits, masks)
        total_loss += loss.item()

        preds   = logits.argmax(dim=1).cpu()
        masks_c = masks.cpu()

        all_preds.append(preds)
        all_targets.append(masks_c)

        input_types = list(batch.get("input_type", ["unknown"] * preds.shape[0]))
        all_input_types.extend(input_types)

        # Boundary F1 accumulators (per sample)
        if compute_boundary:
            for i in range(preds.shape[0]):
                p_np = preds[i].numpy()
                t_np = masks_c[i].numpy()
                wall_bf1_acc.update(p_np, t_np)
                opening_bf1_acc.update(p_np, t_np)

                if compute_grouped:
                    itype = input_types[i] if i < len(input_types) else "unknown"
                    if itype not in group_wall_bf1:
                        group_wall_bf1[itype]    = BoundaryF1Accumulator(1, boundary_tol)
                        group_opening_bf1[itype] = BoundaryF1Accumulator(2, boundary_tol)
                    group_wall_bf1[itype].update(p_np, t_np)
                    group_opening_bf1[itype].update(p_np, t_np)

        if compute_grouped:
            for i in range(preds.shape[0]):
                itype = input_types[i] if i < len(input_types) else "unknown"
                group_preds.setdefault(itype, []).append(preds[i])
                group_targets.setdefault(itype, []).append(masks_c[i])

    preds_cat   = torch.cat([p.view(-1) for p in all_preds])
    targets_cat = torch.cat([t.view(-1) for t in all_targets])

    per_class_iou = compute_iou_per_class(preds_cat, targets_cat, num_classes)
    miou          = compute_miou(preds_cat, targets_cat, num_classes)
    fg_miou       = compute_foreground_miou(preds_cat, targets_cat, num_classes)
    pixel_acc     = compute_pixel_accuracy(preds_cat, targets_cat)
    fg_pixel_acc  = compute_foreground_pixel_accuracy(preds_cat, targets_cat)

    metrics: dict[str, float] = {
        "val_loss":                total_loss / max(len(loader), 1),
        "val_mIoU":                miou,
        "foreground_mIoU":         fg_miou,
        "pixel_accuracy":          pixel_acc,
        "foreground_pixel_accuracy": fg_pixel_acc,
        "wall_boundary_F1":        wall_bf1_acc.compute() if compute_boundary else float("nan"),
        "opening_boundary_F1":     opening_bf1_acc.compute() if compute_boundary else float("nan"),
    }
    for cls_id, iou in enumerate(per_class_iou):
        name = DEFAULT_CLASS_MAPPING.get(cls_id, f"class_{cls_id}")
        metrics[f"{name}_IoU"] = iou

    metrics["val_vector_ready_score"] = compute_vector_ready_score(metrics, vrs_weights)

    # Grouped metrics
    if compute_grouped:
        for itype, g_preds in group_preds.items():
            prefix = "clean" if itype == "svg_rendered_clean" else "original"
            gp = torch.cat([p.view(-1) for p in g_preds])
            gt = torch.cat([t.view(-1) for t in group_targets[itype]])
            g_ious = compute_iou_per_class(gp, gt, num_classes)
            metrics[f"{prefix}_pixel_accuracy"]    = compute_pixel_accuracy(gp, gt)
            metrics[f"{prefix}_foreground_mIoU"]   = compute_foreground_miou(gp, gt, num_classes)
            for cls_id, iou in enumerate(g_ious):
                name = DEFAULT_CLASS_MAPPING.get(cls_id, f"class_{cls_id}")
                metrics[f"{prefix}_{name}_IoU"] = iou
            if compute_boundary and itype in group_wall_bf1:
                metrics[f"{prefix}_wall_boundary_F1"]    = group_wall_bf1[itype].compute()
                metrics[f"{prefix}_opening_boundary_F1"] = group_opening_bf1[itype].compute()

    return metrics


# ---------------------------------------------------------------------------
# Preview image saving
# ---------------------------------------------------------------------------


_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (200, 200, 200),
    1: (30,  30,  30),
    2: (200, 80,  80),
    3: (80,  160, 220),
    4: (80,  200, 100),
}


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in _CLASS_COLORS.items():
        rgb[mask == cls_id] = color
    return rgb


@torch.no_grad()
def save_previews(
    full_model: FloorplanSegModel,
    loader: DataLoader,
    device: torch.device,
    preview_dir: Path,
    n_samples: int = 4,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    full_model.eval()

    saved = 0
    for batch in loader:
        if saved >= n_samples:
            break
        images = batch["image"].to(device)
        masks  = batch["mask"]

        logits = full_model(images)
        preds  = logits.argmax(dim=1).cpu()

        for i in range(images.size(0)):
            if saved >= n_samples:
                break
            img_t    = images[i].cpu()
            mean     = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std      = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_disp = ((img_t * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()

            target_rgb = _mask_to_rgb(masks[i].numpy())
            pred_rgb   = _mask_to_rgb(preds[i].numpy())
            overlay    = (img_disp * 0.5 + pred_rgb * 0.5).astype(np.uint8)

            Image.fromarray(img_disp).save(   preview_dir / f"sample_{saved:03d}_input.png")
            Image.fromarray(target_rgb).save( preview_dir / f"sample_{saved:03d}_target.png")
            Image.fromarray(pred_rgb).save(   preview_dir / f"sample_{saved:03d}_prediction.png")
            Image.fromarray(overlay).save(    preview_dir / f"sample_{saved:03d}_overlay.png")
            saved += 1


# ---------------------------------------------------------------------------
# Training-history helpers
# ---------------------------------------------------------------------------


def _write_history_row(csv_path: Path, row: dict) -> None:
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _save_summary(summary_path: Path, data: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(data, f, indent=2)


def _fmt(v: float) -> str | float:
    return round(v, 6) if v == v else "nan"


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(cfg: dict, args: argparse.Namespace) -> None:
    cfg = _normalize_config(cfg)

    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    device = _setup_device(cfg)

    num_classes = cfg["num_classes"]
    image_size  = cfg["image_size"]
    variant     = cfg["model"]["name"]
    pretrained  = cfg["model"].get("pretrained", True)

    # Checkpoint paths
    ckpt_dir     = Path(cfg["checkpoint"]["output_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path  = ckpt_dir / "latest.pt"
    best_path    = ckpt_dir / "best.pt"
    history_csv  = ckpt_dir / "training_history.csv"
    summary_path = ckpt_dir / "training_summary.json"

    log_dir       = Path(cfg["logging"]["log_dir"])
    preview_every = cfg["logging"].get("save_preview_every_n_epochs", 5)
    n_preview     = cfg["logging"].get("preview_sample_count", 4)

    metrics_cfg      = cfg.get("metrics", {})
    compute_boundary = metrics_cfg.get("compute_boundary_f1", True)
    boundary_tol     = metrics_cfg.get("boundary_tolerance_px", 2)
    compute_grouped  = metrics_cfg.get("compute_grouped_by_input_type", True)
    vrs_weights      = metrics_cfg.get("vector_ready_score", {
        "pixel_accuracy": 0.25, "opening_IoU": 0.25, "opening_boundary_F1": 0.15,
        "foreground_mIoU": 0.15, "room_IoU": 0.10, "wall_IoU": 0.05, "icon_IoU": 0.05,
    })

    run_cfg   = cfg.get("run", {})
    run_name  = run_cfg.get("run_name", "segformer_b0_v005")
    model_ver = run_cfg.get("version", "v005")

    # Build backbone (frozen)
    logger.info("Loading SegFormer backbone: %s  pretrained=%s", variant, pretrained)
    backbone = build_backbone(variant=variant, pretrained=pretrained)
    backbone.to(device).eval()

    # Cache backbone features
    overfit_n = getattr(args, "overfit", 0) or 0
    train_loader, val_loader = build_cached_loaders(
        cfg, backbone, device, debug=args.debug, overfit_n=overfit_n
    )
    backbone.to("cpu")
    logger.info("Backbone features cached.  Moving backbone to CPU to free GPU memory.")

    # Build decoder
    decoder = build_decoder(variant=variant, num_classes=num_classes, output_size=image_size)
    decoder.to(device)
    logger.info("Decoder parameters: %d", sum(p.numel() for p in decoder.parameters() if p.requires_grad))

    # Optimizer
    lr = cfg["training"]["learning_rate"]
    wd = cfg["training"].get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=wd)

    epochs    = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weights
    class_weights_tensor = _build_class_weights(cfg, ckpt_dir)
    class_weights_list   = class_weights_tensor.tolist() if class_weights_tensor is not None else None

    # Loss function
    loss_cfg   = cfg.get("loss", {})
    loss_name  = loss_cfg.get("name", "cross_entropy")
    if loss_name == "weighted_ce_plus_dice" or loss_cfg.get("use_dice", False):
        cw = class_weights_tensor.to(device) if class_weights_tensor is not None else None
        criterion: nn.Module = WeightedCEPlusDice(
            num_classes          = num_classes,
            class_weights        = cw,
            ce_weight            = float(loss_cfg.get("ce_weight", 1.0)),
            dice_weight          = float(loss_cfg.get("dice_weight", 0.5)),
            dice_exclude_background = bool(loss_cfg.get("dice_exclude_background", True)),
        )
        logger.info("Loss: WeightedCE(%.1f) + Dice(%.1f)", loss_cfg.get("ce_weight", 1.0), loss_cfg.get("dice_weight", 0.5))
    else:
        cw = class_weights_tensor.to(device) if class_weights_tensor is not None else None
        criterion = nn.CrossEntropyLoss(weight=cw)
        logger.info("Loss: CrossEntropyLoss")

    # Mixed precision
    use_amp = cfg["training"].get("mixed_precision", False) and torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    best_metric_name  = cfg["checkpoint"].get("monitor", "val_vector_ready_score")
    best_metric_value = float("-inf")
    start_epoch       = 0
    global_step       = 0
    history: list[dict] = []
    class_mapping     = DEFAULT_CLASS_MAPPING

    # Resume
    resume_arg  = args.resume or cfg["checkpoint"].get("resume_from", "auto")
    resume_path = resolve_resume_path(ckpt_dir, resume_arg) if resume_arg else None
    if resume_path:
        try:
            payload = load_checkpoint(resume_path, decoder, optimizer, scheduler, device)
            if payload.get("arch_version") != ARCH_VERSION:
                raise RuntimeError(
                    f"Checkpoint arch_version={payload.get('arch_version')!r} != {ARCH_VERSION!r}"
                )
            start_epoch       = payload.get("epoch", 0) + 1
            global_step       = payload.get("global_step", 0)
            best_metric_value = payload.get("best_metric_value", float("-inf"))
            history           = payload.get("history", [])
            logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)
        except Exception as exc:
            logger.warning("Could not resume from %s (%s). Starting fresh.", resume_path, exc)
            start_epoch       = 0
            global_step       = 0
            best_metric_value = float("-inf")
            history           = []
            if history_csv.exists():
                archived = history_csv.with_suffix(".prev.csv")
                history_csv.rename(archived)
                logger.info("Old training_history.csv archived as %s", archived.name)

    # Preview loader
    preview_loader = build_preview_loader(cfg, debug=args.debug, overfit_n=overfit_n, n_samples=n_preview)

    # Full model (backbone + decoder) for preview generation only
    backbone.to(device)
    full_model = FloorplanSegModel(backbone=backbone, decoder=decoder)

    logger.info(
        "Training: %d epochs | device=%s | amp=%s | train_batches=%d | val_batches=%d",
        epochs, device, use_amp, len(train_loader), len(val_loader),
    )

    for epoch in range(start_epoch, epochs):
        full_model.backbone.to("cpu")  # CPU during training (not called during head-only training)

        train_loss, global_step = train_one_epoch(
            decoder, train_loader, optimizer, criterion, device, scaler, global_step
        )

        val_metrics = validate(
            decoder, val_loader, criterion, device, num_classes,
            vrs_weights     = vrs_weights,
            compute_boundary = compute_boundary,
            boundary_tol    = boundary_tol,
            compute_grouped = compute_grouped,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Best-model checkpoint
        monitor_value = val_metrics.get(best_metric_name, val_metrics.get("val_mIoU", float("nan")))
        best_updated  = False
        if monitor_value == monitor_value and monitor_value > best_metric_value:
            best_metric_value = monitor_value
            best_updated      = True
            save_checkpoint(
                best_path, decoder, optimizer, scheduler,
                epoch, global_step, best_metric_value, best_metric_name,
                cfg, class_mapping, history,
                arch_version=ARCH_VERSION,
                class_weights=class_weights_list,
                model_version=model_ver,
                run_name=run_name,
            )

        # Latest checkpoint (every epoch)
        save_checkpoint(
            latest_path, decoder, optimizer, scheduler,
            epoch, global_step, best_metric_value, best_metric_name,
            cfg, class_mapping, history,
            arch_version=ARCH_VERSION,
            class_weights=class_weights_list,
            model_version=model_ver,
            run_name=run_name,
        )

        # History row
        row: dict[str, Any] = {
            "epoch":                     epoch,
            "train_loss":                _fmt(train_loss),
            "val_loss":                  _fmt(val_metrics.get("val_loss", float("nan"))),
            "val_vector_ready_score":    _fmt(val_metrics.get("val_vector_ready_score", float("nan"))),
            "val_mIoU":                  _fmt(val_metrics.get("val_mIoU", float("nan"))),
            "foreground_mIoU":           _fmt(val_metrics.get("foreground_mIoU", float("nan"))),
            "pixel_accuracy":            _fmt(val_metrics.get("pixel_accuracy", float("nan"))),
            "foreground_pixel_accuracy": _fmt(val_metrics.get("foreground_pixel_accuracy", float("nan"))),
            "background_IoU":            _fmt(val_metrics.get("background_IoU", float("nan"))),
            "wall_IoU":                  _fmt(val_metrics.get("wall_IoU", float("nan"))),
            "opening_IoU":               _fmt(val_metrics.get("opening_IoU", float("nan"))),
            "room_IoU":                  _fmt(val_metrics.get("room_IoU", float("nan"))),
            "icon_IoU":                  _fmt(val_metrics.get("icon_IoU", float("nan"))),
            "wall_boundary_F1":          _fmt(val_metrics.get("wall_boundary_F1", float("nan"))),
            "opening_boundary_F1":       _fmt(val_metrics.get("opening_boundary_F1", float("nan"))),
            "clean_pixel_accuracy":      _fmt(val_metrics.get("clean_pixel_accuracy", float("nan"))),
            "clean_foreground_mIoU":     _fmt(val_metrics.get("clean_foreground_mIoU", float("nan"))),
            "clean_wall_IoU":            _fmt(val_metrics.get("clean_wall_IoU", float("nan"))),
            "clean_opening_IoU":         _fmt(val_metrics.get("clean_opening_IoU", float("nan"))),
            "clean_opening_boundary_F1": _fmt(val_metrics.get("clean_opening_boundary_F1", float("nan"))),
            "original_pixel_accuracy":      _fmt(val_metrics.get("original_pixel_accuracy", float("nan"))),
            "original_foreground_mIoU":     _fmt(val_metrics.get("original_foreground_mIoU", float("nan"))),
            "original_wall_IoU":            _fmt(val_metrics.get("original_wall_IoU", float("nan"))),
            "original_opening_IoU":         _fmt(val_metrics.get("original_opening_IoU", float("nan"))),
            "original_opening_boundary_F1": _fmt(val_metrics.get("original_opening_boundary_F1", float("nan"))),
            "learning_rate":             round(current_lr, 8),
            "checkpoint_saved":          True,
            "best_updated":              best_updated,
        }
        _write_history_row(history_csv, row)
        history.append(row)

        # Console log
        vrs = val_metrics.get("val_vector_ready_score", float("nan"))
        logger.info(
            "Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | vrs=%s | "
            "opening_IoU=%s | opening_bF1=%s | lr=%.2e%s",
            epoch + 1, epochs,
            train_loss,
            val_metrics.get("val_loss", float("nan")),
            f"{vrs:.4f}"   if vrs == vrs   else "nan",
            f"{val_metrics.get('opening_IoU', float('nan')):.4f}"
                if val_metrics.get('opening_IoU', float('nan')) == val_metrics.get('opening_IoU', float('nan')) else "nan",
            f"{val_metrics.get('opening_boundary_F1', float('nan')):.4f}"
                if val_metrics.get('opening_boundary_F1', float('nan')) == val_metrics.get('opening_boundary_F1', float('nan')) else "nan",
            current_lr,
            " [BEST]" if best_updated else "",
        )

        # Preview images
        if (epoch + 1) % preview_every == 0 or epoch == 0:
            preview_dir = log_dir / "previews" / f"epoch_{epoch + 1:03d}"
            full_model.backbone.to(device)
            save_previews(full_model, preview_loader, device, preview_dir, n_samples=n_preview)
            full_model.backbone.to("cpu")

        # Summary JSON
        _save_summary(summary_path, {
            "last_epoch":        epoch,
            "global_step":       global_step,
            "best_metric_name":  best_metric_name,
            "best_metric_value": best_metric_value,
            "val_loss":          val_metrics.get("val_loss"),
            "val_vector_ready_score": vrs if vrs == vrs else None,
            "val_mIoU":          val_metrics.get("val_mIoU"),
            "opening_IoU":       val_metrics.get("opening_IoU"),
            "opening_boundary_F1": val_metrics.get("opening_boundary_F1"),
            "arch_version":      ARCH_VERSION,
            "run_name":          run_name,
            "class_weights":     class_weights_list,
        })

    logger.info("Training complete.  Best %s = %.4f", best_metric_name, best_metric_value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train FloorplanDecoder with frozen SegFormer backbone (Phase 1, spec_v005)."
    )
    parser.add_argument("--config",  required=True, type=Path)
    parser.add_argument("--debug",   action="store_true")
    parser.add_argument("--overfit", type=int, default=0, metavar="N")
    parser.add_argument("--resume",  type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config)
    train(cfg, args)


if __name__ == "__main__":
    main()
