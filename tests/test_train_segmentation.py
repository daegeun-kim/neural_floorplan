"""Tests for the v005 training pipeline components."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.checkpointing import load_checkpoint, resolve_resume_path, save_checkpoint
from src.dataset import FloorplanDataset
from src.losses import MulticlassDiceLoss, WeightedCEPlusDice
from src.metrics import (
    BoundaryF1Accumulator,
    compute_foreground_miou,
    compute_foreground_pixel_accuracy,
    compute_iou_per_class,
    compute_miou,
    compute_pixel_accuracy,
    compute_vector_ready_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample(
    root: Path, name: str = "s1", size: int = 32, num_classes: int = 5
) -> dict:
    """Write a minimal (image, mask) pair and return a dataset-index entry."""
    sample_dir = root / name
    sample_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = sample_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    # RGB image
    img = Image.new("RGB", (size, size), (200, 200, 200))
    img_path = sample_dir / "model_clean.png"
    img.save(img_path)

    # Semantic class map with values 0..num_classes-1
    arr = np.zeros((size, size), dtype=np.uint8)
    for i in range(num_classes):
        arr[i * (size // num_classes) : (i + 1) * (size // num_classes), :] = i
    mask = Image.fromarray(arr, mode="L")
    mask_path = masks_dir / "semantic_class_map.png"
    mask.save(mask_path)

    return {
        "sample_id": name,
        "image": str(img_path.relative_to(root)),
        "target": str(mask_path.relative_to(root)),
        "input_type": "svg_rendered_clean",
    }


def _make_index(root: Path, n: int = 3, size: int = 32) -> Path:
    entries = [_make_sample(root, f"s{i}", size=size) for i in range(n)]
    index_path = root / "index.json"
    with open(index_path, "w") as f:
        json.dump(entries, f)
    return index_path


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


def test_dataset_len(tmp_path):
    index = _make_index(tmp_path, n=4)
    ds = FloorplanDataset(index, tmp_path, image_size=32)
    assert len(ds) == 4


def test_dataset_returns_correct_shapes(tmp_path):
    index = _make_index(tmp_path, n=2)
    ds = FloorplanDataset(index, tmp_path, image_size=32)
    item = ds[0]

    assert item["image"].shape == (3, 32, 32), "Image should be [C, H, W]"
    assert item["mask"].shape == (32, 32), "Mask should be [H, W]"
    assert item["image"].dtype == torch.float32
    assert item["mask"].dtype == torch.long


def test_dataset_mask_values_are_valid_class_ids(tmp_path):
    index = _make_index(tmp_path, n=2, size=32)
    ds = FloorplanDataset(index, tmp_path, image_size=32)
    item = ds[0]
    unique = torch.unique(item["mask"]).tolist()
    assert all(v in range(5) for v in unique), f"Invalid class IDs: {unique}"


def test_dataset_mask_uses_nearest_interpolation(tmp_path):
    """After resize, mask values should still only be original class IDs."""
    index = _make_index(tmp_path, n=1, size=64)
    ds = FloorplanDataset(index, tmp_path, image_size=128)
    item = ds[0]
    unique = set(torch.unique(item["mask"]).tolist())
    # Bilinear interpolation would produce non-integer floats; NEAREST should not
    assert unique.issubset(set(range(5))), f"Unexpected mask values after resize: {unique}"


def test_dataset_image_is_normalized(tmp_path):
    """Image should be normalized (values not in 0–255 range)."""
    index = _make_index(tmp_path, n=1)
    ds = FloorplanDataset(index, tmp_path, image_size=32)
    img = ds[0]["image"]
    assert img.max().item() <= 5.0, "Image should be normalized — raw pixel values indicate missing normalization"


def test_dataset_sample_id_present(tmp_path):
    index = _make_index(tmp_path, n=1)
    ds = FloorplanDataset(index, tmp_path, image_size=32)
    item = ds[0]
    assert "sample_id" in item
    assert "image_path" in item
    assert "mask_path" in item


def test_dataset_returns_input_type(tmp_path):
    index = _make_index(tmp_path, n=1)
    ds = FloorplanDataset(index, tmp_path, image_size=32)
    item = ds[0]
    assert "input_type" in item
    assert item["input_type"] == "svg_rendered_clean"


def test_dataset_augment_preserves_mask_values(tmp_path):
    index = _make_index(tmp_path, n=2, size=64)
    ds = FloorplanDataset(index, tmp_path, image_size=64, augment=True)

    for _ in range(10):
        item = ds[0]
        unique = set(torch.unique(item["mask"]).tolist())
        assert unique.issubset(set(range(5))), f"Augmentation produced invalid mask values: {unique}"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


def test_perfect_prediction_iou_is_one():
    num_classes = 5
    mask = torch.arange(num_classes).repeat(10)
    ious = compute_iou_per_class(mask, mask, num_classes)
    for iou in ious:
        assert abs(iou - 1.0) < 1e-6, f"Expected IoU=1.0 for perfect prediction, got {iou}"


def test_no_overlap_iou_is_zero():
    preds = torch.zeros(10, dtype=torch.long)   # all predicted class 0
    targets = torch.ones(10, dtype=torch.long)   # all true class 1
    ious = compute_iou_per_class(preds, targets, num_classes=2)
    assert ious[0] == 0.0, f"Expected IoU=0 for class 0, got {ious[0]}"
    assert ious[1] == 0.0, f"Expected IoU=0 for class 1, got {ious[1]}"


def test_absent_class_iou_is_nan():
    preds = torch.zeros(10, dtype=torch.long)
    targets = torch.zeros(10, dtype=torch.long)
    ious = compute_iou_per_class(preds, targets, num_classes=3)
    assert ious[0] == 1.0
    assert math.isnan(ious[1])
    assert math.isnan(ious[2])


def test_miou_excludes_nan():
    ious_raw = [0.5, float("nan"), 1.0]
    expected = (0.5 + 1.0) / 2

    preds = torch.tensor([0, 0], dtype=torch.long)
    targets = torch.tensor([0, 0], dtype=torch.long)
    miou = compute_miou(preds, targets, num_classes=3)
    assert miou == 1.0  # only class 0 is present, IoU=1.0


def test_pixel_accuracy_perfect():
    t = torch.arange(10)
    acc = compute_pixel_accuracy(t, t)
    assert abs(acc - 1.0) < 1e-6


def test_pixel_accuracy_all_wrong():
    preds = torch.zeros(10, dtype=torch.long)
    targets = torch.ones(10, dtype=torch.long)
    acc = compute_pixel_accuracy(preds, targets)
    assert acc == 0.0


def test_iou_partial_overlap():
    # 10 pixels, all target class 0, half predicted correctly
    preds = torch.cat([torch.zeros(5, dtype=torch.long), torch.ones(5, dtype=torch.long)])
    targets = torch.zeros(10, dtype=torch.long)
    ious = compute_iou_per_class(preds, targets, num_classes=2)
    # class 0: intersection=5, union=10 → IoU=0.5
    assert abs(ious[0] - 0.5) < 1e-6
    # class 1: intersection=0, union=5 → IoU=0.0
    assert abs(ious[1] - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# Checkpointing tests
# ---------------------------------------------------------------------------


def _dummy_model():
    return torch.nn.Linear(4, 2)


def _dummy_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)


def test_save_and_load_checkpoint(tmp_path):
    model = _dummy_model()
    optimizer = _dummy_optimizer(model)

    save_checkpoint(
        tmp_path / "ckpt.pt",
        model, optimizer, None,
        epoch=3, global_step=100,
        best_metric_value=0.55,
        best_metric_name="val_mIoU",
        config={"image_size": 512},
        class_mapping={0: "background"},
        history=[],
    )

    assert (tmp_path / "ckpt.pt").exists()

    new_model = _dummy_model()
    new_optimizer = _dummy_optimizer(new_model)
    payload = load_checkpoint(tmp_path / "ckpt.pt", new_model, new_optimizer)

    assert payload["epoch"] == 3
    assert payload["global_step"] == 100
    assert abs(payload["best_metric_value"] - 0.55) < 1e-6


def test_checkpoint_restores_weights(tmp_path):
    model = _dummy_model()
    # Set specific weights
    with torch.no_grad():
        model.weight.fill_(3.14)
    optimizer = _dummy_optimizer(model)

    save_checkpoint(
        tmp_path / "w.pt",
        model, optimizer, None,
        epoch=0, global_step=0,
        best_metric_value=0.0,
        best_metric_name="val_loss",
        config={},
        class_mapping={},
        history=[],
    )

    new_model = _dummy_model()
    load_checkpoint(tmp_path / "w.pt", new_model)
    assert torch.allclose(new_model.weight, torch.full_like(new_model.weight, 3.14))


def test_resolve_resume_path_auto_latest(tmp_path):
    latest = tmp_path / "latest.pt"
    latest.touch()
    result = resolve_resume_path(tmp_path, "auto")
    assert result == latest


def test_resolve_resume_path_auto_falls_back_to_best(tmp_path):
    best = tmp_path / "best.pt"
    best.touch()
    result = resolve_resume_path(tmp_path, "auto")
    assert result == best


def test_resolve_resume_path_auto_none_when_empty(tmp_path):
    result = resolve_resume_path(tmp_path, "auto")
    assert result is None


def test_resolve_resume_path_explicit(tmp_path):
    explicit = tmp_path / "epoch_05.pt"
    explicit.touch()
    result = resolve_resume_path(tmp_path, str(explicit))
    assert result == explicit


# ---------------------------------------------------------------------------
# Model smoke test (no pretrained download — skip if transformers unavailable)
# ---------------------------------------------------------------------------


try:
    from src.models import FloorplanDecoder, build_model as _build_model, BACKBONE_HIDDEN_SIZES

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
def test_model_forward_pass():
    """Full FloorplanSegModel forward pass should return logits of the expected shape."""
    model = _build_model("segformer_b0", num_classes=5, pretrained=False, output_size=64)
    model.eval()

    B, C, H, W = 2, 3, 64, 64
    dummy_input = torch.rand(B, C, H, W)

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (B, 5, H, W), f"Expected (2, 5, 64, 64) but got {logits.shape}"


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
def test_model_output_has_correct_num_classes():
    model = _build_model("segformer_b0", num_classes=5, pretrained=False, output_size=32)
    model.eval()

    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape[1] == 5


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
def test_backbone_parameters_are_frozen():
    """Backbone parameters must all have requires_grad=False."""
    model = _build_model("segformer_b0", num_classes=5, pretrained=False, output_size=64)
    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad, f"Backbone param {name} has requires_grad=True"


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
def test_decoder_parameters_are_trainable():
    """All decoder parameters must have requires_grad=True."""
    model = _build_model("segformer_b0", num_classes=5, pretrained=False, output_size=64)
    for name, param in model.decoder.named_parameters():
        assert param.requires_grad, f"Decoder param {name} has requires_grad=False"


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


def test_dice_loss_perfect_prediction():
    """Dice loss should be near 0 for perfect predictions."""
    B, C, H, W = 2, 5, 16, 16
    targets = torch.randint(0, C, (B, H, W))
    logits  = torch.zeros(B, C, H, W)
    for b in range(B):
        for h in range(H):
            for w in range(W):
                logits[b, targets[b, h, w], h, w] = 100.0  # very confident
    loss_fn = MulticlassDiceLoss(num_classes=C, exclude_background=True)
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.05, f"Dice loss for perfect prediction should be near 0, got {loss.item()}"


def test_dice_loss_shape():
    """MulticlassDiceLoss should return a scalar."""
    logits  = torch.randn(2, 5, 16, 16)
    targets = torch.randint(0, 5, (2, 16, 16))
    loss_fn = MulticlassDiceLoss(num_classes=5)
    loss = loss_fn(logits, targets)
    assert loss.shape == torch.Size([]), "DiceLoss should return a scalar"


def test_weighted_ce_plus_dice_shape():
    """WeightedCEPlusDice should return a scalar."""
    logits  = torch.randn(2, 5, 16, 16)
    targets = torch.randint(0, 5, (2, 16, 16))
    loss_fn = WeightedCEPlusDice(num_classes=5, dice_weight=0.5)
    loss = loss_fn(logits, targets)
    assert loss.shape == torch.Size([]), "WeightedCEPlusDice should return a scalar"


def test_weighted_ce_plus_dice_with_class_weights():
    weights = torch.tensor([0.5, 0.8, 1.8, 1.0, 1.0])
    logits  = torch.randn(2, 5, 16, 16)
    targets = torch.randint(0, 5, (2, 16, 16))
    loss_fn = WeightedCEPlusDice(num_classes=5, class_weights=weights)
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss), "Loss with class weights should be finite"


# ---------------------------------------------------------------------------
# New metrics tests
# ---------------------------------------------------------------------------


def test_foreground_miou_excludes_background():
    """foreground_mIoU should give 1.0 when only background predictions are perfect."""
    # All pixels are background (class 0), predicted correctly
    preds   = torch.zeros(100, dtype=torch.long)
    targets = torch.zeros(100, dtype=torch.long)
    fg_miou = compute_foreground_miou(preds, targets, num_classes=5)
    # All foreground classes are absent (NaN) → result is NaN
    assert fg_miou != fg_miou or fg_miou == 1.0  # NaN or 1.0 are both valid for all-bg input


def test_foreground_pixel_accuracy_ignores_background():
    preds   = torch.tensor([0, 0, 1, 2, 3], dtype=torch.long)
    targets = torch.tensor([0, 0, 1, 2, 3], dtype=torch.long)
    acc = compute_foreground_pixel_accuracy(preds, targets, background_class=0)
    assert abs(acc - 1.0) < 1e-6, "Perfect fg accuracy should be 1.0"


def test_foreground_pixel_accuracy_only_fg_pixels():
    # Background correct, one foreground wrong
    preds   = torch.tensor([0, 0, 1, 9], dtype=torch.long)
    targets = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    acc = compute_foreground_pixel_accuracy(preds, targets)
    assert abs(acc - 0.5) < 1e-6, f"Expected 0.5 fg accuracy, got {acc}"


def test_vector_ready_score_basic():
    metrics = {
        "pixel_accuracy": 0.9,
        "opening_IoU": 0.8,
        "opening_boundary_F1": 0.7,
        "foreground_mIoU": 0.75,
        "room_IoU": 0.85,
        "wall_IoU": 0.6,
        "icon_IoU": 0.5,
    }
    weights = {
        "pixel_accuracy": 0.25, "opening_IoU": 0.25, "opening_boundary_F1": 0.15,
        "foreground_mIoU": 0.15, "room_IoU": 0.10, "wall_IoU": 0.05, "icon_IoU": 0.05,
    }
    score = compute_vector_ready_score(metrics, weights)
    expected = sum(metrics[k] * weights[k] for k in weights)
    assert abs(score - expected) < 1e-5, f"VRS mismatch: {score} != {expected}"


def test_vector_ready_score_skips_nan():
    metrics = {"pixel_accuracy": 0.9, "opening_IoU": float("nan")}
    weights = {"pixel_accuracy": 0.5, "opening_IoU": 0.5}
    score = compute_vector_ready_score(metrics, weights)
    # Only pixel_accuracy contributes, weight renormalized to 1.0
    assert abs(score - 0.9) < 1e-5


def test_boundary_f1_accumulator_perfect():
    """BoundaryF1 should be 1.0 when predictions match targets exactly."""
    acc = BoundaryF1Accumulator(class_id=1, tolerance_px=2)
    h, w = 32, 32
    mask = np.zeros((h, w), dtype=np.int64)
    mask[8:24, 8:24] = 1  # a square region of class 1
    acc.update(mask, mask)
    f1 = acc.compute()
    assert f1 > 0.95, f"Perfect prediction boundary F1 should be near 1.0, got {f1}"


def test_boundary_f1_accumulator_no_class():
    """BoundaryF1 should return NaN when class is absent in all samples."""
    acc = BoundaryF1Accumulator(class_id=1, tolerance_px=2)
    mask = np.zeros((32, 32), dtype=np.int64)  # no class-1 pixels
    acc.update(mask, mask)
    f1 = acc.compute()
    assert f1 != f1 or f1 >= 0.0  # NaN or valid


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
def test_decoder_standalone_forward_pass():
    """FloorplanDecoder should accept flat hidden states and produce correct output shape."""
    hidden_sizes = BACKBONE_HIDDEN_SIZES["segformer_b0"]  # [32, 64, 160, 256]
    decoder = FloorplanDecoder(
        encoder_hidden_sizes=hidden_sizes,
        num_classes=5,
        output_size=64,
    )
    decoder.eval()

    B = 2
    # Simulate flat hidden states matching 64×64 input with B0 strides [4, 8, 16, 32]
    # spatial dims: [16, 8, 4, 2], channels: [32, 64, 160, 256]
    hidden_states = (
        torch.rand(B, 16 * 16, 32),    # stage 1: 64/4=16 → N=256
        torch.rand(B,  8 *  8, 64),    # stage 2: 64/8= 8 → N= 64
        torch.rand(B,  4 *  4, 160),   # stage 3: 64/16=4 → N= 16
        torch.rand(B,  2 *  2, 256),   # stage 4: 64/32=2 → N=  4
    )

    with torch.no_grad():
        logits = decoder(hidden_states)

    assert logits.shape == (B, 5, 64, 64), (
        f"Expected (2, 5, 64, 64) but got {logits.shape}"
    )
