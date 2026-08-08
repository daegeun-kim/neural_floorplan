"""Layer L1: run the seven-class SegFormer model over an evaluation population.

Kept out of :mod:`src.evaluation.segmentation` so the metric code stays
importable without torch model loading, and out of the smoke path entirely.

L1 is required *secondary* evidence (spec_v006 §0). It is reported next to the
primary topology numbers, never in place of them.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from src.evaluation.segmentation import NUM_CLASSES, SegmentationAccumulator

IMAGE_SIZE = 512


def _build_model(seg_checkpoint: Path, device: torch.device) -> Any:
    from src.checkpointing import load_checkpoint
    from src.models import FloorplanSegModel, build_backbone, build_decoder

    backbone = build_backbone(variant="segformer_b0", pretrained=True)
    decoder = build_decoder(variant="segformer_b0", num_classes=NUM_CLASSES, output_size=IMAGE_SIZE)
    payload = load_checkpoint(seg_checkpoint, decoder, device=device)
    model = FloorplanSegModel(backbone=backbone, decoder=decoder)
    model.eval().to(device)
    return model, payload


def evaluate_segmentation(
    entries: list[dict],
    dataset_root: Path,
    seg_checkpoint: Path,
    device: str = "cuda",
    on_progress: Any = None,
) -> dict[str, Any]:
    """Score the seven-class model over *entries*.

    Returns the aggregate block described in spec_v006 §4.6, plus the checkpoint
    metadata needed for provenance.
    """
    from src.dataset import FloorplanDataset

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, payload = _build_model(Path(seg_checkpoint), torch_device)

    # FloorplanDataset reads its entry list from a JSON path, so the selected
    # population is materialized to a temporary index rather than reaching into
    # the dataset internals.
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "population_index.json"
        index_path.write_text(json.dumps(entries), encoding="utf-8")
        dataset = FloorplanDataset(
            index_path, Path(dataset_root), image_size=IMAGE_SIZE, augment=False
        )

        accumulator = SegmentationAccumulator()
        skipped: list[dict[str, str]] = []

        with torch.no_grad():
            for i in range(len(dataset)):
                entry = entries[i]
                try:
                    item = dataset[i]
                except Exception as exc:  # noqa: BLE001 - a bad sample is data
                    skipped.append(
                        {
                            "sample_id": entry.get("sample_id", str(i)),
                            "reason": "missing_semantic_reference",
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                image = item["image"].unsqueeze(0).to(torch_device)
                target = item["mask"].to(torch_device)
                logits = model(image)
                preds = logits.argmax(dim=1).squeeze(0)
                accumulator.add(preds.cpu(), target.cpu())
                if on_progress:
                    on_progress(entry.get("sample_id", str(i)))

    aggregate = accumulator.aggregate()
    aggregate["skipped"] = skipped
    aggregate["denominator"] = accumulator.sample_count
    aggregate["checkpoint"] = {
        "arch_version": payload.get("arch_version"),
        "run_name": payload.get("run_name"),
        "epoch": payload.get("epoch"),
        "best_metric_name": payload.get("best_metric_name"),
        "best_metric_value": payload.get("best_metric_value"),
    }
    aggregate["note"] = (
        "Required secondary evidence. Primary claim is wall topology, "
        "circulation, and opening logic."
    )
    return aggregate
