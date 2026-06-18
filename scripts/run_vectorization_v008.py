"""Runner script for the v008 vectorization workflow.

Equivalent to executing notebooks/run_vectorization_v008_run1.ipynb.
Imports from src/ — does not duplicate pipeline logic.

Output layout:
    outputs/vectorization/v008/run1/sample_NNN/{input.png, prediction.png, vector.svg}
    outputs/vectorization/v008/run2/sample_NNN/{input.png, prediction.png, vector.svg}

Usage:
    python scripts/run_vectorization_v008.py --run run1
    python scripts/run_vectorization_v008.py --run run2
    python scripts/run_vectorization_v008.py --run run2 --output-name iteration3_run2
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpointing import load_checkpoint
from src.models import FloorplanSegModel, build_backbone, build_decoder
from src.train_segmentation import make_preview_loader, save_sample_artifacts
from src.vectorization.run_mask_to_vector import (
    _scale_info_from_config,
    load_config,
    process_single,
)

SUPPORTED_RUNS = {"run1", "run2"}
IMAGE_SIZE  = 512
NUM_CLASSES = 5
N_SAMPLES   = 4


def main(model_run: str, output_name: str | None = None) -> None:
    assert model_run in SUPPORTED_RUNS, f"Unsupported run: {model_run!r}. Choose from {SUPPORTED_RUNS}"

    checkpoint_path = PROJECT_ROOT / f"checkpoints/segformer_b0_{model_run}/best.pt"
    output_dir      = PROJECT_ROOT / "outputs/vectorization/v008" / (output_name or model_run)
    train_config    = PROJECT_ROOT / "configs/train_segformer_b0.yaml"
    vectz_config    = PROJECT_ROOT / "configs/vectorization_v008.yaml"

    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    print(f"MODEL_RUN  : {model_run}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Output dir : {output_dir}")

    # --- Load vectorization config ---
    vcfg       = load_config(vectz_config)
    scale_info = _scale_info_from_config(vcfg)
    print(f"Config     : {vectz_config.name}")

    # --- Load model ---
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    backbone = build_backbone(variant="segformer_b0", pretrained=True)
    decoder  = build_decoder(variant="segformer_b0", num_classes=NUM_CLASSES, output_size=IMAGE_SIZE)

    payload = load_checkpoint(checkpoint_path, decoder, device=device)
    print(f"arch_version : {payload.get('arch_version')}")
    print(f"epoch        : {payload.get('epoch')}")
    print(f"best metric  : {payload.get('best_metric_name')} = {payload.get('best_metric_value', 0.0):.4f}")

    model = FloorplanSegModel(backbone=backbone, decoder=decoder)
    model.eval().to(device)

    # --- Load preview samples ---
    loader = make_preview_loader(train_config, n_samples=N_SAMPLES)
    print(f"Preview samples : {N_SAMPLES} from val split")

    # --- Generate predictions ---
    # Only clear this run's subfolder; other run folders are left untouched.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning inference...")
    prediction_paths = save_sample_artifacts(model, loader, device, output_dir, n_samples=N_SAMPLES)
    for p in prediction_paths:
        print(f"  {p.relative_to(PROJECT_ROOT)}")

    # --- Vectorize ---
    print("\nRunning vectorization...")
    for pred_path in prediction_paths:
        sample_dir = pred_path.parent
        process_single(pred_path, vcfg, scale_info, sample_dir, output_filename="vector.svg")

    # --- Verify ---
    print("\nOutput structure:")
    expected = ["input.png", "prediction.png", "vector.svg"]
    all_ok = True
    for i in range(N_SAMPLES):
        sample_dir = output_dir / f"sample_{i:03d}"
        statuses = []
        for name in expected:
            ok = (sample_dir / name).exists()
            if not ok:
                all_ok = False
            statuses.append(f"{name}:{'OK' if ok else 'MISSING'}")
        print(f"  sample_{i:03d}  |  {' | '.join(statuses)}")

    print()
    if all_ok:
        print("All artifacts present.")
    else:
        print("WARNING: some artifacts are missing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run v008 vectorization pipeline")
    parser.add_argument("--run", default="run1", choices=list(SUPPORTED_RUNS),
                        help="Model run to use (default: run1)")
    parser.add_argument("--output-name", default=None,
                        help="Output subfolder name under outputs/vectorization/v008/ "
                             "(default: same as --run)")
    args = parser.parse_args()
    main(args.run, args.output_name)
