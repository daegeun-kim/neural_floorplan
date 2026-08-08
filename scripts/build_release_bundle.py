"""Build the local GitHub Release bundle for the trained segmentation model.

Produces an inference-only, weight-only bundle under an ignored ``dist/`` path
for the repository owner to upload manually. This script never contacts GitHub
and never performs a Git operation.

    python scripts/build_release_bundle.py \
        --checkpoint checkpoints_CNN/segformer_b0_run3/best.pt \
        --config configs/train_segformer_b0_run3.yaml \
        --results evaluation/results/v1 \
        --out dist

What the bundle deliberately excludes:

  * optimizer and scheduler state, training history, and global step
  * the CubiCasa5K dataset and anything derived from it
  * the upstream Raster-to-Graph checkpoint (separate upstream source)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BUNDLE_VERSION = "v1.0.0"
BUNDLE_NAME = f"neural-floorplan-segformer-b0-run3-{BUNDLE_VERSION}"
WEIGHTS_FILENAME = "floorplan_decoder.safetensors"

CLASS_MAPPING = {
    "0": "background",
    "1": "floor",
    "2": "wall",
    "3": "window",
    "4": "door_arc",
    "5": "door_leaf",
    "6": "door_origin",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_weights(checkpoint: Path, destination: Path) -> dict:
    """Write the trained decoder head as safetensors and verify equivalence."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {k: v.contiguous() for k, v in payload["model_state_dict"].items()}

    save_file(state, str(destination))

    # A silently corrupted export would be worse than no export at all.
    reloaded = load_file(str(destination))
    if set(reloaded) != set(state):
        raise RuntimeError("safetensors round-trip changed the tensor set")
    for key, original in state.items():
        if not torch.equal(original, reloaded[key]):
            raise RuntimeError(f"safetensors round-trip altered tensor {key!r}")

    return {
        "tensor_count": len(state),
        "parameter_count": int(sum(t.numel() for t in state.values())),
        "arch_version": payload.get("arch_version"),
        "run_name": payload.get("run_name"),
        "epoch": payload.get("epoch"),
        "best_metric_name": payload.get("best_metric_name"),
        "best_metric_value": payload.get("best_metric_value"),
        "source_checkpoint_sha256": sha256_of(checkpoint),
    }


def build_config(weights_meta: dict) -> dict:
    return {
        "name": BUNDLE_NAME,
        "version": BUNDLE_VERSION,
        "task": "seven-class floor plan semantic segmentation",
        "architecture": {
            "backbone": {
                "source": "huggingface",
                "model_id": "nvidia/mit-b0",
                "trainable": False,
                "note": (
                    "Frozen. NOT included in this bundle and not trained here. "
                    "Downloaded from Hugging Face at load time under NVIDIA's "
                    "own license terms."
                ),
            },
            "decoder": {
                "class": "FloorplanDecoder",
                "module": "src.models",
                "trainable": True,
                "weights_file": WEIGHTS_FILENAME,
                "tensor_count": weights_meta["tensor_count"],
                "parameter_count": weights_meta["parameter_count"],
            },
            "arch_version": weights_meta["arch_version"],
        },
        "num_classes": len(CLASS_MAPPING),
        "class_mapping": CLASS_MAPPING,
        "preprocessing": {
            "image_size": 512,
            "color_mode": "RGB",
            "resize": "bilinear to 512x512",
            "mask_resize": "nearest",
            "normalization": "see build_image_transform in src/dataset.py",
        },
        "training": {
            "run_name": weights_meta["run_name"],
            "epoch": weights_meta["epoch"],
            "selection_metric": weights_meta["best_metric_name"],
            "selection_metric_value": weights_meta["best_metric_value"],
            "dataset": "CubiCasa5K high_quality_architectural (NOT redistributed)",
        },
        "provenance": {
            "source_checkpoint_sha256": weights_meta["source_checkpoint_sha256"],
            "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    }


MODEL_CARD = """# Neural Floorplan — SegFormer-B0 Seven-Class Segmentation ({version})

Semantic segmentation head for architectural floor plan rasters. Produces a
seven-class label map used as opening and wall evidence by the Neural Floorplan
Phase 4 vectorization pipeline.

## What is in this bundle

Only the **trained decoder head** (`{weights}`, {params:,} parameters).

The SegFormer backbone (`nvidia/mit-b0`) is **frozen**, was **not** trained
here, and is **not** included. It is downloaded from Hugging Face at load time
and remains subject to NVIDIA's own license terms.

The upstream Raster-to-Graph wall-graph checkpoint is **not** included. Obtain
it from <https://github.com/SizheHu/Raster-to-Graph>.

## Classes

| id | name | meaning |
|---|---|---|
| 0 | background | outside the plan |
| 1 | floor | enclosed floor area |
| 2 | wall | wall body |
| 3 | window | window opening |
| 4 | door_arc | door swing arc |
| 5 | door_leaf | door leaf |
| 6 | door_origin | door threshold segment |

## Intended use

* Research and portfolio demonstration of floor-plan-to-CAD vectorization.
* Producing semantic evidence for wall/opening reconstruction on clean,
  SVG-rendered architectural floor plans.

## Out-of-scope use

* Production or safety-critical architectural work.
* Legal, structural, code-compliance, or as-built documentation.
* Photographs, hand sketches, perspective drawings, or heavily degraded scans.
  The model was trained on clean CubiCasa5K renderings and is not validated
  outside that distribution.
* Any use requiring pixel-perfect boundaries — that is explicitly not this
  project's goal.

## Training data

CubiCasa5K, `high_quality_architectural` subset
(<https://github.com/CubiCasa/CubiCasa5k>). The dataset is **not** redistributed
in this bundle or in the source repository, and must be obtained separately
under its own terms.

## Evaluation results

Measured on the leakage-free test population defined in
`specs/spec_v006_evaluation.md` §1.4. Full protocol, denominators, and the
primary topology metrics are in the repository under `evaluation/results/`.

{results_table}

These are **secondary** metrics. The project's primary claim is wall topology,
circulation, and opening logic — not pixel agreement.

## Limitations

* The shipped train/val/test splits leak at the plan level: each floor plan has
  two renderings that were split independently. Published numbers therefore use
  a reduced, leakage-free subset ({seg_n} segmentation entries), which is small.
* Door sub-classes (`door_leaf`, `door_origin`) are the weakest classes. They
  are thin, symbolic strokes covering very few pixels.
* Only the decoder was trained; representation quality is bounded by the frozen
  `nvidia/mit-b0` backbone.
* Performance on `original_raster` scans is weaker than on clean SVG renderings.

## Loading and inference

```python
import torch
from safetensors.torch import load_file
from src.models import FloorplanSegModel, build_backbone, build_decoder

backbone = build_backbone(variant="segformer_b0", pretrained=True)  # nvidia/mit-b0
decoder = build_decoder(variant="segformer_b0", num_classes=7, output_size=512)
decoder.load_state_dict(load_file("{weights}"))

model = FloorplanSegModel(backbone=backbone, decoder=decoder).eval()

# image: float tensor [1, 3, 512, 512], preprocessed per config.json
with torch.no_grad():
    class_map = model(image).argmax(dim=1)   # [1, 512, 512], values 0-6
```

`safetensors` is used because it loads without executing pickled Python, unlike
a raw `torch.load` of a `.pt` file.

## License and attribution

This bundle is distributed under **GPL-3.0-only**; see `LICENSE`. `NOTICE`
carries the full third-party attribution, including the vendored GPL-3.0
Raster-to-Graph code, the Apache-2.0 Deformable DETR / DETR notices it contains,
and the separate terms covering the `nvidia/mit-b0` backbone and the CubiCasa5K
dataset.

Model weights are not source code and are not covered by the GPL grant; the
frozen backbone this head depends on carries NVIDIA's own terms.
"""


def results_table(results_dir: Path) -> tuple[str, int]:
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        return "_No evaluation summary was available at bundle build time._", 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seg = summary.get("secondary_segmentation")
    if not isinstance(seg, dict) or "macro" in seg is None:
        return "_No segmentation results were available._", 0

    macro = seg.get("macro", {})
    micro = seg.get("micro", {})
    boundary = seg.get("boundary_f1", {})
    per_class = macro.get("per_class_iou", {})

    lines = [
        "| class | IoU (macro) | IoU (micro) | boundary F1 |",
        "|---|---|---|---|",
    ]
    for name in CLASS_MAPPING.values():
        lines.append(
            f"| {name} | {_fmt(per_class.get(name))} | "
            f"{_fmt(micro.get('per_class_iou', {}).get(name))} | "
            f"{_fmt(boundary.get(name))} |"
        )
    lines.append("")
    lines.append(
        f"mean IoU (background included): **{_fmt(macro.get('miou_background_included'))}** "
        f"· foreground mean IoU: **{_fmt(macro.get('foreground_miou'))}** "
        f"· pixel accuracy: **{_fmt(macro.get('pixel_accuracy'))}**"
    )
    lines.append("")
    lines.append(f"Evaluated on {seg.get('denominator', 0)} leakage-free entries.")
    return "\n".join(lines), int(seg.get("denominator", 0))


def _fmt(value: object) -> str:
    if value is None or not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("evaluation/results/v1"))
    parser.add_argument("--out", type=Path, default=Path("dist"))
    args = parser.parse_args()

    for path, label in ((args.checkpoint, "checkpoint"), (args.config, "train config")):
        if not path.exists():
            print(f"error: {label} not found: {path}", file=sys.stderr)
            return 2

    bundle_dir = args.out / BUNDLE_NAME
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    weights_meta = export_weights(args.checkpoint, bundle_dir / WEIGHTS_FILENAME)
    config = build_config(weights_meta)
    (bundle_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    shutil.copy2(args.config, bundle_dir / "train_config.yaml")
    for legal in ("LICENSE", "NOTICE"):
        source = PROJECT_ROOT / legal
        if source.exists():
            shutil.copy2(source, bundle_dir / legal)

    summary_src = args.results / "summary.json"
    if summary_src.exists():
        shutil.copy2(summary_src, bundle_dir / "evaluation_summary.json")

    table, seg_n = results_table(args.results)
    (bundle_dir / "MODEL_CARD.md").write_text(
        MODEL_CARD.format(
            version=BUNDLE_VERSION,
            weights=WEIGHTS_FILENAME,
            params=weights_meta["parameter_count"],
            results_table=table,
            seg_n=seg_n,
        ),
        encoding="utf-8",
    )

    checksums = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{sha256_of(path)}  {path.relative_to(bundle_dir).as_posix()}")
    (bundle_dir / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")

    print(f"bundle: {bundle_dir}")
    for line in checksums:
        digest, name = line.split("  ", 1)
        size = (bundle_dir / name).stat().st_size
        print(f"  {name:28s} {size / 1024:8.1f} KB  {digest[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
