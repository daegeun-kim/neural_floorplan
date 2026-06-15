"""Build train/val/test split JSON files from the prepared dataset (spec_v005, task02).

Scans the dataset root for sample folders that have:
  - F1_scaled.png    (original CubiCasa raster — primary real-world input)
  - model_clean.png  (SVG-rendered raster with all floors visible, task01)
  - masks/semantic_class_map.png

Each valid (image, mask) pair becomes one dataset entry.  Both image types
exist in a folder, they produce two separate entries sharing one mask.

Usage:
    python -m src.build_splits <dataset_root> --train 0.8 --val 0.1 --test 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Order matters: F1_scaled first so sample_ids are consistent
CLEAN_IMAGE_NAMES = ["F1_scaled.png", "model_clean.png"]
MASKS_DIR = "masks"

_INPUT_TYPE: dict[str, str] = {
    "F1_scaled.png":   "original_raster",
    "model_clean.png": "svg_rendered_clean",
}


def collect_entries(root_dir: Path) -> list[dict]:
    """Walk root_dir and return dataset index entries."""
    entries: list[dict] = []
    sample_dirs = sorted(d for d in root_dir.iterdir() if d.is_dir())

    for sample_dir in sample_dirs:
        mask_path = sample_dir / MASKS_DIR / "semantic_class_map.png"
        if not mask_path.exists():
            continue

        for img_name in CLEAN_IMAGE_NAMES:
            img_path = sample_dir / img_name
            if not img_path.exists():
                continue

            input_type = _INPUT_TYPE.get(img_name, "unknown")
            sample_id = f"{sample_dir.name}_{img_name.replace('.', '_')}"

            entries.append(
                {
                    "sample_id": sample_id,
                    "image": str(img_path.relative_to(root_dir)),
                    "target": str(mask_path.relative_to(root_dir)),
                    "source_svg": str((sample_dir / "model.svg").relative_to(root_dir)),
                    "input_type": input_type,
                }
            )

    return entries


def split_entries(
    entries: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = entries.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def write_split(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"  {path}  ({len(entries)} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset split JSON files.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--debug-train-n", type=int, default=20,
                        help="Number of samples for debug_train.json")
    parser.add_argument("--debug-val-n", type=int, default=5,
                        help="Number of samples for debug_val.json")
    args = parser.parse_args()

    print(f"Scanning {args.dataset_root} ...")
    entries = collect_entries(args.dataset_root)
    print(f"Found {len(entries)} valid (image, mask) pairs.")

    if not entries:
        print("No valid entries found.  Run generate_semantic_masks.py first.")
        return

    train, val, test = split_entries(entries, args.train, args.val, seed=args.seed)
    print("Writing splits:")
    write_split(args.splits_dir / "train.json", train)
    write_split(args.splits_dir / "val.json", val)
    write_split(args.splits_dir / "test.json", test)

    # Debug subsets
    rng = random.Random(args.seed)
    debug_train = rng.sample(train, min(args.debug_train_n, len(train)))
    debug_val = rng.sample(val, min(args.debug_val_n, len(val)))
    write_split(args.splits_dir / "debug_train.json", debug_train)
    write_split(args.splits_dir / "debug_val.json", debug_val)

    print("Done.")


if __name__ == "__main__":
    main()
