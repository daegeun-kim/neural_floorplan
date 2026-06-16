"""Load prediction preview images from the configured directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def find_prediction_images(preview_dir: str | Path, filename_contains: str = "prediction") -> list[Path]:
    preview_dir = Path(preview_dir)
    if not preview_dir.exists():
        raise FileNotFoundError(f"Preview directory not found: {preview_dir}")
    matches = sorted(
        p for p in preview_dir.iterdir()
        if p.is_file() and filename_contains in p.name and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    return matches


def load_image_as_array(image_path: str | Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    return np.array(img, dtype=np.uint8)
