"""PyTorch Dataset for floorplan semantic segmentation (spec_v005)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

# Allow PIL to load slightly truncated PNG files instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

# ImageNet normalization for pretrained SegFormer backbone
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


class FloorplanDataset(Dataset):
    """Loads (image, mask) pairs from a JSON split index.

    Each entry in the index must have:
        "image"  — path relative to dataset_root
        "target" — path relative to dataset_root
    """

    def __init__(
        self,
        index_path: str | Path,
        dataset_root: str | Path,
        image_size: int = 512,
        augment: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size
        self.augment = augment

        with open(index_path) as f:
            self.entries: list[dict[str, Any]] = json.load(f)

        self.image_transform = build_image_transform(image_size)
        self._aug_spatial = transforms.RandomApply(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ],
            p=0.8,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Retry up to len(dataset) times to skip corrupted files
        for attempt in range(len(self.entries)):
            real_idx = (idx + attempt) % len(self.entries)
            entry = self.entries[real_idx]
            image_path = self.dataset_root / entry["image"]
            mask_path = self.dataset_root / entry["target"]
            try:
                with Image.open(image_path) as img:
                    image = img.convert("RGB")
                with Image.open(mask_path) as m:
                    mask = m.convert("L")
                break
            except Exception as exc:
                logger.warning("Skipping corrupted file %s: %s", image_path, exc)
        else:
            raise RuntimeError(f"No valid sample found starting from idx={idx}")

        # Resize mask using NEAREST to preserve class IDs
        mask_resized = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.augment:
            image, mask_resized = self._apply_spatial_augment(image, mask_resized)

        image_tensor = self.image_transform(image)
        mask_tensor = torch.as_tensor(np.array(mask_resized), dtype=torch.long)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": entry.get("sample_id", str(idx)),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "input_type": entry.get("input_type", "unknown"),
        }

    def _apply_spatial_augment(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        """Apply the same geometric transform to image and mask."""
        import random

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        k = random.choice([0, 1, 2, 3])
        if k:
            image = image.rotate(k * 90, expand=True).resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
            mask = mask.rotate(k * 90, expand=True).resize(
                (self.image_size, self.image_size), Image.NEAREST
            )

        return image, mask
