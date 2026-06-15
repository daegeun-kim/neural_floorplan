"""Backbone feature caching utilities for Phase-1 frozen-backbone training (spec_v005).

Workflow (spec §7 "Frozen Backbone Feature Cache Mode"):
  1. Load pretrained SegFormer backbone and freeze its parameters.
  2. Run each training image through the backbone ONCE.
  3. Save extracted feature tensors to disk as features/<sample_id>.pt.
  4. During head-only training, load cached features instead of calling the backbone.
  5. The backbone forward pass is NEVER called during training epochs.

Cache format (each .pt file):
  {
    "hidden_states":            list of 4 float16 tensors [N_i, C_i],
    "sample_id":                str,
    "image_path":               str,
    "target_mask_path":         str,
    "feature_shape":            list of [N_i, C_i],
    "backbone_name":            str,
    "preprocessing_config_hash": str (12-char md5),
  }

Cache regeneration triggers (spec §7):
  - image preprocessing changes  (image_size, normalization)
  - backbone architecture changes (model name / pretrained flag)
  - input resolution changes
  - force_rebuild=True
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from src.dataset import build_image_transform
from src.models import SegFormerBackboneExtractor

# Allow PIL to load slightly truncated PNG files
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config hash (cache invalidation key)
# ---------------------------------------------------------------------------


def compute_config_hash(config: dict) -> str:
    """Return a 12-char hash of the fields that affect backbone output.

    If any of these fields change the cached features must be regenerated:
      - image_size
      - model.name
      - model.pretrained
      - ImageNet normalization constants (hardcoded for now)
    """
    relevant: dict = {
        "image_size": config.get("image_size"),
        "model_name": config.get("model", {}).get("name"),
        "pretrained":  config.get("model", {}).get("pretrained"),
        # ImageNet normalization is fixed for pretrained SegFormer backbones
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
    }
    raw = json.dumps(relevant, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()[:12]  # noqa: S324  (non-security use)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_id_from_entry(entry: dict) -> str:
    """Derive a cache-safe sample ID from an index entry."""
    return entry.get(
        "sample_id",
        entry["image"].replace("/", "_").replace("\\", "_").replace(".", "_"),
    )


def _cache_path(cache_dir: Path, sample_id: str) -> Path:
    return cache_dir / f"{sample_id}.pt"


def _is_cache_valid(path: Path, expected_hash: str) -> bool:
    """Return True if the .pt file exists and its config hash matches."""
    if not path.exists():
        return False
    try:
        meta = torch.load(path, map_location="cpu", weights_only=False)
        return meta.get("preprocessing_config_hash") == expected_hash
    except Exception:
        return False


def _all_cached(cache_dir: Path, entries: list[dict], config_hash: str) -> bool:
    """Return True only when every entry already has a valid cache file."""
    return all(
        _is_cache_valid(_cache_path(cache_dir, _sample_id_from_entry(e)), config_hash)
        for e in entries
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_features_for_split(
    backbone: SegFormerBackboneExtractor,
    index_path: str | Path,
    dataset_root: str | Path,
    cache_dir: str | Path,
    config: dict,
    force_rebuild: bool = False,
    device: torch.device | None = None,
) -> None:
    """Extract and cache backbone features for all samples in an index file.

    Features are stored in **float16** to reduce disk usage (~2 MB/sample for B0 at 512²).
    Skips samples that already have a valid up-to-date cache file.

    Args:
        backbone:      Frozen :class:`SegFormerBackboneExtractor`.
        index_path:    Path to the split JSON index (list of entry dicts).
        dataset_root:  Root directory that index paths are relative to.
        cache_dir:     Directory in which ``.pt`` cache files are stored.
        config:        Full training config dict (used to compute hash).
        force_rebuild: Re-extract even when cache files are up-to-date.
        device:        Torch device; auto-selects CUDA if available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir   = Path(cache_dir)
    dataset_root = Path(dataset_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(index_path) as f:
        entries: list[dict] = json.load(f)

    config_hash = compute_config_hash(config)

    # Fast-path: all files already cached
    if not force_rebuild and _all_cached(cache_dir, entries, config_hash):
        logger.info(
            "Feature cache up-to-date (%d samples): %s", len(entries), cache_dir
        )
        return

    logger.info(
        "Extracting backbone features for %d samples → %s  [device=%s]",
        len(entries), cache_dir, device,
    )
    backbone = backbone.to(device)
    backbone.eval()

    transform = build_image_transform(config["image_size"])

    for idx, entry in enumerate(entries):
        sid  = _sample_id_from_entry(entry)
        path = _cache_path(cache_dir, sid)

        # Skip valid caches unless forced
        if not force_rebuild and _is_cache_valid(path, config_hash):
            continue

        image_abs  = dataset_root / entry["image"]
        target_abs = dataset_root / entry["target"]

        try:
            with Image.open(image_abs) as img:
                image = img.convert("RGB")
        except Exception as exc:
            logger.warning("Skipping %s — cannot open image: %s", image_abs, exc)
            continue

        # Apply preprocessing transform; add batch dimension
        img_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

        # Extract multi-scale backbone features
        hidden_states = backbone(img_tensor)  # tuple of 4 tensors [1, N_i, C_i]

        # Persist to disk in float16 (saves ~50 % disk vs float32)
        payload = {
            "hidden_states": [
                hs.squeeze(0).half().cpu()  # [N_i, C_i] in float16
                for hs in hidden_states
            ],
            "sample_id":                sid,
            "image_path":               str(image_abs),
            "target_mask_path":         str(target_abs),
            "feature_shape":            [list(hs.squeeze(0).shape) for hs in hidden_states],
            "backbone_name":            backbone.variant,
            "preprocessing_config_hash": config_hash,
        }
        torch.save(payload, path)

        if (idx + 1) % 200 == 0 or idx == len(entries) - 1:
            logger.info("  [%d / %d] cached", idx + 1, len(entries))

    logger.info("Feature extraction complete → %s", cache_dir)


# ---------------------------------------------------------------------------
# Cached-features Dataset
# ---------------------------------------------------------------------------


class CachedFloorplanDataset(Dataset):
    """Dataset that loads pre-computed backbone features + semantic masks.

    Used during Phase-1 head-only training.  The backbone is NEVER called
    during ``__getitem__``; features are simply deserialized from disk.

    Each sample returns a dict:
      ``hidden_states``  — list of 4 float32 tensors ``[N_i, C_i]``
      ``mask``           — ``[H, W]`` long tensor of class IDs
      ``sample_id``      — str
      ``image_path``     — str  (original image, used for preview display)
      ``mask_path``      — str

    Args:
        index_path:   Path to the split JSON index.
        dataset_root: Dataset root directory (masks are loaded relative to it).
        cache_dir:    Directory containing ``.pt`` feature cache files.
        image_size:   Spatial size to which masks are resized (square).
    """

    def __init__(
        self,
        index_path: str | Path,
        dataset_root: str | Path,
        cache_dir: str | Path,
        image_size: int = 512,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.cache_dir    = Path(cache_dir)
        self.image_size   = image_size

        with open(index_path) as f:
            self.entries: list[dict[str, Any]] = json.load(f)

        self._sample_ids = [_sample_id_from_entry(e) for e in self.entries]

        # Warn about missing cache files so the user knows to run extraction
        missing = sum(
            1
            for sid in self._sample_ids
            if not _cache_path(self.cache_dir, sid).exists()
        )
        if missing:
            logger.warning(
                "%d / %d cache files missing — run extract_features_for_split() first.",
                missing, len(self.entries),
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Retry up to len(dataset) times to skip corrupted cache files
        for attempt in range(len(self.entries)):
            real_idx = (idx + attempt) % len(self.entries)
            entry    = self.entries[real_idx]
            sid      = self._sample_ids[real_idx]
            path     = _cache_path(self.cache_dir, sid)

            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                # float16 on disk → float32 for computation
                hidden_states = [hs.float() for hs in payload["hidden_states"]]
            except Exception as exc:
                logger.warning("Failed to load cache %s: %s — retrying next sample.", path, exc)
                continue

            # Load and resize the semantic mask (nearest-neighbour to preserve class IDs)
            mask_path = self.dataset_root / entry["target"]
            try:
                with Image.open(mask_path) as m:
                    mask = m.convert("L")
                mask_resized = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            except Exception as exc:
                logger.warning("Failed to load mask %s: %s — retrying.", mask_path, exc)
                continue

            mask_tensor = torch.as_tensor(np.array(mask_resized), dtype=torch.long)

            return {
                "hidden_states": hidden_states,        # list[Tensor]  [N_i, C_i]
                "mask":          mask_tensor,           # [H, W] long
                "sample_id":     sid,
                "image_path":    entry.get("image", ""),
                "mask_path":     str(mask_path),
                "input_type":    entry.get("input_type", "unknown"),
            }

        raise RuntimeError(f"No valid cached sample found starting from idx={idx}")
