"""Run the 7-class SegFormer model on a preprocessed 512x512 image.

Loads the FloorplanSegModel (backbone + decoder) from a training checkpoint
and returns a 512x512 uint8 class-ID mask.  Model is loaded once and cached
in-process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Deferred imports so this module can be imported in CPU-only environments.
_SEG_MODEL_CACHE: dict[str, Any] = {}

# Imagenet normalization (same as training)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

NUM_CLASSES = 7


def _load_seg_model(checkpoint_path: str | Path, device_str: str = "cuda") -> Any:
    """Load (or return cached) FloorplanSegModel from training checkpoint."""
    key = str(checkpoint_path)
    if key in _SEG_MODEL_CACHE:
        return _SEG_MODEL_CACHE[key]

    import torch
    from src.models import (  # type: ignore[import]
        BACKBONE_HIDDEN_SIZES,
        FloorplanDecoder,
        FloorplanSegModel,
        SegFormerBackboneExtractor,
    )

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Checkpoint contains decoder-only weights (backbone was frozen and not saved).
    # Load decoder from checkpoint; reload backbone fresh from HuggingFace pretrained.
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    decoder_state = ckpt.get("model_state_dict", ckpt)

    hidden_sizes = BACKBONE_HIDDEN_SIZES["segformer_b0"]
    decoder = FloorplanDecoder(hidden_sizes, num_classes=NUM_CLASSES, output_size=512)
    decoder.load_state_dict(decoder_state)

    backbone = SegFormerBackboneExtractor(variant="segformer_b0", pretrained=True)
    model = FloorplanSegModel(backbone, decoder).to(device)
    model.eval()

    _SEG_MODEL_CACHE[key] = (model, device)
    return model, device


def run_segmentation(
    canvas_pil: Any,  # PIL.Image 512x512 RGB
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> np.ndarray:
    """Run 7-class segmentation on the 512x512 preprocessed image.

    Returns a 512x512 uint8 class-ID mask (values 0-6).
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image

    model, dev = _load_seg_model(checkpoint_path, device)

    if canvas_pil.size != (512, 512):
        canvas_pil = canvas_pil.resize((512, 512), Image.LANCZOS)

    arr = np.array(canvas_pil.convert("RGB"), dtype=np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - _MEAN[c]) / _STD[c]
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = model(tensor)  # [1, 7, 512, 512]
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def seg_mask_to_color_preview(class_map: np.ndarray) -> np.ndarray:
    """Convert a class-ID mask to an RGB preview image."""
    from ..decode_prediction import CLASS_PALETTE
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_PALETTE.items():
        rgb[class_map == cls_id] = color
    return rgb
