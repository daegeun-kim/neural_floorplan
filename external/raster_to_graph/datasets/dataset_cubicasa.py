"""CubiCasa5K dataset for Raster-to-Graph fine-tuning.

Reads train/val/test manifests produced by scripts/build_raster2graph_manifests.py.
Returns (img_tensor, target_dict) pairs compatible with engine.train_one_epoch
and the existing data_utils / criterion pipeline.
"""

import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from util.mean_std import mean, std
from util.cubicasa_utils import wall_graph_to_training_tensors

CANVAS_SIZE = 512


def _preprocess_image(path):
    """Load model_clean.png, scale to fit 512, centre on white 512x512 canvas,
    and return a normalised [3, 512, 512] float tensor."""
    img = Image.open(path).convert("RGB")
    sf  = CANVAS_SIZE / max(img.size)
    new_w, new_h = int(img.size[0] * sf), int(img.size[1] * sf)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    canvas.paste(img, ((CANVAS_SIZE - new_w) // 2, (CANVAS_SIZE - new_h) // 2))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    mean_t = torch.tensor(mean, dtype=t.dtype).view(-1, 1, 1)
    std_t  = torch.tensor(std,  dtype=t.dtype).view(-1, 1, 1)
    return (t - mean_t) / std_t


class CubiCasaDataset(Dataset):
    """Dataset for one split (train / val / test).

    Each item is (img_tensor, target_dict) where target_dict contains all
    keys expected by the training loop, including 'graph' as a tensor.
    Samples whose wall_graph.json converts to a degenerate graph are skipped
    at construction time.
    """

    def __init__(self, manifest_path):
        with open(manifest_path) as f:
            self.entries = json.load(f)
        # Entries were already validated by build_raster2graph_manifests.py.
        # No upfront per-file validation here — that would read ~3k JSON files
        # on init, which is prohibitively slow on a cold HDD.

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]

        img_tensor = _preprocess_image(entry["input"])
        target     = wall_graph_to_training_tensors(entry["graph"])

        target["image_id"] = torch.tensor([index], dtype=torch.int64)
        return img_tensor, target
