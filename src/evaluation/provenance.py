"""Run provenance record (spec_v006 §2, Task 42 §2).

Every run emits one of these. It carries split hashes, checkpoint hashes, the
class map, config identifiers, seed, software versions, device, timestamps, and
the full exclusion list. It deliberately records *relative* paths and file names
rather than absolute personal paths, and never records secrets.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evaluation.segmentation import CLASS_NAMES

_HASH_CHUNK = 1 << 20


def file_sha256(path: str | Path) -> str:
    """Streaming SHA-256; checkpoints are too large to read whole."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def software_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name, module in (
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("shapely", "shapely"),
        ("skimage", "skimage"),
        ("cv2", "cv2"),
    ):
        try:
            mod = __import__(module)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - a missing optional dep is recorded, not fatal
            versions[name] = "not installed"
    return versions


def device_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {"requested": device, "resolved": device}
    try:
        import torch

        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                info["resolved"] = "unavailable"
            else:
                index = 0
                if ":" in device:
                    index = int(device.split(":", 1)[1])
                info["gpu_name"] = torch.cuda.get_device_name(index)
                info["cuda_version"] = torch.version.cuda
    except Exception:  # noqa: BLE001 - provenance must never crash the run
        info["resolved"] = "unknown"
    return info


def build_provenance(
    *,
    population: Any,
    seg_checkpoint: Path | None,
    r2g_checkpoint: Path | None,
    train_config: Path | None,
    vectorization_config: Path | None,
    device: str,
    seed: int,
    dataset_root: Path | None,
    layers: list[str],
    leakage_filter_enforced: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance record for one evaluation run."""
    checkpoints: dict[str, Any] = {}
    for label, path in (
        ("segformer", seg_checkpoint),
        ("raster_to_graph", r2g_checkpoint),
    ):
        if path is None:
            checkpoints[label] = {"present": False}
            continue
        path = Path(path)
        if not path.exists():
            checkpoints[label] = {"present": False, "name": path.name}
            continue
        checkpoints[label] = {
            "present": True,
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    return {
        "spec": "spec_v006_evaluation",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "layers_evaluated": layers,
        "seed": seed,
        "device": device_info(device),
        "software": software_versions(),
        "class_map": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "configs": {
            "train": Path(train_config).name if train_config else None,
            "vectorization": (Path(vectorization_config).name if vectorization_config else None),
        },
        "checkpoints": checkpoints,
        # Recorded as a bare name so the record carries no personal path.
        "dataset_root_name": Path(dataset_root).name if dataset_root else None,
        "dataset": {
            "name": "CubiCasa5K",
            "subset": "high_quality_architectural",
            "redistributed": False,
        },
        "split": {
            "manifest_sha256": population.manifest_hashes,
            "total_test_entries": population.total_test_entries,
            "leakage_filter_enforced": leakage_filter_enforced,
            "leakage_stats": population.leakage_stats,
            "scored_sample_count": len(population.entries),
            "scored_population_sha256": population.population_hash(),
            "excluded_count": len(population.excluded),
        },
        "exclusions": _group_exclusions(population.excluded),
        **(extra or {}),
    }


def _group_exclusions(excluded: list[dict]) -> dict[str, Any]:
    """Group exclusions by reason.

    The leakage filter excludes hundreds of entries for one identical reason;
    repeating the detail string per sample would bloat the record without adding
    information. The sample ids are still listed in full, so the exclusion set
    remains fully auditable.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for item in excluded:
        reason = item.get("reason", "unknown")
        bucket = grouped.setdefault(
            reason, {"count": 0, "sample_ids": [], "example_detail": item.get("detail", "")}
        )
        bucket["count"] += 1
        bucket["sample_ids"].append(item.get("sample_id", ""))
    for bucket in grouped.values():
        bucket["sample_ids"].sort()
    return grouped
