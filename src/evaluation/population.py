"""Evaluation population selection and exclusion accounting (spec_v006 §1).

The shipped splits leak at the plan level: ``src/build_splits.py`` shuffles
manifest *entries*, and each CubiCasa plan contributes two renderings, so one
plan's ``F1_scaled.png`` can sit in train while its ``model_clean.png`` sits in
test. This module derives the leakage-free population and records every
exclusion with a reason, so no sample is ever silently dropped.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_BASE_PLAN_RE = re.compile(r"^(\d+)")


def base_plan_id(sample_id: str) -> str:
    """Leading CubiCasa folder id shared by both renderings of one plan."""
    match = _BASE_PLAN_RE.match(sample_id)
    return match.group(1) if match else sample_id


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Split manifest must be a JSON list: {path}")
    return data


@dataclass
class Population:
    """The samples an evaluation run will score, plus everything it excluded."""

    entries: list[dict] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    total_test_entries: int = 0
    leakage_stats: dict[str, int] = field(default_factory=dict)
    manifest_hashes: dict[str, str] = field(default_factory=dict)

    def exclude(self, sample_id: str, reason: str, detail: str = "") -> None:
        self.excluded.append({"sample_id": sample_id, "reason": reason, "detail": detail})

    @property
    def sample_ids(self) -> list[str]:
        return [e["sample_id"] for e in self.entries]

    def population_hash(self) -> str:
        """Stable hash of the scored sample list, for provenance."""
        joined = "\n".join(sorted(self.sample_ids))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_population(
    splits_dir: Path,
    input_type: str | None = None,
    enforce_leakage_filter: bool = True,
    limit: int | None = None,
) -> Population:
    """Build the canonical evaluation population from the split manifests.

    Args:
        splits_dir: Directory holding train.json / val.json / test.json.
        input_type: Restrict to one rendering, e.g. ``svg_rendered_clean``.
            Phase 4 consumes ``model_clean.png``, so L2/L3 pass this.
        enforce_leakage_filter: Apply the spec_v006 §1.4 ``test_clean`` rule.
            Setting this False produces the leakage-affected population, which
            may only be reported when explicitly labelled as such.
        limit: Optional cap for smoke runs, applied after sorting by sample_id
            so the subset is deterministic.
    """
    splits_dir = Path(splits_dir)
    train_path = splits_dir / "train.json"
    val_path = splits_dir / "val.json"
    test_path = splits_dir / "test.json"

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Split manifest not found: {path}")

    train = load_manifest(train_path)
    val = load_manifest(val_path)
    test = load_manifest(test_path)

    seen_bases = {base_plan_id(e["sample_id"]) for e in train}
    seen_bases |= {base_plan_id(e["sample_id"]) for e in val}

    population = Population(
        total_test_entries=len(test),
        manifest_hashes={
            "train.json": manifest_sha256(train_path),
            "val.json": manifest_sha256(val_path),
            "test.json": manifest_sha256(test_path),
        },
    )

    train_bases = {base_plan_id(e["sample_id"]) for e in train}
    val_bases = {base_plan_id(e["sample_id"]) for e in val}
    test_bases = {base_plan_id(e["sample_id"]) for e in test}
    population.leakage_stats = {
        "train_test_base_overlap": len(train_bases & test_bases),
        "val_test_base_overlap": len(val_bases & test_bases),
        "leaked_test_entries": sum(1 for e in test if base_plan_id(e["sample_id"]) in seen_bases),
    }

    candidates = sorted(test, key=lambda e: e["sample_id"])
    for entry in candidates:
        sample_id = entry["sample_id"]
        if input_type is not None and entry.get("input_type") != input_type:
            continue
        if enforce_leakage_filter and base_plan_id(sample_id) in seen_bases:
            population.exclude(
                sample_id,
                "leaked_base_plan",
                f"base plan {base_plan_id(sample_id)} appears in train or validation",
            )
            continue
        population.entries.append(entry)

    if limit is not None and limit >= 0:
        population.entries = population.entries[:limit]

    return population
