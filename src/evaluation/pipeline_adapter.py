"""Bridge between the Phase 4 pipeline and the evaluation harness.

Kept separate from :mod:`src.evaluation.runner` so the harness, its tests, and
the smoke path never import torch, the vendored Raster-to-Graph code, or the
segmentation model. Only the full run touches this module.

Both evaluated paths come from the *same* pipeline invocation shape, so the
baseline and the current system share preprocessing, reference data, and metric
code exactly as spec_v006 §5.1 requires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation import adapters
from src.evaluation.population import base_plan_id
from src.evaluation.references import ReferenceUnavailable
from src.evaluation.runner import PredictionBundle
from src.evaluation.transforms import CanvasTransform

IMAGE_NAME = "model_clean.png"


class Phase4Runner:
    """Runs the Phase 4 pipeline once per sample and serves both evaluated paths.

    Raster-to-Graph inference is Monte-Carlo (``monte_times`` attempts reranked
    by validity score), so invoking the pipeline separately for the baseline and
    for the current system would score them against *different* graphs. That
    would break the spec_v006 §5.1 requirement that both paths share the same
    preprocessing and the same samples, and would make the comparison
    irreproducible. Caching one result per sample removes both problems and
    halves the runtime.
    """

    def __init__(
        self,
        dataset_root: Path,
        seg_checkpoint: Path,
        r2g_checkpoint: Path,
        output_dir: Path,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.seg_checkpoint = Path(seg_checkpoint)
        self.r2g_checkpoint = Path(r2g_checkpoint)
        self.output_dir = Path(output_dir)
        self._cache: dict[str, Any] = {}

    def _bundles(self, entry: dict) -> tuple[PredictionBundle, PredictionBundle]:
        """Run the pipeline once and extract (current, baseline).

        Only the two small bundles are cached; the full ``Phase4Result`` (canvas
        image, segmentation map, component masks) is released immediately so
        memory stays flat across a full-population run.
        """
        sample_id = entry["sample_id"]
        if sample_id in self._cache:
            return self._cache[sample_id]

        from src.vectorization.phase4.pipeline import run_phase4_pipeline

        base = base_plan_id(sample_id)
        image_path = self.dataset_root / base / IMAGE_NAME
        if not image_path.exists():
            raise ReferenceUnavailable("missing_input", str(image_path.name))

        result = run_phase4_pipeline(
            image_path=image_path,
            output_dir=self.output_dir / sample_id,
            seg_checkpoint=self.seg_checkpoint,
            r2g_checkpoint=self.r2g_checkpoint,
        )

        transform = CanvasTransform.from_manifest(result.preprocessing_manifest)

        current_graph = adapters.graph_from_payload(result.aligned_graph or {})
        opening_gaps = result.trimmed_graph.opening_gaps if result.trimmed_graph else []
        current = PredictionBundle(
            graph=current_graph,
            openings=adapters.openings_from_phase4(
                result.hosted_doors,
                result.hosted_windows,
                current_graph,
                opening_gaps=opening_gaps,
            ),
            transform=transform,
        )
        baseline = PredictionBundle(
            graph=adapters.graph_from_payload(result.raw_graph or {}),
            openings=[],
            transform=transform,
        )

        self._cache[sample_id] = (current, baseline)
        return current, baseline

    def current(self, entry: dict) -> PredictionBundle:
        """The full Phase 4 pipeline: aligned graph plus hosted openings."""
        return self._bundles(entry)[0]

    def baseline(self, entry: dict) -> PredictionBundle:
        """Raw Raster-to-Graph output, before Phase 4 alignment/merge/recovery."""
        return self._bundles(entry)[1]

    def release(self) -> None:
        """Drop cached pipeline results once both paths have been scored."""
        self._cache.clear()


def make_predictor(
    dataset_root: Path,
    seg_checkpoint: Path,
    r2g_checkpoint: Path,
    output_dir: Path,
    raw_baseline: bool = False,
) -> Any:
    """Return ``predict(entry) -> PredictionBundle`` for a single path."""
    runner = Phase4Runner(dataset_root, seg_checkpoint, r2g_checkpoint, output_dir)
    return runner.baseline if raw_baseline else runner.current
