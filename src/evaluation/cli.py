"""Evaluation CLI (spec_v006 §7, Task 42 §1).

Two modes:

    smoke : fixture-only. No dataset, no checkpoints, no GPU, no network.
            Validates every output schema. Runs in ordinary CI.
    full  : explicit local dataset root and checkpoint paths.

The CLI never downloads data or checkpoints, and fails early with an actionable
message rather than part-way through a long run.

    python -m src.evaluation.cli smoke --output-dir outputs/eval_smoke
    python -m src.evaluation.cli full \
        --dataset-root docs/high_quality_architectural \
        --splits-dir splits \
        --seg-checkpoint checkpoints_CNN/segformer_b0_run3/best.pt \
        --r2g-checkpoint checkpoints_Raster2Graph/checkpoint0299.pth \
        --output-dir evaluation/results/v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.evaluation import fixtures
from src.evaluation.population import build_population
from src.evaluation.provenance import build_provenance
from src.evaluation.runner import (
    evaluate_population,
    set_deterministic_seed,
    write_results,
)

DEFAULT_SEED = 42


class UsageError(SystemExit):
    """Actionable, early failure. Carries exit code 2."""

    def __init__(self, message: str) -> None:
        super().__init__(2)
        self.message = message


def _require_dir(path: Path, what: str) -> Path:
    if not path.exists():
        raise UsageError(
            f"{what} not found: {path}\n"
            f"  This evaluation never downloads data. Obtain it separately and "
            f"pass the correct --{what.lower().replace(' ', '-')}."
        )
    if not path.is_dir():
        raise UsageError(f"{what} is not a directory: {path}")
    return path


def _require_file(path: Path, what: str) -> Path:
    if not path.exists():
        raise UsageError(
            f"{what} not found: {path}\n"
            f"  Checkpoints are never downloaded implicitly and are not stored "
            f"in this repository. See the README 'Checkpoints' section."
        )
    return path


def _require_writable(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise UsageError(f"Output directory is not writable: {path}\n  {exc}") from exc
    return path


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if not torch.cuda.is_available():
            raise UsageError(
                f"Device {requested!r} was requested but CUDA is not available.\n"
                f"  Pass --device cpu, or run on a machine with a working NVIDIA GPU."
            )
    except ImportError as exc:
        raise UsageError("PyTorch is not installed; cannot use a CUDA device.") from exc
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.cli",
        description=(
            "Topology-first evaluation harness (spec_v006). Primary: wall "
            "topology, circulation, openings. Secondary: per-class mIoU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    smoke = sub.add_parser(
        "smoke",
        help="Fixture-only run. No dataset, checkpoints, GPU, or network.",
    )
    smoke.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the result files.",
    )
    smoke.add_argument("--seed", type=int, default=DEFAULT_SEED)

    full = sub.add_parser(
        "full",
        help="Full run against local data and checkpoints.",
    )
    full.add_argument("--dataset-root", type=Path, required=True)
    full.add_argument("--splits-dir", type=Path, default=Path("splits"))
    full.add_argument("--seg-checkpoint", type=Path, required=True)
    full.add_argument("--r2g-checkpoint", type=Path, required=True)
    full.add_argument("--train-config", type=Path, default=None)
    full.add_argument("--vectorization-config", type=Path, default=None)
    full.add_argument("--output-dir", type=Path, required=True)
    full.add_argument(
        "--work-dir",
        type=Path,
        default=Path("outputs/evaluation_work"),
        help=(
            "Scratch directory for raw per-sample pipeline output. Kept OUTSIDE "
            "--output-dir so the published result set stays compact."
        ),
    )
    full.add_argument("--device", default="cuda")
    full.add_argument("--seed", type=int, default=DEFAULT_SEED)
    full.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the sample count for a smaller run (deterministic prefix).",
    )
    full.add_argument(
        "--input-type",
        default="svg_rendered_clean",
        help="Rendering to evaluate; Phase 4 consumes model_clean.png.",
    )
    full.add_argument(
        "--include-leaked",
        action="store_true",
        help=(
            "Disable the spec_v006 §1.4 leakage filter. The result is "
            "leakage-affected and must be labelled as such wherever published."
        ),
    )
    full.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the raw Raster-to-Graph baseline path.",
    )
    full.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Skip the L1 seven-class segmentation layer.",
    )
    return parser


def run_smoke(args: argparse.Namespace) -> int:
    """Fixture-only evaluation that still exercises every output schema."""
    set_deterministic_seed(args.seed)
    output_dir = _require_writable(Path(args.output_dir))

    population = fixtures.synthetic_population()
    records = evaluate_population(
        population,
        predict=fixtures.synthetic_predictor(),
        dataset_root=Path("."),
        include_circulation=True,
        reference_loader=fixtures.synthetic_reference_loader(),
    )
    baseline_records = evaluate_population(
        population,
        predict=fixtures.synthetic_predictor(degrade=True),
        dataset_root=Path("."),
        include_circulation=True,
        reference_loader=fixtures.synthetic_reference_loader(),
    )

    provenance = build_provenance(
        population=population,
        seg_checkpoint=None,
        r2g_checkpoint=None,
        train_config=None,
        vectorization_config=None,
        device="cpu",
        seed=args.seed,
        dataset_root=None,
        layers=["L2", "L3"],
        leakage_filter_enforced=True,
        extra={"mode": "smoke", "assets_required": False},
    )

    summary = write_results(
        output_dir,
        provenance,
        records,
        segmentation_aggregate=fixtures.synthetic_segmentation_aggregate(),
        baseline_records=baseline_records,
    )
    scored = summary["denominators"]["scored"]
    print(f"[smoke] scored {scored}/{len(records)} fixture samples")
    print(f"[smoke] results written to {output_dir}")
    return 0


def run_full(args: argparse.Namespace) -> int:
    from src.evaluation import pipeline_adapter, segmentation_eval

    set_deterministic_seed(args.seed)

    dataset_root = _require_dir(Path(args.dataset_root), "Dataset root")
    splits_dir = _require_dir(Path(args.splits_dir), "Splits dir")
    seg_ckpt = _require_file(Path(args.seg_checkpoint), "SegFormer checkpoint")
    r2g_ckpt = _require_file(Path(args.r2g_checkpoint), "Raster-to-Graph checkpoint")
    output_dir = _require_writable(Path(args.output_dir))
    device = _resolve_device(args.device)

    population = build_population(
        splits_dir,
        input_type=args.input_type,
        enforce_leakage_filter=not args.include_leaked,
        limit=args.limit,
    )
    if not population.entries:
        raise UsageError(
            "The evaluation population is empty.\n  Check --input-type and the split manifests."
        )

    print(
        f"[full] population: {len(population.entries)} scored candidates "
        f"({len(population.excluded)} excluded, "
        f"leakage filter {'off' if args.include_leaked else 'on'})"
    )

    # One runner serves both paths so the baseline and the current system are
    # scored against the same Monte-Carlo draw (spec_v006 §5.1).
    runner = pipeline_adapter.Phase4Runner(
        dataset_root=dataset_root,
        seg_checkpoint=seg_ckpt,
        r2g_checkpoint=r2g_ckpt,
        output_dir=_require_writable(Path(args.work_dir)),
    )
    records = evaluate_population(
        population,
        predict=runner.current,
        dataset_root=dataset_root,
        on_progress=_progress,
    )

    baseline_records = None
    if not args.no_baseline:
        baseline_records = evaluate_population(
            population,
            predict=runner.baseline,
            dataset_root=dataset_root,
            include_circulation=False,
        )
    runner.release()

    segmentation_aggregate = None
    if not args.no_segmentation:
        # L1 applies to BOTH renderings of a plan, so it uses its own population
        # built without the input_type restriction that L2/L3 need.
        seg_population = build_population(
            splits_dir,
            input_type=None,
            enforce_leakage_filter=not args.include_leaked,
            limit=args.limit,
        )
        print(f"[full] segmentation population: {len(seg_population.entries)} entries")
        segmentation_aggregate = segmentation_eval.evaluate_segmentation(
            seg_population.entries,
            dataset_root=dataset_root,
            seg_checkpoint=seg_ckpt,
            device=device,
        )

    provenance = build_provenance(
        population=population,
        seg_checkpoint=seg_ckpt,
        r2g_checkpoint=r2g_ckpt,
        train_config=args.train_config,
        vectorization_config=args.vectorization_config,
        device=device,
        seed=args.seed,
        dataset_root=dataset_root,
        layers=["L2", "L3"],
        leakage_filter_enforced=not args.include_leaked,
        extra={"mode": "full", "input_type": args.input_type},
    )

    summary = write_results(
        output_dir,
        provenance,
        records,
        segmentation_aggregate=segmentation_aggregate,
        baseline_records=baseline_records,
    )
    print(
        f"[full] scored {summary['denominators']['scored']}/"
        f"{summary['denominators']['population_size']}"
    )
    print(f"[full] results written to {output_dir}")
    return 0


def _progress(record) -> None:
    status = "ok" if record.scored else f"excluded:{record.exclusion_reason}"
    print(f"  {record.sample_id:32s} {status}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "smoke":
            return run_smoke(args)
        return run_full(args)
    except UsageError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
