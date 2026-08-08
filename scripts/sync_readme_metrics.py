"""Generate the README metric tables from the canonical evaluation result files.

Every number the README publishes is written by this script from
``evaluation/results/<version>/``. Nothing is transcribed by hand, so the README
cannot drift away from the measured results.

    python scripts/sync_readme_metrics.py                 # rewrite README tables
    python scripts/sync_readme_metrics.py --check         # fail if out of date

The ``--check`` mode is exercised by ``tests/test_readme_metrics.py``, so drift
is caught by the ordinary CI gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BEGIN = "<!-- BEGIN GENERATED METRICS -->"
END = "<!-- END GENERATED METRICS -->"

CLASS_ORDER = [
    "background",
    "floor",
    "wall",
    "window",
    "door_arc",
    "door_leaf",
    "door_origin",
]

PRIMARY_ROWS = [
    (
        "macro_junction_degree_exact_f1",
        "Wall junction F1 (degree-exact)",
        "junction recovered with the correct number of incident walls",
    ),
    (
        "macro_junction_degree_within_1_f1",
        "Wall junction F1 (degree ±1)",
        "same, allowing one incident-wall difference",
    ),
    ("macro_node_f1", "Wall node F1 (position only)", "junction position found, degree ignored"),
    (
        "macro_edge_f1",
        "Wall edge F1 (topological)",
        "both endpoints matched to the same reference wall",
    ),
    (
        "macro_edge_overlap_f1",
        "Wall edge F1 (segment overlap)",
        "collinear and ≥50% overlapping, independent of node split",
    ),
    ("macro_door_f1", "Door detection F1", "typed door matched within 12 px"),
    ("macro_window_f1", "Window detection F1", "typed window matched within 12 px"),
    (
        "macro_host_accuracy",
        "Opening host accuracy",
        "matched opening attached to the correct wall",
    ),
    (
        "macro_gap_accuracy",
        "Opening gap correctness",
        "hosted opening interrupts its wall exactly once, correct width",
    ),
    ("macro_adjacency_f1", "Space adjacency F1", "rooms connected through the right openings"),
    (
        "macro_reachability_agreement",
        "Reachable-pair agreement",
        "circulation connectivity matches the reference",
    ),
    (
        "macro_component_count_abs_error",
        "Wall component count error",
        "mean absolute difference in connected components (lower is better)",
    ),
]

BASELINE_ROWS = [
    ("macro_node_f1", "Wall node F1 (position only)"),
    ("macro_junction_degree_exact_f1", "Wall junction F1 (degree-exact)"),
    ("macro_junction_degree_within_1_f1", "Wall junction F1 (degree ±1)"),
    ("macro_edge_f1", "Wall edge F1 (topological)"),
    ("macro_edge_overlap_f1", "Wall edge F1 (segment overlap)"),
    ("macro_component_count_abs_error", "Wall component count error (lower is better)"),
]

FAILURE_LABELS = {
    "missing_doors": "Doors in the reference that were not produced",
    "spurious_doors": "Doors produced with no reference counterpart",
    "missing_windows": "Windows in the reference that were not produced",
    "spurious_windows": "Windows produced with no reference counterpart",
    "unhosted_openings": "Matched openings not attached to any wall",
    "broken_junctions": "Junctions found at the right place with the wrong degree",
    "disconnected_circulation": "Space pairs reachable in the reference but not in the output",
    "duplicate_gaps": "Hosted openings producing more than one wall gap",
    "mistyped_openings": "Openings matched by position but with the wrong type",
}


def fmt(value: object, places: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "n/a"
    return f"{number:.{places}f}"


def load(results: Path) -> dict:
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((results / "provenance.json").read_text(encoding="utf-8"))
    failures = json.loads((results / "failures.json").read_text(encoding="utf-8"))
    selected = json.loads((results / "selected_cases.json").read_text(encoding="utf-8"))
    with open(results / "baseline_comparison.csv", encoding="utf-8") as handle:
        baseline = {row["metric"]: row for row in csv.DictReader(handle)}
    return {
        "summary": summary,
        "provenance": provenance,
        "failures": failures,
        "selected": selected,
        "baseline": baseline,
    }


def build_block(data: dict, results: Path) -> str:
    summary = data["summary"]
    provenance = data["provenance"]
    split = provenance["split"]
    primary = summary["primary_spatial_logic"]
    seg = summary.get("secondary_segmentation")
    lines: list[str] = []

    lines.append("### Evaluated population and provenance")
    lines.append("")
    leak = split["leakage_stats"]
    lines.append(
        f"| | |\n|---|---|\n"
        f"| Dataset | CubiCasa5K `high_quality_architectural` (not redistributed) |\n"
        f"| Test entries in `splits/test.json` | {split['total_test_entries']} |\n"
        f"| Excluded as leaked (plan seen in train/val) | {leak['leaked_test_entries']} |\n"
        f"| Wall-graph / circulation population | {split['scored_sample_count']} plans, "
        f"{summary['denominators']['scored']} scored, "
        f"{summary['denominators']['excluded']} unscorable |\n"
        f"| Segmentation population | {seg.get('denominator', 'n/a') if isinstance(seg, dict) else 'n/a'} entries |\n"
        f"| Matched openings scored | {summary['denominators']['matched_openings']} |\n"
        f"| Seed / device | {provenance['seed']} / "
        f"{provenance['device'].get('gpu_name', provenance['device'].get('resolved', 'cpu'))} |\n"
        f"| SegFormer checkpoint | `sha256:{provenance['checkpoints']['segformer']['sha256'][:16]}…` |\n"
        f"| Raster-to-Graph checkpoint | `sha256:{provenance['checkpoints']['raster_to_graph']['sha256'][:16]}…` |"
    )
    lines.append("")
    lines.append(
        f"Split manifest hashes: `train {split['manifest_sha256']['train.json'][:12]}…`, "
        f"`val {split['manifest_sha256']['val.json'][:12]}…`, "
        f"`test {split['manifest_sha256']['test.json'][:12]}…`."
    )
    lines.append("")

    lines.append("### Primary: spatial logic")
    lines.append("")
    lines.append("| Metric | Score | What it measures |")
    lines.append("|---|---|---|")
    for key, label, meaning in PRIMARY_ROWS:
        if key in primary:
            lines.append(f"| {label} | **{fmt(primary[key])}** | {meaning} |")
    lines.append("")
    lines.append(
        "Macro aggregation (per-sample, then averaged). Micro values are in "
        f"[`topology.csv`]({results.as_posix()}/topology.csv)."
    )
    lines.append("")

    lines.append("### Primary: raw Raster-to-Graph baseline vs Phase 4")
    lines.append("")
    lines.append("| Metric | Raw R2G | Phase 4 | Δ |")
    lines.append("|---|---|---|---|")
    for key, label in BASELINE_ROWS:
        row = data["baseline"].get(key)
        if not row:
            continue
        base = row["baseline_raw_r2g"]
        cur = row["current_phase4"]
        delta = row["delta"]
        arrow = ""
        if delta not in ("", None):
            arrow = f"{float(delta):+.3f}"
        lines.append(f"| {label} | {fmt(base)} | **{fmt(cur)}** | {arrow} |")
    lines.append("")
    lines.append(
        "Both paths score the same plans, the same preprocessing, and the same "
        "reference data, using one pipeline run per sample. The raw baseline "
        "produces no openings and no circulation, so those metrics are reported "
        "`unavailable` for it rather than counted as zero."
    )
    lines.append("")

    if isinstance(seg, dict) and "macro" in seg:
        macro = seg["macro"]
        micro = seg["micro"]
        boundary = seg.get("boundary_f1", {})
        lines.append("### Secondary: seven-class segmentation")
        lines.append("")
        lines.append("| Class | IoU (macro) | IoU (micro) | Boundary F1 |")
        lines.append("|---|---|---|---|")
        for name in CLASS_ORDER:
            lines.append(
                f"| `{name}` | {fmt(macro['per_class_iou'].get(name))} | "
                f"{fmt(micro['per_class_iou'].get(name))} | "
                f"{fmt(boundary.get(name))} |"
            )
        lines.append(
            f"| **mean IoU** (background included) | **{fmt(macro['miou_background_included'])}** | "
            f"{fmt(micro['miou_background_included'])} | |"
        )
        lines.append(
            f"| **foreground mean IoU** | **{fmt(macro['foreground_miou'])}** | "
            f"{fmt(micro['foreground_miou'])} | |"
        )
        lines.append(
            f"| **pixel accuracy** | {fmt(macro['pixel_accuracy'])} | "
            f"{fmt(micro['pixel_accuracy'])} | |"
        )
        lines.append("")

    lines.append("### Failure categories")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---|")
    for key, count in sorted(data["failures"]["totals"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {FAILURE_LABELS.get(key, key)} | {count} |")
    lines.append("")
    excluded = data["failures"].get("excluded_by_reason", {})
    if excluded:
        parts = ", ".join(f"`{k}` × {v['count']}" for k, v in sorted(excluded.items()))
        lines.append(f"Unscorable samples: {parts}.")
        lines.append("")

    cases = data["selected"]["cases"]
    lines.append("Representative cases are chosen by a fixed rule, not by eye:")
    lines.append("")
    lines.append("| Case | Sample | End-to-end score |")
    lines.append("|---|---|---|")
    for label in ("success", "mixed", "failure"):
        if label in cases:
            lines.append(
                f"| {label} | `{cases[label]['sample_id']}` | {fmt(cases[label]['score'])} |"
            )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "evaluation/results/v1")
    parser.add_argument("--readme", type=Path, default=PROJECT_ROOT / "README.md")
    parser.add_argument("--check", action="store_true", help="Verify without writing.")
    args = parser.parse_args()

    if not (args.results / "summary.json").exists():
        print(f"error: no results at {args.results}", file=sys.stderr)
        return 2

    rel = args.results.relative_to(PROJECT_ROOT) if args.results.is_absolute() else args.results
    block = build_block(load(args.results), rel)

    text = args.readme.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(
            f"error: {args.readme.name} is missing the {BEGIN} / {END} markers",
            file=sys.stderr,
        )
        return 2

    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    updated = f"{head}{BEGIN}\n\n{block}\n{END}{tail}"

    if args.check:
        if updated != text:
            print(
                "README metric tables are out of date with "
                f"{rel.as_posix()}. Run: python scripts/sync_readme_metrics.py",
                file=sys.stderr,
            )
            return 1
        print("README metrics are in sync with the canonical result files.")
        return 0

    args.readme.write_text(updated, encoding="utf-8")
    print(f"README metrics synced from {rel.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
