"""Guard the README against drifting away from the published evaluation results.

Task 44 requires that README numbers are generated or validated from the
canonical result files rather than transcribed by hand. This test runs the
generator in ``--check`` mode, so any edit to either side that desynchronizes
them fails the ordinary CI gate.

The test skips when no published result set is present, so a fresh clone without
``evaluation/results/`` still has a green suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "evaluation" / "results" / "v1"
README = PROJECT_ROOT / "README.md"
SYNC_SCRIPT = PROJECT_ROOT / "scripts" / "sync_readme_metrics.py"

MARKERS = ("<!-- BEGIN GENERATED METRICS -->", "<!-- END GENERATED METRICS -->")


def test_readme_has_generated_metric_markers():
    text = README.read_text(encoding="utf-8")
    for marker in MARKERS:
        assert marker in text, f"README is missing {marker}"


@pytest.mark.skipif(
    not (RESULTS / "summary.json").exists(),
    reason="no published evaluation results in this checkout",
)
def test_readme_metrics_match_published_results():
    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        "README metric tables are out of sync with evaluation/results/v1.\n"
        "Run: python scripts/sync_readme_metrics.py\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.skipif(
    not (RESULTS / "summary.json").exists(),
    reason="no published evaluation results in this checkout",
)
def test_readme_images_referenced_by_readme_exist():
    """Every local image the README points at must actually be present."""
    import re

    text = README.read_text(encoding="utf-8")
    missing = []
    for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        if path.startswith(("http://", "https://")):
            continue
        if not (PROJECT_ROOT / path).exists():
            missing.append(path)
    assert not missing, f"README references missing images: {missing}"
