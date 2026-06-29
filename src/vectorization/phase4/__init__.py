"""Phase 4 graph-to-vector pipeline (spec_v008_phase4_vectorization.md).

Public API:
    run_phase4_pipeline  — full pipeline from image path to output files
    Phase4Result         — all intermediate artifacts from one run
"""

from .pipeline import Phase4Result, run_phase4_pipeline

__all__ = ["Phase4Result", "run_phase4_pipeline"]
