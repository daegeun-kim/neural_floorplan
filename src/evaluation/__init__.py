"""Reproducible evaluation harness (spec_v006).

Implements the three separately reported layers of the evaluation contract:

    L1  seven-class SegFormer semantic quality      (required secondary)
    L2  Raster-to-Graph wall topology quality       (primary)
    L3  end-to-end Phase 4 circulation and openings (primary)

Nothing in this package modifies the model or the vectorizer. It measures the
current system; it never tunes against the test set.
"""

from src.evaluation.transforms import CanvasTransform
from src.evaluation.types import (
    EXCLUSION_REASONS,
    Graph,
    Opening,
    SampleRecord,
)

__all__ = [
    "EXCLUSION_REASONS",
    "CanvasTransform",
    "Graph",
    "Opening",
    "SampleRecord",
]
