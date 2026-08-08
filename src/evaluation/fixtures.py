"""Synthetic fixtures for the asset-independent smoke evaluation.

These build small, exactly-known floor plans in canvas coordinates so the smoke
run exercises every metric and every output schema without CubiCasa5K, model
checkpoints, a GPU, or network access.

Nothing here is published as a result. It exists to prove the harness runs and
that its output schemas are valid.
"""

from __future__ import annotations

from typing import Any

from src.evaluation.population import Population
from src.evaluation.runner import PredictionBundle
from src.evaluation.segmentation import CLASS_NAMES
from src.evaluation.transforms import identity
from src.evaluation.types import Graph, Opening

_FIXTURE_SAMPLES = ("9001_fixture_a", "9002_fixture_b", "9003_fixture_c")


def two_room_plan(
    offset: float = 0.0,
    drop_opening: bool = False,
    shift_wall: float = 0.0,
) -> tuple[Graph, list[Opening]]:
    """A 2-room rectangle split by a party wall with a door in it.

    Layout on the 512 canvas:

        (60,60) ------------- (260,60) ------------- (460,60)
           |                     |                     |
           |      room A         | party wall + door   |   room B
           |                     |                     |
        (60,360) ------------ (260,360) ------------ (460,360)
    """
    nodes = [
        (60.0 + offset, 60.0 + offset),
        (260.0 + offset + shift_wall, 60.0 + offset),
        (460.0 + offset, 60.0 + offset),
        (60.0 + offset, 360.0 + offset),
        (260.0 + offset + shift_wall, 360.0 + offset),
        (460.0 + offset, 360.0 + offset),
    ]
    edges = [
        (0, 1),  # top left
        (1, 2),  # top right
        (3, 4),  # bottom left
        (4, 5),  # bottom right
        (0, 3),  # left
        (2, 5),  # right
        (1, 4),  # party wall
    ]
    graph = Graph(nodes=nodes, edges=edges)

    openings: list[Opening] = []
    if not drop_opening:
        openings.append(
            Opening(
                opening_type="door",
                center=(260.0 + offset + shift_wall, 210.0 + offset),
                width=40.0,
                orientation="vertical",
                host_edge=6,
                gap_count=1,
                gap_width=40.0,
            )
        )
    openings.append(
        Opening(
            opening_type="window",
            center=(160.0 + offset, 60.0 + offset),
            width=30.0,
            orientation="horizontal",
            host_edge=0,
            gap_count=1,
            gap_width=30.0,
        )
    )
    return graph, openings


def synthetic_population() -> Population:
    """A population whose entries the synthetic predictor can serve."""
    population = Population(
        total_test_entries=len(_FIXTURE_SAMPLES) + 1,
        leakage_stats={
            "train_test_base_overlap": 0,
            "val_test_base_overlap": 0,
            "leaked_test_entries": 1,
        },
        manifest_hashes={"train.json": "fixture", "val.json": "fixture", "test.json": "fixture"},
    )
    population.entries = [
        {"sample_id": sid, "input_type": "svg_rendered_clean"} for sid in _FIXTURE_SAMPLES
    ]
    # Exercise the exclusion path so the schema is proven, not assumed.
    population.exclude("9004_fixture_leaked", "leaked_base_plan", "fixture exclusion")
    return population


def synthetic_predictor(degrade: bool = False) -> Any:
    """Return a ``predict(entry) -> PredictionBundle`` callable.

    ``degrade=True`` stands in for the raw-baseline path: the geometry is
    nudged and openings are dropped, which is exactly the shape of the real
    baseline's limitation (no openings, no circulation).
    """

    def predict(entry: dict) -> PredictionBundle:
        sample_id = entry["sample_id"]
        if degrade:
            graph, _ = two_room_plan(shift_wall=6.0)
            return PredictionBundle(graph=graph, openings=[], transform=identity())

        if sample_id.startswith("9001"):
            graph, openings = two_room_plan()  # exact match
        elif sample_id.startswith("9002"):
            graph, openings = two_room_plan(shift_wall=4.0)  # within tolerance
        else:
            graph, openings = two_room_plan(drop_opening=True)  # missing door
        return PredictionBundle(graph=graph, openings=openings, transform=identity())

    return predict


def synthetic_reference() -> tuple[Graph, list[Opening]]:
    return two_room_plan()


def synthetic_reference_loader() -> Any:
    """A ``reference_loader`` that serves the fixture plan for every sample."""

    def load(dataset_root: Any, base_plan: str, transform: Any) -> tuple[Graph, list[Opening]]:
        return two_room_plan()

    return load


def synthetic_segmentation_aggregate() -> dict[str, Any]:
    """A schema-valid segmentation block for the smoke run.

    Values are deliberately flagged as fixture data so they can never be
    mistaken for measured results.
    """
    per_class = dict.fromkeys(CLASS_NAMES, 0.0)
    return {
        "fixture": True,
        "note": "synthetic smoke values - not a measured result",
        "sample_count": 0,
        "macro": {
            "per_class_iou": per_class,
            "contributing_samples": dict.fromkeys(CLASS_NAMES, 0),
            "miou_background_included": float("nan"),
            "foreground_miou": float("nan"),
            "pixel_accuracy": float("nan"),
        },
        "micro": {
            "per_class_iou": per_class,
            "miou_background_included": float("nan"),
            "foreground_miou": float("nan"),
            "pixel_accuracy": float("nan"),
        },
        "boundary_f1": {},
        "boundary_f1_tolerance_px": 2,
    }
