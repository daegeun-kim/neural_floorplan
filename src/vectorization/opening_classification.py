"""Heuristic door/window classification of OpeningPrimitive candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .primitives import DoorPrimitive, OpeningPrimitive, WindowPrimitive


@dataclass
class ClassificationConfig:
    window_min_aspect_ratio: float = 2.5
    door_max_aspect_ratio: float = 2.0
    min_confidence_for_type: float = 0.65


def _aspect_ratio_from_mask_component(
    opening_mask: np.ndarray, label_mask: np.ndarray
) -> float:
    pts = np.argwhere(label_mask > 0)
    if len(pts) < 3:
        return 1.0
    _, (mw, mh), _ = cv2.minAreaRect(pts[:, ::-1].astype(np.float32))
    long_side = max(mw, mh, 1.0)
    short_side = max(min(mw, mh), 1.0)
    return long_side / short_side


def classify_openings(
    openings: list[OpeningPrimitive],
    opening_mask: np.ndarray,
    config: ClassificationConfig | None = None,
) -> tuple[list[DoorPrimitive], list[WindowPrimitive], list[OpeningPrimitive]]:
    if config is None:
        config = ClassificationConfig()

    n_labels, labels = cv2.connectedComponents(opening_mask, connectivity=8)

    doors: list[DoorPrimitive] = []
    windows: list[WindowPrimitive] = []
    unresolved: list[OpeningPrimitive] = []

    for op in openings:
        cx, cy = op.center
        ix, iy = int(round(cx)), int(round(cy))
        h, w = opening_mask.shape
        if 0 <= iy < h and 0 <= ix < w:
            comp_label = labels[iy, ix]
        else:
            comp_label = 0

        if comp_label > 0:
            comp_mask = (labels == comp_label).astype(np.uint8) * 255
            aspect = _aspect_ratio_from_mask_component(opening_mask, comp_mask)
        else:
            long_side = max(op.width, 1.0)
            short_side = max(op.width / 3.0, 1.0)
            aspect = long_side / short_side

        if aspect >= config.window_min_aspect_ratio:
            label = "window_candidate"
            conf = min(1.0, (aspect - config.window_min_aspect_ratio) / 2.0 + 0.65)
        elif aspect <= config.door_max_aspect_ratio:
            label = "door_candidate"
            conf = min(1.0, (config.door_max_aspect_ratio - aspect) / 1.5 + 0.65)
        else:
            label = "generic"
            conf = 0.4

        if label == "window_candidate" and conf >= config.min_confidence_for_type:
            windows.append(
                WindowPrimitive(
                    primitive_id=op.primitive_id.replace("opening", "window"),
                    center=op.center,
                    width=op.width,
                    orientation_angle=op.orientation_angle,
                    host_wall_id=op.host_wall_id,
                    confidence=conf,
                    scale_info=op.scale_info,
                )
            )
        elif label == "door_candidate" and conf >= config.min_confidence_for_type:
            hx, hy = op.start
            doors.append(
                DoorPrimitive(
                    primitive_id=op.primitive_id.replace("opening", "door"),
                    hinge_point=(hx, hy),
                    width=op.width,
                    orientation_angle=op.orientation_angle,
                    swing_direction="left",
                    host_wall_id=op.host_wall_id,
                    confidence=conf,
                    scale_info=op.scale_info,
                )
            )
        else:
            op.opening_type = label  # type: ignore[assignment]
            unresolved.append(op)

    return doors, windows, unresolved
