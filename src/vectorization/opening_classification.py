"""Heuristic door/window classification of OpeningPrimitive candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .primitives import DoorPrimitive, OpeningPrimitive, WallPrimitive, WindowPrimitive


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


def _pick_swing_direction(
    opening: OpeningPrimitive,
    room_mask: np.ndarray,
    probe_dist: int = 20,
    n_probes: int = 6,
) -> str:
    """Return "left" or "right" swing direction based on which side has more room pixels.

    "left"  = CCW perpendicular from the wall direction (angle + 90°)
    "right" = CW perpendicular from the wall direction  (angle − 90°)
    Falls back to "left" when room_mask is unavailable or evidence is tied.
    """
    if room_mask is None:
        return "left"

    cx, cy = opening.center
    angle_rad = math.radians(opening.orientation_angle)
    h, w = room_mask.shape

    def _count_room(perp_rad: float) -> int:
        cos_p, sin_p = math.cos(perp_rad), math.sin(perp_rad)
        count = 0
        for step in range(1, n_probes + 1):
            dist = probe_dist * step / n_probes
            px = int(round(cx + cos_p * dist))
            py = int(round(cy + sin_p * dist))
            if 0 <= px < w and 0 <= py < h and room_mask[py, px] > 0:
                count += 1
        return count

    left_count  = _count_room(angle_rad + math.pi / 2.0)
    right_count = _count_room(angle_rad - math.pi / 2.0)

    return "left" if left_count >= right_count else "right"


def classify_openings(
    openings: list[OpeningPrimitive],
    opening_mask: np.ndarray,
    config: ClassificationConfig | None = None,
    room_mask: Optional[np.ndarray] = None,
    walls: Optional[list[WallPrimitive]] = None,
) -> tuple[list[DoorPrimitive], list[WindowPrimitive], list[OpeningPrimitive]]:
    if config is None:
        config = ClassificationConfig()
    wall_map = {w.primitive_id: w for w in walls} if walls else {}

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
            host_wall = wall_map.get(op.host_wall_id) if op.host_wall_id else None
            thickness = host_wall.thickness if host_wall else 8.0
            windows.append(
                WindowPrimitive(
                    primitive_id=op.primitive_id.replace("opening", "window"),
                    center=op.center,
                    width=op.width,
                    orientation_angle=op.orientation_angle,
                    thickness=thickness,
                    host_wall_id=op.host_wall_id,
                    confidence=conf,
                    scale_info=op.scale_info,
                )
            )
        elif label == "door_candidate" and conf >= config.min_confidence_for_type:
            hx, hy = op.start
            swing = _pick_swing_direction(op, room_mask)
            doors.append(
                DoorPrimitive(
                    primitive_id=op.primitive_id.replace("opening", "door"),
                    hinge_point=(hx, hy),
                    width=op.width,
                    orientation_angle=op.orientation_angle,
                    swing_direction=swing,
                    host_wall_id=op.host_wall_id,
                    confidence=conf,
                    scale_info=op.scale_info,
                )
            )
        else:
            op.opening_type = label  # type: ignore[assignment]
            unresolved.append(op)

    return doors, windows, unresolved
