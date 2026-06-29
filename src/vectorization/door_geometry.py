"""Generate door leaf and door arc geometry from accepted door-origin graph
edges (spec_v008 SS13).

The door-origin edge's two points already carry the resolved, module-snapped
hinge/far-point geometry (point_detection.py validated and snapped it before
ever creating the points), so this module's job is purely procedural: build
the perpendicular leaf and the 90-degree arc using the unchanged
``primitives/door.py`` math, picking the swing side from door_leaf/door_arc
evidence density - ported unchanged from the retired door_extraction.py.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .graph_types import GraphEdge, GraphPoint
from .primitives.door import DoorArcPrimitive, DoorLeafPrimitive, DoorOriginPrimitive, SwingSide


def _count_evidence_in_radius(mask: np.ndarray, point: tuple[float, float], radius: float) -> int:
    x, y = point
    h, w = mask.shape
    x0, x1 = max(0, int(x - radius)), min(w, int(x + radius) + 1)
    y0, y1 = max(0, int(y - radius)), min(h, int(y + radius) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0
    sub = mask[y0:y1, x0:x1]
    yy, xx = np.nonzero(sub)
    if len(xx) == 0:
        return 0
    dist2 = (xx + x0 - x) ** 2 + (yy + y0 - y) ** 2
    return int(np.sum(dist2 <= radius * radius))


def _pick_swing_side(
    hinge: tuple[float, float],
    orientation_angle: float,
    width: float,
    evidence_mask: Optional[np.ndarray],
    n_probes: int = 6,
    probe_radius: float = 10.0,
) -> SwingSide:
    """Probe both perpendicular directions from the hinge; the side with more
    door_leaf/door_arc evidence is the swing side (spec_v008 SS13.1)."""
    if evidence_mask is None or not evidence_mask.any():
        return "left"
    angle_rad = math.radians(orientation_angle)
    left_count = 0
    right_count = 0
    for k in range(1, n_probes + 1):
        dist = width * k / n_probes
        for sign, is_left in ((1.0, True), (-1.0, False)):
            perp = angle_rad + sign * math.pi / 2.0
            probe_point = (hinge[0] + dist * math.cos(perp), hinge[1] + dist * math.sin(perp))
            count = _count_evidence_in_radius(evidence_mask, probe_point, probe_radius)
            if is_left:
                left_count += count
            else:
                right_count += count
    return "left" if left_count >= right_count else "right"


def generate_door_geometry(
    points: list[GraphPoint],
    door_origin_edges: list[GraphEdge],
    door_leaf_mask: Optional[np.ndarray] = None,
    door_arc_mask: Optional[np.ndarray] = None,
    scale_info=None,
) -> tuple[list[DoorOriginPrimitive], list[DoorLeafPrimitive], list[DoorArcPrimitive]]:
    """Build origin/leaf/arc primitives for every accepted door-origin edge
    (spec_v008 SS13 / SS7 step 10)."""
    points_by_id = {p.id: p for p in points}
    evidence_mask = None
    if door_leaf_mask is not None and door_arc_mask is not None:
        evidence_mask = np.maximum(door_leaf_mask, door_arc_mask)
    elif door_leaf_mask is not None:
        evidence_mask = door_leaf_mask
    elif door_arc_mask is not None:
        evidence_mask = door_arc_mask

    origins: list[DoorOriginPrimitive] = []
    leaves: list[DoorLeafPrimitive] = []
    arcs: list[DoorArcPrimitive] = []
    counter = 0

    for edge in door_origin_edges:
        pa = points_by_id.get(edge.point_a_id)
        pb = points_by_id.get(edge.point_b_id)
        if pa is None or pb is None:
            continue
        if pa.point_type == "wall_door_hinge_point":
            hinge, end = pa, pb
        elif pb.point_type == "wall_door_hinge_point":
            hinge, end = pb, pa
        else:
            continue

        hinge_point = hinge.coordinate
        far_point = end.coordinate
        width_px = math.hypot(far_point[0] - hinge_point[0], far_point[1] - hinge_point[1])
        if width_px < 1e-6:
            continue
        orientation_angle = math.degrees(math.atan2(far_point[1] - hinge_point[1], far_point[0] - hinge_point[0]))
        center = ((hinge_point[0] + far_point[0]) / 2.0, (hinge_point[1] + far_point[1]) / 2.0)

        counter += 1
        origin = DoorOriginPrimitive(
            primitive_id=f"door_origin_{counter:04d}",
            center=center,
            width=width_px,
            orientation_angle=orientation_angle,
            width_mm=edge.length_mm,
            scale_info=scale_info,
            source_class_ids=[6],
        )
        origins.append(origin)

        swing_direction = _pick_swing_side(hinge_point, orientation_angle, width_px, evidence_mask)

        leaves.append(
            DoorLeafPrimitive(
                primitive_id=f"door_leaf_{counter:04d}",
                hinge_point=hinge_point,
                width=width_px,
                orientation_angle=orientation_angle,
                swing_direction=swing_direction,
                scale_info=scale_info,
                source_class_ids=[5],
            )
        )
        arcs.append(
            DoorArcPrimitive(
                primitive_id=f"door_arc_{counter:04d}",
                hinge_point=hinge_point,
                origin_far_point=far_point,
                width=width_px,
                orientation_angle=orientation_angle,
                swing_direction=swing_direction,
                scale_info=scale_info,
                source_class_ids=[4],
            )
        )

    return origins, leaves, arcs
