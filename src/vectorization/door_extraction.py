"""Reconstruct doors from door_arc / door_leaf / door_origin mask evidence.

Arc-group-led procedure per task10: red door_arc connected components are
the sole standard for door count and location - no door_arc group means no
door, full stop, regardless of how much door_origin/door_leaf evidence
exists nearby. Per red arc group:

  1. Find the hinge from the orange(door_leaf)/purple(door_origin)
     intersection near the arc group; fall back to inferring the hinge from
     the arc's own geometry + nearest wall if that intersection is missing.
  2. Snap the hinge candidate onto the nearest wall (outer or inner).
  3. Pair the snapped hinge with the door_origin evidence's far endpoint
     (the required orange/purple pair) - no pairing partner means the
     evidence stays debug-only, never becomes a door.
  4. Snap the resulting width to an architectural-scale door module
     (700/900mm) - no scale, no door (no pixel-sized fallback).
  5. Build DoorOriginPrimitive/DoorLeafPrimitive/DoorArcPrimitive
     procedurally from the resolved hinge/far-point/swing-side - never
     tracing the raw leaf/arc pixel contours directly.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry_rules import nearest_wall, project_pixels_onto_wall, select_host_wall_for_opening
from .primitives import OpeningPrimitive, ScaleInfo, WallPrimitive
from .primitives.door import DoorArcPrimitive, DoorLeafPrimitive, DoorOriginPrimitive, SwingSide
from .wall_geometry import _project_point_onto_line

DOOR_WIDTH_MODULES_MM: tuple[float, ...] = (700.0, 900.0)


def _nearest_door_module_mm(value_mm: float, modules_mm: tuple[float, ...]) -> float:
    """Always pick the nearest module - unlike scale.snap_to_module_mm (which
    is a scale-estimation voting helper that backs off when nothing is
    close), a door's final width must always commit to one of the valid
    modules (task10: door width is exactly 700mm or 900mm)."""
    best = modules_mm[0]
    best_err = abs(value_mm - best)
    for module in modules_mm[1:]:
        err = abs(value_mm - module)
        if err < best_err:
            best_err = err
            best = module
    return best


def _count_evidence_in_radius(
    mask: np.ndarray, point: tuple[float, float], radius: float
) -> int:
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
    evidence_mask: np.ndarray,
    n_probes: int = 6,
    probe_radius: float = 10.0,
) -> SwingSide:
    """Probe both perpendicular directions from the hinge; the side with more
    door_leaf/door_arc evidence is the swing side."""
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


def _extend_point(
    origin: tuple[float, float], through: tuple[float, float], new_distance: float
) -> tuple[float, float]:
    """Point on the ray from `origin` through `through`, at `new_distance`."""
    ox, oy = origin
    tx, ty = through
    dx, dy = tx - ox, ty - oy
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return through
    ux, uy = dx / dist, dy / dist
    return (ox + ux * new_distance, oy + uy * new_distance)


def _orange_purple_intersection(
    leaf_mask: np.ndarray,
    origin_mask: np.ndarray,
    near_point: tuple[float, float],
    search_radius: float,
    tolerance_px: float,
) -> tuple[float, float] | None:
    """Centroid of the orange(door_leaf)/purple(door_origin) overlap near
    `near_point`, or None if they don't intersect (within `tolerance_px`)."""
    h, w = leaf_mask.shape
    nx, ny = near_point
    x0, x1 = max(0, int(nx - search_radius)), min(w, int(nx + search_radius) + 1)
    y0, y1 = max(0, int(ny - search_radius)), min(h, int(ny + search_radius) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    leaf_roi = leaf_mask[y0:y1, x0:x1]
    origin_roi = origin_mask[y0:y1, x0:x1]
    if not leaf_roi.any() or not origin_roi.any():
        return None

    tol = max(int(round(tolerance_px)), 0)
    if tol > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        leaf_dilated = cv2.dilate(leaf_roi, kernel)
    else:
        leaf_dilated = leaf_roi

    overlap = cv2.bitwise_and(leaf_dilated, origin_roi)
    ys, xs = np.nonzero(overlap)
    if len(xs) == 0:
        return None
    return (float(xs.mean()) + x0, float(ys.mean()) + y0)


def _infer_hinge_from_arc_geometry(
    arc_pixel_coords: np.ndarray, host_wall: WallPrimitive
) -> tuple[float, float]:
    """Fallback hinge: the corner of the arc evidence's oriented bounding box
    closest to the host wall's centerline (the swing wedge's pivot corner)."""
    rect = cv2.minAreaRect(arc_pixel_coords.astype(np.float32))
    corners = cv2.boxPoints(rect)
    best_point = (float(corners[0][0]), float(corners[0][1]))
    best_dist = math.inf
    for cx, cy in corners:
        _, dist, _ = _project_point_onto_line((float(cx), float(cy)), host_wall.start, host_wall.end)
        if dist < best_dist:
            best_dist = dist
            best_point = (float(cx), float(cy))
    return best_point


def _hinge_snap_to_wall(
    hinge_candidate: tuple[float, float],
    walls: list[WallPrimitive],
    max_dist_px: float,
) -> tuple[tuple[float, float], WallPrimitive] | None:
    """Snap a single hinge point onto the nearest wall (outer or inner)."""
    host_wall = nearest_wall(hinge_candidate, walls, max_dist=max_dist_px)
    if host_wall is None:
        return None
    proj, _dist, _t = _project_point_onto_line(hinge_candidate, host_wall.start, host_wall.end)
    return proj, host_wall


def _find_paired_far_point(
    door_origin_mask: np.ndarray,
    origin_labels: np.ndarray,
    origin_centroids: np.ndarray | None,
    hinge_point: tuple[float, float],
    host_wall: WallPrimitive,
    probe_radius: float,
) -> tuple[tuple[float, float], int] | None:
    """Find the purple door_origin evidence paired with `hinge_point` and
    return its far endpoint projected onto `host_wall`, plus the claimed
    origin component label (for unpaired-evidence bookkeeping)."""
    if door_origin_mask is None or not door_origin_mask.any():
        return None
    h, w = door_origin_mask.shape
    hx, hy = hinge_point
    x0, x1 = max(0, int(hx - probe_radius)), min(w, int(hx + probe_radius) + 1)
    y0, y1 = max(0, int(hy - probe_radius)), min(h, int(hy + probe_radius) + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    window = origin_labels[y0:y1, x0:x1]
    candidate_labels = np.unique(window[window > 0])
    if candidate_labels.size == 0:
        return None
    if candidate_labels.size > 1 and origin_centroids is not None:
        candidate_labels = sorted(
            candidate_labels,
            key=lambda lbl: math.hypot(
                origin_centroids[lbl][0] - hx, origin_centroids[lbl][1] - hy
            ),
        )
    chosen_label = int(candidate_labels[0])

    seg_len = host_wall.length
    if seg_len < 1e-6:
        return None
    ys, xs = np.where(origin_labels == chosen_label)
    pixel_coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    _center, _width, t_min_norm, t_max_norm = project_pixels_onto_wall(pixel_coords, host_wall)

    x1w, y1w = host_wall.start
    x2w, y2w = host_wall.end
    ux, uy = (x2w - x1w) / seg_len, (y2w - y1w) / seg_len
    t_min_px, t_max_px = t_min_norm * seg_len, t_max_norm * seg_len
    pt_min = (x1w + ux * t_min_px, y1w + uy * t_min_px)
    pt_max = (x1w + ux * t_max_px, y1w + uy * t_max_px)
    dist_min = math.hypot(pt_min[0] - hx, pt_min[1] - hy)
    dist_max = math.hypot(pt_max[0] - hx, pt_max[1] - hy)
    far_point = pt_max if dist_max >= dist_min else pt_min
    return far_point, chosen_label


def raw_door_origin_lengths_px(door_origin_mask: np.ndarray) -> list[float]:
    """Per-component long-axis length (px) of every door_origin component,
    measured directly from the mask - no wall hosting required. Used to
    resolve metric scale before doors are extracted."""
    if door_origin_mask is None or not door_origin_mask.any():
        return []
    n, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        door_origin_mask, connectivity=8
    )
    lengths: list[float] = []
    for i in range(1, n):
        ys, xs = np.where(labels == i)
        if len(xs) < 2:
            continue
        pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        _center, (rw, rh), _angle = cv2.minAreaRect(pts)
        lengths.append(float(max(rw, rh)))
    return lengths


def _unresolved_opening(
    primitive_id: str,
    center: tuple[float, float],
    width: float,
    orientation_angle: float,
    host_wall_id: str | None,
    opening_type: str,
    confidence: float,
    scale_info: ScaleInfo,
    source_class_ids: list[int],
    bbox_px: tuple[float, float, float, float] | None,
    area_px: float | None,
) -> OpeningPrimitive:
    return OpeningPrimitive(
        primitive_id=primitive_id,
        center=center,
        width=width,
        orientation_angle=orientation_angle,
        host_wall_id=host_wall_id,
        opening_type=opening_type,
        confidence=confidence,
        scale_info=scale_info,
        source_class_ids=source_class_ids,
        source_evidence_bbox_px=bbox_px,
        source_evidence_area_px=area_px,
    )


def extract_doors(
    door_origin_mask: np.ndarray,
    door_leaf_mask: np.ndarray,
    door_arc_mask: np.ndarray,
    walls: list[WallPrimitive],
    min_area: int = 4,
    max_wall_dist: float = 40.0,
    min_hosted_width_px: float = 10.0,
    hinge_probe_radius: float = 14.0,
    min_confidence_for_metric: float = 0.70,
    scale_info: ScaleInfo | None = None,
    hinge_intersection_tolerance_px: float = 6.0,
    hinge_snap_to_wall_max_dist_px: float = 40.0,
    hinge_arc_inference_enabled: bool = True,
    door_width_modules_mm: tuple[float, ...] = DOOR_WIDTH_MODULES_MM,
    corner_ambiguity_px: float = 25.0,
    min_remainder_px: float = 3.0,
) -> tuple[
    list[DoorOriginPrimitive],
    list[DoorLeafPrimitive],
    list[DoorArcPrimitive],
    list[OpeningPrimitive],
]:
    """Return (door_origins, door_leaves, door_arcs, unresolved debug markers)."""
    scale_info = scale_info or ScaleInfo()
    origins: list[DoorOriginPrimitive] = []
    leaves: list[DoorLeafPrimitive] = []
    arcs: list[DoorArcPrimitive] = []
    unresolved: list[OpeningPrimitive] = []

    if door_arc_mask is None or not door_arc_mask.any():
        # No red door_arc evidence at all -> no doors, regardless of any
        # door_origin/door_leaf evidence that might exist (task10).
        return origins, leaves, arcs, unresolved

    evidence_mask = np.maximum(door_leaf_mask, door_arc_mask)

    has_origin = door_origin_mask is not None and door_origin_mask.any()
    if has_origin:
        n_origin, origin_labels, origin_stats, origin_centroids = cv2.connectedComponentsWithStats(
            door_origin_mask, connectivity=8
        )
    else:
        n_origin, origin_labels, origin_stats, origin_centroids = 1, None, None, None
    claimed_origin_labels: set[int] = set()

    n_arc, arc_labels, arc_stats, arc_centroids = cv2.connectedComponentsWithStats(
        door_arc_mask, connectivity=8
    )

    for i in range(1, n_arc):
        area = arc_stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        arc_cx, arc_cy = arc_centroids[i]
        arc_center = (float(arc_cx), float(arc_cy))
        bw = arc_stats[i, cv2.CC_STAT_WIDTH]
        bh = arc_stats[i, cv2.CC_STAT_HEIGHT]
        x0 = arc_stats[i, cv2.CC_STAT_LEFT]
        y0 = arc_stats[i, cv2.CC_STAT_TOP]
        bbox_px = (float(x0), float(y0), float(x0 + bw), float(y0 + bh))

        ys, xs = np.where(arc_labels == i)
        arc_pixel_coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

        hinge_candidate = _orange_purple_intersection(
            door_leaf_mask,
            door_origin_mask,
            arc_center,
            search_radius=hinge_probe_radius * 2.0,
            tolerance_px=hinge_intersection_tolerance_px,
        )

        if hinge_candidate is None:
            if not hinge_arc_inference_enabled:
                unresolved.append(
                    _unresolved_opening(
                        f"door_arc_unresolved_{i:04d}", arc_center, float(max(bw, bh)), 0.0,
                        None, "unresolved_door_arc", 0.3, scale_info, [4], bbox_px, float(area),
                    )
                )
                continue
            provisional_host = select_host_wall_for_opening(
                arc_pixel_coords, walls, max_dist=max_wall_dist,
                corner_ambiguity_px=corner_ambiguity_px, min_remainder_px=min_remainder_px,
            )
            if provisional_host is None:
                unresolved.append(
                    _unresolved_opening(
                        f"door_arc_unresolved_{i:04d}", arc_center, float(max(bw, bh)), 0.0,
                        None, "unresolved_door_arc", 0.3, scale_info, [4], bbox_px, float(area),
                    )
                )
                continue
            hinge_candidate = _infer_hinge_from_arc_geometry(arc_pixel_coords, provisional_host)

        snapped = _hinge_snap_to_wall(hinge_candidate, walls, hinge_snap_to_wall_max_dist_px)
        if snapped is None:
            unresolved.append(
                _unresolved_opening(
                    f"door_hinge_unresolved_{i:04d}", hinge_candidate, float(max(bw, bh)), 0.0,
                    None, "unresolved_door_hinge", 0.3, scale_info, [4], bbox_px, float(area),
                )
            )
            continue
        hinge_point, host_wall = snapped

        far_result = _find_paired_far_point(
            door_origin_mask, origin_labels, origin_centroids, hinge_point, host_wall, hinge_probe_radius
        )
        if far_result is None:
            # Orange (hinge) evidence found, but no paired purple (far point)
            # evidence nearby - required-pair rule: stays debug-only.
            unresolved.append(
                _unresolved_opening(
                    f"door_hinge_unresolved_{i:04d}", hinge_point, 0.0, host_wall.orientation_angle,
                    host_wall.primitive_id, "unresolved_door_hinge", 0.3, scale_info, [4], bbox_px, float(area),
                )
            )
            continue
        far_point, origin_label = far_result
        claimed_origin_labels.add(origin_label)

        raw_width_px = math.hypot(far_point[0] - hinge_point[0], far_point[1] - hinge_point[1])
        if raw_width_px < min_hosted_width_px:
            unresolved.append(
                _unresolved_opening(
                    f"door_too_narrow_{i:04d}", hinge_point, raw_width_px, host_wall.orientation_angle,
                    host_wall.primitive_id, "unresolved_door_too_narrow", 0.2, scale_info, [4, 6], bbox_px, float(area),
                )
            )
            continue

        if (
            scale_info.px_to_mm is None
            or scale_info.scale_status not in ("resolved", "estimated")
            or scale_info.confidence < min_confidence_for_metric
        ):
            # Scale not resolved/confident enough - do not generate a
            # pixel-sized door (task10: no architectural rule silently
            # falls back to arbitrary pixel geometry).
            unresolved.append(
                _unresolved_opening(
                    f"door_scale_blocked_{i:04d}", hinge_point, raw_width_px, host_wall.orientation_angle,
                    host_wall.primitive_id, "unresolved_door_scale_blocked", 0.4, scale_info, [4, 6], bbox_px, float(area),
                )
            )
            continue

        width_mm = _nearest_door_module_mm(raw_width_px * scale_info.px_to_mm, door_width_modules_mm)
        snapped_width_px = width_mm / scale_info.px_to_mm
        far_point = _extend_point(hinge_point, far_point, snapped_width_px)
        center = ((hinge_point[0] + far_point[0]) / 2.0, (hinge_point[1] + far_point[1]) / 2.0)

        origin = DoorOriginPrimitive(
            primitive_id=f"door_origin_{i:04d}",
            center=center,
            width=snapped_width_px,
            orientation_angle=host_wall.orientation_angle,
            width_mm=width_mm,
            host_wall_id=host_wall.primitive_id,
            confidence=1.0,
            scale_info=scale_info,
            source_class_ids=[6],
            source_evidence_bbox_px=bbox_px,
            source_evidence_area_px=float(area),
        )
        origins.append(origin)

        swing_direction = _pick_swing_side(
            hinge_point, host_wall.orientation_angle, snapped_width_px, evidence_mask
        )
        leaves.append(
            DoorLeafPrimitive(
                primitive_id=f"door_leaf_{i:04d}",
                hinge_point=hinge_point,
                width=snapped_width_px,
                orientation_angle=host_wall.orientation_angle,
                swing_direction=swing_direction,
                host_wall_id=host_wall.primitive_id,
                confidence=1.0,
                scale_info=scale_info,
                source_class_ids=[5],
            )
        )
        arcs.append(
            DoorArcPrimitive(
                primitive_id=f"door_arc_{i:04d}",
                hinge_point=hinge_point,
                origin_far_point=far_point,
                width=snapped_width_px,
                orientation_angle=host_wall.orientation_angle,
                swing_direction=swing_direction,
                host_wall_id=host_wall.primitive_id,
                confidence=1.0,
                scale_info=scale_info,
                source_class_ids=[4],
            )
        )

    if has_origin:
        for j in range(1, n_origin):
            if j in claimed_origin_labels:
                continue
            area_j = origin_stats[j, cv2.CC_STAT_AREA]
            if area_j < min_area:
                continue
            cx, cy = origin_centroids[j]
            bw = origin_stats[j, cv2.CC_STAT_WIDTH]
            bh = origin_stats[j, cv2.CC_STAT_HEIGHT]
            x0 = origin_stats[j, cv2.CC_STAT_LEFT]
            y0 = origin_stats[j, cv2.CC_STAT_TOP]
            unresolved.append(
                _unresolved_opening(
                    f"door_origin_unpaired_{j:04d}", (float(cx), float(cy)), float(max(bw, bh)), 0.0,
                    None, "unresolved_door_origin", 0.3, scale_info, [6],
                    (float(x0), float(y0), float(x0 + bw), float(y0 + bh)), float(area_j),
                )
            )

    return origins, leaves, arcs, unresolved
