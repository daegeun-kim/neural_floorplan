"""Main entry point for the v008 strict 7-class mask-to-vector reconstruction pipeline.

Pipeline order (per spec_v008 SS6, reordered by task10): decode + clean masks
-> wall topology (outer + inner, px) -> resolve scale -> inner-wall
outer-loop mm-attachment -> windows -> doors (arc-group-led) -> 45-degree
snap + opening reprojection -> wall splitting at openings -> floor -> export.

Scale is resolved right after wall extraction (not at the very end) because
several task10 rules (inner-wall outer-attachment, window minimum width,
door module snapping) are explicitly real-world-millimeter rules with no
pixel fallback, so they need a resolved/estimated ScaleInfo before they run.

Usage:
    python -m src.vectorization.run_mask_to_vector
    python -m src.vectorization.run_mask_to_vector --config configs/vectorization_v008.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .cleanup import (
    clean_door_arc_mask,
    clean_door_leaf_mask,
    clean_door_origin_mask,
    clean_wall_mask,
    clean_window_mask,
)
from .decode_prediction import decode_color_mask
from .door_extraction import extract_doors, raw_door_origin_lengths_px
from .export_svg import build_svg, save_svg
from .floor_extraction import extract_floor
from .geometry_rules import project_opening_onto_wall, snap_walls_to_45, split_walls_at_openings
from .load_prediction import find_prediction_images, load_image_as_array
from .masks import split_class_masks
from .primitives import ScaleInfo
from .primitives.scale import WALL_MODULES_MM, resolve_scale, snap_to_module_mm
from .wall_extraction import extract_walls, wall_thickness_samples_px
from .wall_geometry import snap_inner_endpoints_to_outer_wall_mm
from .window_extraction import extract_windows

DEFAULT_DOOR_WIDTH_MODULES_MM = (700.0, 900.0)


def load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _scale_info_from_config(cfg: dict) -> ScaleInfo:
    """Placeholder ScaleInfo used only before pixel evidence has been measured.

    The real scale is resolved per-sample inside process_single() once wall
    thickness and door_origin widths are available - this just carries any
    explicit px_to_mm override from config through to resolve_scale().
    """
    scale_cfg = cfg.get("scale", {})
    explicit = scale_cfg.get("explicit_px_to_mm")
    return ScaleInfo(
        unit="px",
        px_to_mm=explicit,
        scale_status="resolved" if explicit else "unknown",
        scale_source="explicit_metadata" if explicit else "none",
        confidence=1.0 if explicit else 0.0,
    )


def _rehost_door_geometry_after_snap(door_origins, door_leaves, door_arcs) -> None:
    """After walls are snapped/projected, origin.start/end shift slightly.

    Re-anchor each door's hinge/far point to the (now snapped) origin
    endpoints, keeping whichever endpoint was already closest to the old
    hinge as the hinge - the snap is small enough that it never flips which
    physical corner is the hinge.
    """
    for origin, leaf, arc in zip(door_origins, door_leaves, door_arcs):
        new_start, new_end = origin.start, origin.end
        old_hinge = leaf.hinge_point
        d_start = math.hypot(new_start[0] - old_hinge[0], new_start[1] - old_hinge[1])
        d_end = math.hypot(new_end[0] - old_hinge[0], new_end[1] - old_hinge[1])
        new_hinge, new_far = (new_start, new_end) if d_start <= d_end else (new_end, new_start)
        leaf.hinge_point = new_hinge
        leaf.orientation_angle = origin.orientation_angle
        arc.hinge_point = new_hinge
        arc.origin_far_point = new_far
        arc.orientation_angle = origin.orientation_angle


_DIMMED_ORANGE = (255, 200, 150)
_SOLID_ORANGE = (255, 136, 0)
_DIMMED_PURPLE = (210, 170, 220)
_SOLID_PURPLE = (160, 70, 180)
_SCALE_BLOCKED_GRAY = (160, 160, 160)


def _build_debug_overlay(
    rgb: np.ndarray,
    walls,
    windows,
    door_origins,
    door_leaves,
    door_arcs,
    outer_loop,
    unresolved,
    scale_info: ScaleInfo,
) -> Image.Image:
    """Pragmatic debug raster: wall centerlines, outer loop, host markers,
    hinge/far-point pairs, unresolved evidence, and a scale annotation.

    Per task10: orange square = hinge marker, purple circle = door-origin
    far/end marker. Resolved (paired) doors draw both solid; unresolved
    evidence draws a hollow/dimmed variant so pairing problems are visually
    distinct from scale-blocked problems (gray dashed).
    """
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)

    if outer_loop is not None and outer_loop.centerline:
        pts = outer_loop.centerline + [outer_loop.centerline[0]]
        draw.line(pts, fill=(150, 150, 150), width=1)

    for wall in walls:
        color = (80, 80, 200) if wall.wall_type == "outer" else (80, 180, 80)
        draw.line([wall.start, wall.end], fill=color, width=1)

    for window in windows:
        cx, cy = window.center
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=(60, 120, 220), width=2)

    for leaf, arc in zip(door_leaves, door_arcs):
        hx, hy = leaf.hinge_point
        draw.rectangle([hx - 4, hy - 4, hx + 4, hy + 4], outline=_SOLID_ORANGE, width=2)
        fx, fy = arc.origin_far_point
        draw.ellipse([fx - 4, fy - 4, fx + 4, fy + 4], outline=_SOLID_PURPLE, width=2)

    for op in unresolved:
        cx, cy = op.center
        if op.opening_type.endswith("_scale_blocked"):
            draw.rectangle([cx - 5, cy - 5, cx + 5, cy + 5], outline=_SCALE_BLOCKED_GRAY, width=1)
        elif op.opening_type == "unresolved_door_origin":
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=_DIMMED_PURPLE, width=1)
        else:
            draw.rectangle([cx - 5, cy - 5, cx + 5, cy + 5], outline=_DIMMED_ORANGE, width=1)

    label = (
        f"unit={scale_info.unit} status={scale_info.scale_status} "
        f"px_to_mm={scale_info.px_to_mm} conf={scale_info.confidence:.2f}"
    )
    draw.text((4, 4), label, fill=(255, 0, 0))
    return img


def process_single(
    image_path: Path,
    config: dict,
    scale_info: ScaleInfo,
    output_dir: Path,
    output_filename: str | None = None,
) -> None:
    print(f"  Processing: {image_path.name}")

    rgb = load_image_as_array(image_path)
    class_map = decode_color_mask(rgb)
    raw_masks = split_class_masks(class_map)

    cleanup_cfg = config.get("cleanup", {})
    wall_mask = clean_wall_mask(
        raw_masks["wall"],
        min_area=cleanup_cfg.get("min_wall_component_area", 20),
        close_gap_px=cleanup_cfg.get("close_wall_gap_px", 3),
    )
    window_mask = clean_window_mask(
        raw_masks["window"],
        min_area=cleanup_cfg.get("min_window_component_area", 8),
    )
    door_arc_mask = clean_door_arc_mask(
        raw_masks["door_arc"],
        min_area=cleanup_cfg.get("min_door_arc_component_area", 4),
    )
    door_leaf_mask = clean_door_leaf_mask(
        raw_masks["door_leaf"],
        min_area=cleanup_cfg.get("min_door_leaf_component_area", 4),
    )
    door_origin_mask = clean_door_origin_mask(
        raw_masks["door_origin"],
        min_area=cleanup_cfg.get("min_door_origin_component_area", 4),
    )

    scale_cfg = config.get("scale", {})
    min_confidence_for_metric = scale_cfg.get("min_scale_confidence_for_metric", 0.70)
    door_modules_mm = tuple(scale_cfg.get("door_width_modules_mm", DEFAULT_DOOR_WIDTH_MODULES_MM))
    wall_modules_mm = tuple(scale_cfg.get("wall_thickness_modules_mm", WALL_MODULES_MM))

    # --- 1. Wall topology: outer rectilinear loop first, then inner walls (px) ---
    # The outer loop must follow wall + opening (window/door) evidence only -
    # never floor, which is the CNN's least accurate class (task08).
    wall_cfg = config.get("walls", config.get("wall_extraction", {}))
    opening_evidence_mask = np.maximum.reduce(
        [window_mask, door_arc_mask, door_leaf_mask, door_origin_mask]
    )
    outer_walls, inner_walls, outer_polygon, outer_loop = extract_walls(
        wall_mask,
        opening_evidence_mask=opening_evidence_mask,
        door_origin_mask=door_origin_mask,
        merge_distance_px=wall_cfg.get("merge_distance_px", 6.0),
        min_wall_length_px=wall_cfg.get("min_wall_length_px", 10.0),
        connect_gap_px=wall_cfg.get("connect_gap_px", 20.0),
        scale_info=scale_info,
    )

    # --- 2. Resolve metric scale early (task10): several downstream rules ---
    # (inner-wall outer-attachment, window minimum width, door module snap)
    # are explicit real-world-mm rules with no pixel fallback, so scale must
    # be known before they run, not just annotated post-hoc at the end.
    explicit_px_to_mm = scale_info.px_to_mm if scale_info.scale_status == "resolved" else None
    door_lengths_px = raw_door_origin_lengths_px(door_origin_mask)
    wall_thickness_px = wall_thickness_samples_px(wall_mask)
    resolved_scale = resolve_scale(
        door_origin_lengths_px=door_lengths_px,
        wall_thickness_px=wall_thickness_px,
        explicit_px_to_mm=explicit_px_to_mm,
        door_modules_mm=door_modules_mm,
        wall_modules_mm=wall_modules_mm,
        min_confidence=min_confidence_for_metric,
    )
    scale_blocked: list[str] = []

    for wall in outer_walls + inner_walls:
        wall.scale_info = resolved_scale
        wall.thickness_mm, _ = snap_to_module_mm(
            wall.thickness, resolved_scale, wall_modules_mm, min_confidence_for_metric
        )

    # --- 3. Inner-wall endpoint attachment to the outer loop (mm, task10) ---
    inner_attach_threshold_mm = wall_cfg.get("inner_attach_outer_threshold_mm", 500.0)
    if resolved_scale.px_to_mm is not None and resolved_scale.scale_status in ("resolved", "estimated"):
        snapped_endpoints = snap_inner_endpoints_to_outer_wall_mm(
            inner_walls, outer_walls, resolved_scale, threshold_mm=inner_attach_threshold_mm
        )
    else:
        snapped_endpoints = {}
        scale_blocked.append("inner_wall_outer_attach_mm")
    walls = outer_walls + inner_walls

    # --- 4. Windows: host on walls (corner-safe), enforce 300mm minimum (px+mm) ---
    opening_cfg = config.get("openings", {})
    windows_cfg = config.get("windows", {})
    min_hosted_width_px = opening_cfg.get("min_hosted_width_px", 10.0)
    corner_ambiguity_px = opening_cfg.get("corner_ambiguity_px", 25.0)
    min_remainder_px = opening_cfg.get("min_remainder_px", 3.0)
    windows, unresolved_windows = extract_windows(
        window_mask,
        walls,
        max_wall_dist=opening_cfg.get("max_host_wall_dist_px", 40.0),
        min_hosted_width_px=min_hosted_width_px,
        min_confidence_for_metric=min_confidence_for_metric,
        scale_info=resolved_scale,
        min_width_mm=windows_cfg.get("min_width_mm", 300.0),
        corner_ambiguity_px=corner_ambiguity_px,
        min_remainder_px=min_remainder_px,
    )
    if any(o.opening_type == "unresolved_window_scale_blocked" for o in unresolved_windows):
        scale_blocked.append("window_min_width_mm")

    # --- 5. Doors: arc-group-led, hinge from orange/purple pairing (task10) ---
    doors_cfg = config.get("doors", {})
    door_origins, door_leaves, door_arcs, unresolved_doors = extract_doors(
        door_origin_mask,
        door_leaf_mask,
        door_arc_mask,
        walls,
        max_wall_dist=opening_cfg.get("max_host_wall_dist_px", 40.0),
        min_hosted_width_px=min_hosted_width_px,
        min_confidence_for_metric=min_confidence_for_metric,
        scale_info=resolved_scale,
        hinge_intersection_tolerance_px=doors_cfg.get("hinge_intersection_tolerance_px", 6.0),
        hinge_snap_to_wall_max_dist_px=doors_cfg.get("hinge_snap_to_wall_max_dist_px", 40.0),
        hinge_arc_inference_enabled=doors_cfg.get("hinge_arc_inference_enabled", True),
        door_width_modules_mm=tuple(
            doors_cfg.get("door_width_modules_mm", DEFAULT_DOOR_WIDTH_MODULES_MM)
        ),
        corner_ambiguity_px=corner_ambiguity_px,
        min_remainder_px=min_remainder_px,
    )
    if any(o.opening_type == "unresolved_door_scale_blocked" for o in unresolved_doors):
        scale_blocked.append("door_module_snap_mm")
    unresolved = unresolved_windows + unresolved_doors

    # --- 6. Snap walls to 45 degrees (orthogonal-first), then re-project hosted evidence ---
    walls = snap_walls_to_45(
        walls,
        ortho_snap_deg=wall_cfg.get("ortho_snap_degrees", 20.0),
        diagonal_snap_deg=wall_cfg.get("diagonal_snap_degrees", 10.0),
    )
    wall_map = {w.primitive_id: w for w in walls}
    for hosted in (*windows, *door_origins):
        if hosted.host_wall_id and hosted.host_wall_id in wall_map:
            project_opening_onto_wall(hosted, wall_map[hosted.host_wall_id])
    _rehost_door_geometry_after_snap(door_origins, door_leaves, door_arcs)

    # --- 7. Split walls at the now-snapped hosted windows/door-origins ---
    hosted_for_split = [o for o in (*windows, *door_origins) if o.host_wall_id]
    if hosted_for_split:
        walls = split_walls_at_openings(walls, hosted_for_split)

    # --- 8. Floor: direct translation of the outer wall loop ---
    floor = extract_floor(outer_polygon, scale_info=resolved_scale)

    # --- 9. Export SVG + debug overlay + metrics ---
    h, w = rgb.shape[:2]
    svg_cfg = config.get("svg", {})
    svg_content = build_svg(
        image_width=w,
        image_height=h,
        walls=walls,
        windows=windows,
        door_origins=door_origins,
        door_leaves=door_leaves,
        door_arcs=door_arcs,
        floor=floor,
        scale_info=resolved_scale,
        svg_config=svg_cfg,
    )

    if output_filename is not None:
        out_path = output_dir / output_filename
    else:
        stem = image_path.stem.replace("_prediction", "")
        out_path = output_dir / f"{stem}_vector.svg"
    save_svg(svg_content, out_path)

    debug_overlay = _build_debug_overlay(
        rgb, walls, windows, door_origins, door_leaves, door_arcs, outer_loop, unresolved, resolved_scale
    )
    debug_overlay.save(out_path.with_name("debug_overlay.png"))

    unresolved_doors_by_type: dict[str, int] = {}
    for op in unresolved_doors:
        unresolved_doors_by_type[op.opening_type] = unresolved_doors_by_type.get(op.opening_type, 0) + 1

    metrics = {
        "image": image_path.name,
        "walls": {
            "outer": len(outer_walls),
            "inner": len(inner_walls),
            "inner_attached_to_outer": len(snapped_endpoints),
            "final": len(walls),
        },
        "windows": {"resolved": len(windows), "unresolved": len(unresolved_windows)},
        "doors": {
            "resolved": len(door_origins),
            "unresolved": len(unresolved_doors),
            "unresolved_by_type": unresolved_doors_by_type,
        },
        "floor_present": floor is not None,
        "outer_loop_closed": outer_loop.is_closed() if outer_loop is not None else False,
        "scale_blocked": scale_blocked,
        "scale": {
            "unit": resolved_scale.unit,
            "px_to_mm": resolved_scale.px_to_mm,
            "scale_status": resolved_scale.scale_status,
            "scale_source": resolved_scale.scale_source,
            "confidence": resolved_scale.confidence,
        },
    }
    (out_path.with_name("metrics.json")).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"    walls={len(walls)} (outer={len(outer_walls)}, inner={len(inner_walls)}), "
        f"floor={'yes' if floor else 'no'}, windows={len(windows)}, doors={len(door_origins)}, "
        f"unresolved={len(unresolved)}, scale={resolved_scale.scale_status}, "
        f"scale_blocked={scale_blocked}"
    )
    print(f"    -> {out_path}")


def run(config_path: str | Path = "configs/vectorization_v008.yaml") -> None:
    config = load_config(config_path)
    scale_info = _scale_info_from_config(config)

    input_cfg = config.get("input", {})
    preview_dir = Path(input_cfg.get("preview_dir", "runs/segformer_b0_run3/previews/epoch_030"))
    filename_contains = input_cfg.get("filename_contains", "prediction")

    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("output_dir", "outputs/vectorization/v008"))

    images = find_prediction_images(preview_dir, filename_contains)
    if not images:
        print(f"No prediction images found in: {preview_dir}")
        return

    print(f"Found {len(images)} prediction image(s) in {preview_dir}")
    for img_path in images:
        process_single(img_path, config, scale_info, output_dir)

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict 7-class mask-to-vector pipeline (v008)")
    parser.add_argument(
        "--config",
        default="configs/vectorization_v008.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
