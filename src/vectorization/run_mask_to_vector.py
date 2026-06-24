"""Main entry point for the v008 orthogonal point-graph mask-to-vector
pipeline (spec_v008 SS7):

    load config -> decode + reject incompatible mask -> clean masks and
    extract connected components -> resolve scale -> search the seven point
    types -> validate points -> align points onto orthogonal axes -> connect
    wall/window/door-origin graph edges -> validate graph -> generate door
    leaf/arc -> generate wall/window final geometry -> export SVG -> write
    debug overlay + metrics.

Usage:
    python -m src.vectorization.run_mask_to_vector
    python -m src.vectorization.run_mask_to_vector --config configs/vectorization_v008.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs.
    yaml = None

from .components import extract_all_components
from .debug import build_debug_overlay, build_metrics, write_metrics
from .decode_prediction import IncompatibleMaskError, decode_class_id_mask, decode_color_mask
from .door_geometry import generate_door_geometry
from .export_svg import build_svg, save_svg
from .graph_types import MaskToVectorResult
from .load_prediction import find_prediction_images, load_image_as_array
from .masks import split_class_masks
from .point_alignment import align_points
from .point_connection import connect_points
from .point_detection import build_door_candidate_records, detect_points, validate_points
from .scale import ScaleInfo, resolve_scale_from_components
from .wall_geometry import wall_edges_to_primitives, window_edges_to_primitives

DEFAULT_DOOR_WIDTH_MODULES_MM = (700.0, 900.0)
DEFAULT_WALL_THICKNESS_MODULES_MM = (100.0, 200.0)


def load_config(config_path: str | Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load vectorization config files")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _scale_info_from_config(cfg: dict) -> ScaleInfo:
    """Placeholder ScaleInfo carrying any explicit px_to_mm override from
    config through to resolve_scale_from_components() - the real scale is
    resolved per-sample inside process_single() once component evidence is
    available."""
    scale_cfg = cfg.get("scale", {})
    explicit = scale_cfg.get("explicit_px_to_mm")
    return ScaleInfo(
        unit="px",
        px_to_mm=explicit,
        scale_status="resolved" if explicit else "unknown",
        scale_source="explicit_metadata" if explicit else "none",
        confidence=1.0 if explicit else 0.0,
    )


def decode_prediction_image(arr: np.ndarray, rgb_tolerance: int = 20) -> np.ndarray:
    """Decode either a class-ID mask (2D) or an RGB preview (3D) into a
    7-class class-ID map, rejecting any retired/incompatible input."""
    if arr.ndim == 2:
        return decode_class_id_mask(arr)
    return decode_color_mask(arr, tolerance=rgb_tolerance)


def process_single(
    image_path: Path,
    config: dict,
    scale_info: ScaleInfo,
    output_dir: Path,
    output_filename: str | None = None,
) -> MaskToVectorResult:
    print(f"  Processing: {image_path.name}")
    result = MaskToVectorResult()

    input_cfg = config.get("input", {})
    rgb = load_image_as_array(image_path)
    class_map = decode_prediction_image(rgb, input_cfg.get("rgb_tolerance", 20))

    all_masks = split_class_masks(class_map)
    result.decoded_masks = all_masks
    # Floor is ignored entirely for this restart (spec_v008 SS1/SS2).
    masks = {k: v for k, v in all_masks.items() if k != "floor"}

    components_cfg = config.get("components", {})
    min_area_px = {
        "wall": components_cfg.get("min_wall_area_px", 4.0),
        "window": components_cfg.get("min_window_area_px", 4.0),
        "door_arc": components_cfg.get("min_door_arc_area_px", 4.0),
        "door_leaf": components_cfg.get("min_door_leaf_area_px", 2.0),
        "door_origin": components_cfg.get("min_door_origin_area_px", 2.0),
    }
    components, rejected_components = extract_all_components(masks, min_area_px=min_area_px)
    result.components = components
    result.rejected_evidence.extend(rejected_components)

    scale_cfg = config.get("scale", {})
    explicit_px_to_mm = scale_info.px_to_mm if scale_info.scale_status == "resolved" else None
    resolved_scale = resolve_scale_from_components(
        components.get("door_arc", []),
        components.get("door_origin", []),
        components.get("wall", []),
        explicit_px_to_mm=explicit_px_to_mm,
        door_modules_mm=tuple(scale_cfg.get("door_width_modules_mm", DEFAULT_DOOR_WIDTH_MODULES_MM)),
        wall_modules_mm=tuple(scale_cfg.get("wall_thickness_modules_mm", DEFAULT_WALL_THICKNESS_MODULES_MM)),
        min_confidence=scale_cfg.get("min_scale_confidence_for_metric", 0.70),
    )
    result.scale_info = resolved_scale

    geometry_cfg = config.get("geometry", {})
    doors_cfg = config.get("doors", {})
    windows_cfg = config.get("windows", {})

    detect_cfg = {
        "cardinal_tolerance_deg": geometry_cfg.get("cardinal_tolerance_deg", 25.0),
        # task15 problem 4: a red door_arc cluster's *recognition* as a door
        # must not depend on point-search precision (must-rule 51 only
        # allows rejecting for min-area or zero plausible geometry) - these
        # two host-wall search radii are not must-rule-mandated numbers
        # (unlike the 200mm arc-bbox proximity floor below), so they're
        # raised to effectively "any wall anywhere in the image."
        "max_wall_dist": geometry_cfg.get("max_host_wall_dist_px", 100000.0),
        "min_hosted_width_px": geometry_cfg.get("min_hosted_width_px", 3.0),
        "corner_ambiguity_px": geometry_cfg.get("corner_ambiguity_px", 25.0),
        "min_remainder_px": geometry_cfg.get("min_remainder_px", 3.0),
        # task16: probe band (px) around each door_arc bbox edge/corner used
        # to score purple/black/orange evidence when picking the hinge/end
        # vertex pair.
        "hinge_probe_radius": doors_cfg.get("bbox_corner_probe_px", 14.0),
        # task18: a red door_arc bbox far from square (e.g. 1:3) is rejected
        # outright - the long/short side ratio must be at most 2:1.
        "max_door_bbox_aspect_ratio": doors_cfg.get("max_bbox_aspect_ratio", 2.0),
        "hinge_snap_to_wall_max_dist_px": doors_cfg.get("hinge_snap_to_wall_max_dist_px", 100000.0),
        "door_width_modules_mm": tuple(doors_cfg.get("door_width_modules_mm", DEFAULT_DOOR_WIDTH_MODULES_MM)),
        "min_window_width_mm": windows_cfg.get("min_width_mm", 300.0),
        "free_end_opening_proximity_px": geometry_cfg.get("free_end_opening_proximity_px", 20.0),
        # Rule 17 fixes this at 200mm exactly - not a tunable search radius,
        # kept unchanged.
        "door_point_max_dist_from_arc_mm": doors_cfg.get("max_hinge_end_distance_from_arc_mm", 200.0),
    }

    # --- 1. Search the seven allowed point types directly (SS9) ---
    points, point_rejected, wall_skeleton_edges = detect_points(components, masks, resolved_scale, detect_cfg)
    result.rejected_evidence.extend(point_rejected)
    result.raw_points = points

    # --- 2. Validate searched point counts and attachment directions (SS10) ---
    # "Accepted" (rule 40) excludes door_arc clusters point_detection.py
    # itself already legitimately rejected (rule 51: below min area, or no
    # plausible hinge/end geometry within rule 17's 200mm floor even after
    # fallback) - those are correct rejections, not a hinge/cluster-count
    # bug, so they must not double-count against this check.
    rejected_arc_ids = {r.component_id for r in point_rejected if r.class_name == "door_arc"}
    accepted_door_arc_count = len(components.get("door_arc", [])) - len(rejected_arc_ids)
    point_validation = validate_points(points, accepted_door_arc_count=accepted_door_arc_count)
    result.point_validation = point_validation

    # --- 3. Align compatible points onto orthogonal axes (SS11) ---
    align_cfg = {
        "axis_alignment_tolerance_mm": geometry_cfg.get("axis_alignment_tolerance_mm", 500.0),
        "px_fallback_tolerance": geometry_cfg.get("px_fallback_tolerance_px", 6.0),
    }
    aligned_points, alignment_issues = align_points(
        points, components.get("wall", []), resolved_scale, align_cfg, wall_skeleton_edges
    )
    result.aligned_points = aligned_points

    # task13: one door-candidate report per accepted red door_arc cluster.
    result.door_candidates = build_door_candidate_records(
        components.get("door_arc", []), aligned_points, result.rejected_evidence,
        masks, components.get("wall", []), resolved_scale,
    )

    # --- 4. Connect aligned points into wall/window/door-origin edges (SS12) ---
    connect_cfg = {
        "node_match_tolerance_px": geometry_cfg.get("node_match_tolerance_px", 12.0),
        "opening_match_tolerance_px": geometry_cfg.get("free_end_opening_proximity_px", 20.0),
        "corridor_slack_px": geometry_cfg.get("corridor_slack_px", 20.0),
    }
    edges, graph_validation = connect_points(
        aligned_points, wall_skeleton_edges, resolved_scale, connect_cfg, components.get("wall", [])
    )
    result.edges = edges
    result.graph_validation = alignment_issues + graph_validation

    # --- 5. Generate door leaf and door arc geometry (SS13) ---
    door_origin_edges = [e for e in edges if e.edge_type == "door_origin"]
    door_origins, door_leaves, door_arcs = generate_door_geometry(
        aligned_points, door_origin_edges, masks.get("door_leaf"), masks.get("door_origin"), resolved_scale,
    )
    result.door_origins = door_origins
    result.door_leaves = door_leaves
    result.door_arcs = door_arcs

    # --- 6. Generate wall and window final geometry (SS14) ---
    wall_edges = [e for e in edges if e.edge_type == "wall"]
    window_edges = [e for e in edges if e.edge_type == "window"]
    walls = wall_edges_to_primitives(wall_edges, resolved_scale)
    windows = window_edges_to_primitives(window_edges, resolved_scale)
    result.walls = walls
    result.windows = windows

    # --- 7. Export SVG ---
    h, w = rgb.shape[:2]
    svg_cfg = config.get("svg", {})
    svg_content = build_svg(
        image_width=w, image_height=h,
        walls=walls, windows=windows,
        door_origins=door_origins, door_leaves=door_leaves, door_arcs=door_arcs,
        scale_info=resolved_scale, svg_config=svg_cfg,
    )
    result.svg = svg_content

    if output_filename is not None:
        out_path = output_dir / output_filename
    else:
        stem = image_path.stem.replace("_prediction", "")
        out_path = output_dir / f"{stem}_vector.svg"
    save_svg(svg_content, out_path)

    # --- 8. Write debug overlay and metrics (SS15) ---
    debug_overlay = build_debug_overlay(
        rgb, aligned_points, edges, result.rejected_evidence, resolved_scale, result.door_candidates
    )
    debug_overlay.save(out_path.with_name("debug_overlay.png"))

    metrics = build_metrics(
        image_name=image_path.name,
        components=components,
        rejected_evidence=result.rejected_evidence,
        points=aligned_points,
        edges=edges,
        validation_issues=result.validation_issues,
        scale_info=resolved_scale,
        door_candidates=result.door_candidates,
    )
    write_metrics(out_path.with_name("metrics.json"), metrics)

    print(
        f"    walls={len(walls)}, windows={len(windows)}, doors={len(door_origins)}, "
        f"rejected={len(result.rejected_evidence)}, scale={resolved_scale.scale_status}"
    )
    print(f"    -> {out_path}")
    return result


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
        try:
            process_single(img_path, config, scale_info, output_dir)
        except IncompatibleMaskError as exc:
            print(f"  Skipping {img_path.name}: {exc}")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Orthogonal point-graph mask-to-vector pipeline (v008)")
    parser.add_argument("--config", default="configs/vectorization_v008.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
