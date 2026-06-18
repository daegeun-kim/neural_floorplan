"""Main entry point for the v008 mask-to-vector reconstruction pipeline.

Pipeline order (per spec_v008 / task06): wall -> floor -> opening -> icon.

Usage:
    python -m src.vectorization.run_mask_to_vector
    python -m src.vectorization.run_mask_to_vector --config configs/vectorization_v008.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .cleanup import clean_icon_mask, clean_opening_mask, clean_room_mask, clean_wall_mask
from .decode_prediction import decode_color_mask
from .export_svg import build_svg, save_svg
from .floor_extraction import extract_floor
from .geometry_rules import apply_geometry_rules, split_walls_at_openings
from .icon_extraction import extract_icons
from .load_prediction import find_prediction_images, load_image_as_array
from .masks import split_class_masks
from .opening_classification import ClassificationConfig, classify_openings
from .opening_extraction import extract_openings
from .primitives import ScaleInfo
from .wall_extraction import extract_walls


def load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _scale_info_from_config(cfg: dict) -> ScaleInfo:
    scale_cfg = cfg.get("scale", {})
    return ScaleInfo(
        scale_factor=scale_cfg.get("fallback_scale", 1.0),
        unit=scale_cfg.get("fallback_unit", "px"),
        scale_status="unknown",
    )


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
    opening_mask = clean_opening_mask(
        raw_masks["opening"],
        min_area=cleanup_cfg.get("min_opening_component_area", 8),
    )
    room_mask = clean_room_mask(
        raw_masks["room"],
        min_area=cleanup_cfg.get("min_room_component_area", 100),
    )
    icon_mask = clean_icon_mask(
        raw_masks["icon"],
        min_area=cleanup_cfg.get("min_icon_component_area", 20),
    )

    # --- 1. Wall: outer rectilinear loop first, then inner walls ---
    wall_cfg = config.get("wall_extraction", {})
    outer_walls, inner_walls, outer_polygon = extract_walls(
        wall_mask,
        room_mask=room_mask,
        snap_angle_deg=wall_cfg.get("snap_angle_degrees", 8.0),
        merge_distance_px=wall_cfg.get("merge_distance_px", 6.0),
        min_wall_length_px=wall_cfg.get("min_wall_length_px", 10.0),
        scale_info=scale_info,
    )
    walls = outer_walls + inner_walls

    # --- 2. Floor: direct translation of the outer wall loop ---
    floor = extract_floor(outer_polygon, scale_info=scale_info)

    # --- 3. Opening: extract, snap+project onto walls, classify, split ---
    openings_raw = extract_openings(opening_mask, walls, scale_info=scale_info)

    walls, openings_raw = apply_geometry_rules(
        walls, openings_raw,
        snap_threshold_deg=wall_cfg.get("snap_angle_degrees", 8.0),
    )

    cls_cfg = config.get("opening_classification", {})
    cls_config = ClassificationConfig(
        window_min_aspect_ratio=cls_cfg.get("window_min_aspect_ratio", 2.5),
        door_max_aspect_ratio=cls_cfg.get("door_max_aspect_ratio", 2.0),
        min_confidence_for_type=cls_cfg.get("min_confidence_for_type", 0.65),
    )
    enabled = cls_cfg.get("enabled", True)
    if enabled:
        doors, windows, unresolved = classify_openings(
            openings_raw, opening_mask, cls_config, room_mask=room_mask, walls=walls
        )
    else:
        doors, windows, unresolved = [], [], openings_raw

    hosted_for_split = [o for o in (doors + windows) if o.host_wall_id]
    if hosted_for_split:
        walls = split_walls_at_openings(walls, hosted_for_split)

    # --- 4. Icon: simplified filled shapes, generated last ---
    icon_cfg = config.get("icon_extraction", {})
    icons = extract_icons(
        icon_mask,
        min_area=icon_cfg.get("min_icon_area_px", 20),
        scale_info=scale_info,
    )

    h, w = rgb.shape[:2]
    svg_cfg = config.get("svg", {})
    svg_content = build_svg(
        image_width=w,
        image_height=h,
        walls=walls,
        doors=doors,
        windows=windows,
        icons=icons,
        floor=floor,
        unresolved_openings=unresolved,
        scale_info=scale_info,
        svg_config=svg_cfg,
    )

    if output_filename is not None:
        out_path = output_dir / output_filename
    else:
        stem = image_path.stem.replace("_prediction", "")
        out_path = output_dir / f"{stem}_vector.svg"
    save_svg(svg_content, out_path)

    print(
        f"    walls={len(walls)} (outer={len(outer_walls)}, inner={len(inner_walls)}), "
        f"floor={'yes' if floor else 'no'}, doors={len(doors)}, "
        f"windows={len(windows)}, icons={len(icons)}, "
        f"unresolved={len(unresolved)}"
    )
    print(f"    -> {out_path}")


def run(config_path: str | Path = "configs/vectorization_v008.yaml") -> None:
    config = load_config(config_path)
    scale_info = _scale_info_from_config(config)

    input_cfg = config.get("input", {})
    preview_dir = Path(input_cfg.get("preview_dir", "runs/segformer_b0/previews/epoch_030"))
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
    parser = argparse.ArgumentParser(description="Mask-to-vector pipeline (v008)")
    parser.add_argument(
        "--config",
        default="configs/vectorization_v008.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
