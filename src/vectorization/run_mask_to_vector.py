"""Main entry point for the v008 mask-to-vector reconstruction pipeline.

Usage:
    python -m src.vectorization.run_mask_to_vector
    python -m src.vectorization.run_mask_to_vector --config configs/vectorization_v008.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .cleanup import clean_opening_mask, clean_room_mask, clean_wall_mask
from .decode_prediction import decode_color_mask
from .export_svg import build_svg, save_svg
from .geometry_rules import apply_geometry_rules
from .load_prediction import find_prediction_images, load_image_as_array
from .masks import split_class_masks
from .opening_classification import ClassificationConfig, classify_openings
from .opening_extraction import extract_openings
from .primitives import ScaleInfo
from .room_extraction import extract_rooms
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

    wall_cfg = config.get("wall_extraction", {})
    walls = extract_walls(
        wall_mask,
        snap_angle_deg=wall_cfg.get("snap_angle_degrees", 8.0),
        merge_distance_px=wall_cfg.get("merge_distance_px", 6.0),
        min_wall_length_px=wall_cfg.get("min_wall_length_px", 10.0),
        scale_info=scale_info,
    )

    openings_raw = extract_openings(
        opening_mask,
        walls,
        scale_info=scale_info,
    )

    cls_cfg = config.get("opening_classification", {})
    cls_config = ClassificationConfig(
        window_min_aspect_ratio=cls_cfg.get("window_min_aspect_ratio", 2.5),
        door_max_aspect_ratio=cls_cfg.get("door_max_aspect_ratio", 2.0),
        min_confidence_for_type=cls_cfg.get("min_confidence_for_type", 0.65),
    )
    enabled = cls_cfg.get("enabled", True)
    if enabled:
        doors, windows, unresolved = classify_openings(openings_raw, opening_mask, cls_config)
    else:
        doors, windows, unresolved = [], [], openings_raw

    rooms = extract_rooms(room_mask, scale_info=scale_info)

    walls, unresolved = apply_geometry_rules(
        walls, unresolved,
        snap_threshold_deg=wall_cfg.get("snap_angle_degrees", 8.0),
    )

    h, w = rgb.shape[:2]
    svg_cfg = config.get("svg", {})
    svg_content = build_svg(
        image_width=w,
        image_height=h,
        walls=walls,
        openings=[],
        doors=doors,
        windows=windows,
        rooms=rooms,
        unresolved_openings=unresolved,
        scale_info=scale_info,
        svg_config=svg_cfg,
    )

    stem = image_path.stem.replace("_prediction", "")
    out_path = output_dir / f"{stem}_vector.svg"
    save_svg(svg_content, out_path)

    print(
        f"    walls={len(walls)}, doors={len(doors)}, "
        f"windows={len(windows)}, rooms={len(rooms)}, "
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
