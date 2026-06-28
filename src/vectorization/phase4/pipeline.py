"""Phase 4 graph-to-vector pipeline orchestrator (spec_v008_phase4_vectorization.md).

Entry point: run_phase4_pipeline()

Full pipeline:
    1. Shared preprocessing  -> input.png + manifest
    2. R2G inference         -> graph_pred.json / graph_pred.svg / graph_overlay.png
    3. Graph orthogonalization -> graph_overlay_aligned.png
    4. Segmentation inference -> image_segmentation.png
    5. Component extraction
    6. Scale inference
    7. Opening detection (doors + windows)
    8. Opening hosting on same wall edge
    9. Conflict resolution
    10. Wall interval trimming
    11. Wall chain buffering
    12. Export final_vector.svg + final_vector.json
    13. Write image_debug_overlay.png
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .debug_overlay import build_debug_overlay, write_debug_overlay
from .door_geometry import DoorGeometry, compute_door_geometry
from .export_json import build_final_vector_json, write_final_vector_json
from .export_svg import build_final_svg, write_final_svg
from .graph_alignment import normalize_graph
from .opening_detection import detect_door_candidates, detect_window_candidates
from .opening_hosting import RejectedOpening, host_openings
from .preprocessing import preprocess_image
from .scale_inference import infer_scale_from_components
from .wall_buffering import WallGeometry, buffer_wall_chains
from .wall_interval_editing import (
    TrimmedGraph,
    apply_adjusted_intervals_to_hosted_openings,
    trim_wall_intervals,
)
from ..components import extract_all_components
from ..masks import split_class_masks
from ..primitives.scale import ScaleInfo

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EXTERNAL_DIR = _PROJECT_ROOT / "external" / "raster_to_graph"
_DEFAULT_SEG_CKPT = _PROJECT_ROOT / "checkpoints_CNN" / "segformer_b0_run3" / "best.pt"
_DEFAULT_R2G_CKPT = _PROJECT_ROOT / "checkpoints_Raster2Graph" / "checkpoint0299.pth"


@dataclass
class Phase4Result:
    """All intermediate artifacts from one Phase 4 pipeline run."""
    # Inputs
    source_image_path: str = ""
    output_dir: str = ""
    # Preprocessing
    preprocessing_manifest: dict = field(default_factory=dict)
    input_pil: Optional[Any] = None          # PIL.Image 512x512
    # R2G graph (raw)
    raw_graph: dict = field(default_factory=dict)
    # Aligned graph
    aligned_graph: dict = field(default_factory=dict)
    # Segmentation
    seg_class_map: Optional[np.ndarray] = None
    seg_masks: dict = field(default_factory=dict)
    components: dict = field(default_factory=dict)
    # Scale
    scale_info: Optional[ScaleInfo] = None
    # Opening candidates
    door_candidates_accepted: list = field(default_factory=list)
    door_candidates_rejected: list = field(default_factory=list)
    window_candidates_accepted: list = field(default_factory=list)
    window_candidates_rejected: list = field(default_factory=list)
    # Hosted openings
    hosted_doors: list = field(default_factory=list)
    hosted_windows: list = field(default_factory=list)
    rejected_openings: list = field(default_factory=list)
    # Trimmed wall graph
    trimmed_graph: Optional[TrimmedGraph] = None
    # Final geometry
    wall_geometry: Optional[WallGeometry] = None
    # Outputs
    final_vector_json: dict = field(default_factory=dict)
    final_vector_svg: str = ""
    # Metrics
    metrics: dict = field(default_factory=dict)
    elapsed_s: float = 0.0


def _ensure_r2g_importable() -> None:
    ext = str(_EXTERNAL_DIR)
    if ext not in sys.path:
        sys.path.insert(0, ext)


def _run_r2g_inference(
    input_pil: Image.Image,
    output_dir: Path,
    r2g_ckpt: Path,
) -> dict:
    """Run Raster-to-Graph inference and return the predicted graph dict."""
    import gc
    import torch

    _ensure_r2g_importable()
    from args import get_args_parser  # type: ignore[import]
    from models.build import build_model, build_postprocessor  # type: ignore[import]
    from util.random_utils import set_random_seed  # type: ignore[import]
    from run_inference_generous_phase4 import (  # type: ignore[import]
        GENEROUS, MASK_RERUN, MERGE, FILTERS, SCORING,
        run_generous_multistart,
        merge_components,
        apply_light_post_merge_filter,
        compute_soft_scores,
        make_svg_merged,
        make_overlay_normal,
        _find_components,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — R2G inference requires a GPU.")

    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda")
    args_p = get_args_parser()
    args = args_p.parse_args([])
    args.device = "cuda"
    set_random_seed(args)

    model = build_model(args).to(device)
    postprocessor = build_postprocessor()
    ckpt = torch.load(str(r2g_ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    gray_arr = np.asarray(input_pil.convert("L"))

    accepted, discarded = run_generous_multistart(
        model, postprocessor, input_pil, device, GENEROUS, MASK_RERUN
    )

    merged_pts_raw, merged_edges_raw = merge_components(
        accepted,
        snap_tol  = MERGE["node_snap_tolerance_px"],
        inter_tol = MERGE["edge_intersection_tolerance_px"],
        col_tol   = MERGE["collinear_overlap_tolerance_px"],
    )

    merged_pts, merged_edges, _ = apply_light_post_merge_filter(
        merged_pts_raw, merged_edges_raw, gray_arr, FILTERS
    )

    graph_json = {
        "nodes": [[int(x), int(y)] for x, y in merged_pts],
        "edges": [[int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])] for p1, p2 in merged_edges],
    }

    # Save R2G outputs
    (output_dir / "graph_pred.json").write_text(
        json.dumps(graph_json, indent=2), encoding="utf-8"
    )
    (output_dir / "graph_pred.svg").write_text(
        make_svg_merged(merged_pts, merged_edges), encoding="utf-8"
    )
    make_overlay_normal(input_pil, merged_pts, merged_edges).save(
        str(output_dir / "graph_overlay.png")
    )

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return graph_json


def _draw_graph_overlay(
    input_pil: Image.Image,
    aligned_graph: dict,
    output_path: Path,
) -> None:
    """Draw the orthogonally aligned graph on the input image."""
    import cv2
    img = np.array(input_pil.convert("RGB")).copy()
    for e in aligned_graph.get("aligned_edges", aligned_graph.get("edges", [])):
        x1, y1, x2, y2 = int(e[0]), int(e[1]), int(e[2]), int(e[3])
        cv2.line(img, (x1, y1), (x2, y2), (0, 160, 0), 2)
    for node in aligned_graph.get("aligned_nodes", aligned_graph.get("nodes", [])):
        cv2.circle(img, (int(node[0]), int(node[1])), 4, (0, 200, 0), -1)
    Image.fromarray(img).save(str(output_path))


def run_phase4_pipeline(
    image_path: str | Path,
    output_dir: str | Path,
    seg_checkpoint: Optional[str | Path] = None,
    r2g_checkpoint: Optional[str | Path] = None,
    explicit_px_to_mm: Optional[float] = None,
    max_perp_dist_px: float = 20.0,
    run_r2g: bool = True,
    existing_graph_json: Optional[str | Path] = None,
    preview_half_width_px: float = 8.0,
) -> Phase4Result:
    """Run the full Phase 4 graph-to-vector pipeline.

    Args:
        image_path: Input floor plan image (any format/size)
        output_dir: Directory to write all output files
        seg_checkpoint: Path to 7-class SegFormer checkpoint (best.pt)
        r2g_checkpoint: Path to Raster-to-Graph checkpoint
        explicit_px_to_mm: Override scale (skip inference)
        max_perp_dist_px: Max perpendicular distance for opening hosting
        run_r2g: If False, use existing_graph_json instead of running R2G
        existing_graph_json: Path to pre-existing graph_pred.json
        preview_half_width_px: Wall half-width fallback when scale unknown

    Returns:
        Phase4Result with all intermediate and final artifacts
    """
    t0 = time.perf_counter()
    image_path = Path(image_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_ckpt = Path(seg_checkpoint) if seg_checkpoint else _DEFAULT_SEG_CKPT
    r2g_ckpt = Path(r2g_checkpoint) if r2g_checkpoint else _DEFAULT_R2G_CKPT

    result = Phase4Result(
        source_image_path=str(image_path),
        output_dir=str(output_dir),
    )

    # --- 1. Shared preprocessing ---
    print("[phase4] 1/13 preprocessing...")
    input_pil, manifest = preprocess_image(image_path)
    result.preprocessing_manifest = manifest
    result.input_pil = input_pil
    input_pil.save(str(output_dir / "input.png"))

    # --- 2. R2G inference ---
    if run_r2g:
        print("[phase4] 2/13 R2G inference...")
        raw_graph = _run_r2g_inference(input_pil, output_dir, r2g_ckpt)
    elif existing_graph_json is not None:
        print("[phase4] 2/13 loading existing graph_pred.json...")
        raw_graph = json.loads(Path(existing_graph_json).read_text(encoding="utf-8"))
        # Copy svg/overlay if they exist
        src_dir = Path(existing_graph_json).parent
        for fname in ("graph_pred.svg", "graph_pred.json", "graph_overlay.png"):
            src = src_dir / fname
            if src.exists() and src.resolve() != (output_dir / fname).resolve():
                import shutil
                shutil.copy2(str(src), str(output_dir / fname))
    else:
        raise ValueError("Either run_r2g=True or existing_graph_json must be provided.")
    result.raw_graph = raw_graph

    # --- 3. Graph orthogonalization ---
    print("[phase4] 3/13 graph alignment...")
    aligned_graph = normalize_graph(raw_graph)
    result.aligned_graph = aligned_graph
    _draw_graph_overlay(input_pil, aligned_graph, output_dir / "graph_overlay_aligned.png")

    aligned_edges = aligned_graph.get("aligned_edges", aligned_graph.get("edges", []))

    # --- 4. Segmentation inference ---
    print("[phase4] 4/13 segmentation inference...")
    from .segmentation_inference import run_segmentation, seg_mask_to_color_preview
    seg_class_map = run_segmentation(input_pil, seg_ckpt)
    result.seg_class_map = seg_class_map

    seg_preview_rgb = seg_mask_to_color_preview(seg_class_map)
    Image.fromarray(seg_preview_rgb).save(str(output_dir / "image_segmentation.png"))

    # --- 5. Component extraction ---
    print("[phase4] 5/13 component extraction...")
    seg_masks = split_class_masks(seg_class_map)
    result.seg_masks = seg_masks
    masks_for_components = {k: v for k, v in seg_masks.items() if k != "floor"}
    components, _ = extract_all_components(masks_for_components)
    result.components = components

    # --- 6. Scale inference ---
    print("[phase4] 6/13 scale inference...")
    scale_info = infer_scale_from_components(
        door_arc_components=components.get("door_arc", []),
        door_origin_components=components.get("door_origin", []),
        wall_components=components.get("wall", []),
        explicit_px_to_mm=explicit_px_to_mm,
    )
    result.scale_info = scale_info
    print(f"  scale: {scale_info.scale_status} px_to_mm={scale_info.px_to_mm}")

    # --- 7. Opening detection ---
    print("[phase4] 7/13 opening detection...")
    door_acc, door_rej = detect_door_candidates(
        door_arc_components=components.get("door_arc", []),
        aligned_graph_edges=aligned_edges,
    )
    win_acc, win_rej = detect_window_candidates(
        window_components=components.get("window", []),
    )
    result.door_candidates_accepted = door_acc
    result.door_candidates_rejected = door_rej
    result.window_candidates_accepted = win_acc
    result.window_candidates_rejected = win_rej
    print(f"  doors detected: {len(door_acc)} accepted, {len(door_rej)} rejected")
    print(f"  windows detected: {len(win_acc)} accepted, {len(win_rej)} rejected")

    # --- 8. Opening hosting ---
    print("[phase4] 8/13 opening hosting...")
    hosted, rejected_hosting = host_openings(
        door_candidates=door_acc,
        window_candidates=win_acc,
        aligned_graph_edges=aligned_edges,
        scale_info=scale_info,
        max_perp_dist_px=max_perp_dist_px,
    )
    hosted_doors = [h for h in hosted if h.opening_type == "door"]
    hosted_windows = [h for h in hosted if h.opening_type == "window"]

    all_rejected = (
        [RejectedOpening(
            opening_type="door", source_component_id=d.component_id,
            raw_points=d.raw_points, rejection_reason=d.rejection_reason,
            debug_confidence=d.confidence,
        ) for d in door_rej] +
        [RejectedOpening(
            opening_type="window", source_component_id=w.component_id,
            raw_points=w.raw_points, rejection_reason=w.rejection_reason,
            debug_confidence=w.confidence,
        ) for w in win_rej] +
        rejected_hosting
    )

    result.hosted_doors = hosted_doors    # pre-adjustment (for debug reference)
    result.hosted_windows = hosted_windows
    result.rejected_openings = all_rejected
    print(f"  hosted: {len(hosted_doors)} doors, {len(hosted_windows)} windows")
    print(f"  rejected total: {len(all_rejected)}")

    # --- 9-10. Wall interval trimming ---
    print("[phase4] 9/13 wall interval trimming...")
    trimmed = trim_wall_intervals(
        aligned_edges,
        hosted_doors + hosted_windows,
        px_to_mm=scale_info.px_to_mm,
    )
    result.trimmed_graph = trimmed
    if trimmed.last_resort_rejected:
        print(f"  last-resort rejected (no feasible interval): {len(trimmed.last_resort_rejected)}")
    print(f"  wall edges after trimming: {len(trimmed.wall_edges)}")

    # --- Part A: propagate adjusted intervals to final opening objects ---
    # After conflict resolution, opening gaps carry the adjusted endpoints.
    # Rebuild hosted_doors/windows so their snapped_points == trim endpoints.
    final_doors   = apply_adjusted_intervals_to_hosted_openings(trimmed, hosted_doors)
    final_windows = apply_adjusted_intervals_to_hosted_openings(trimmed, hosted_windows)

    # --- 11. Wall chain buffering (Part C: topology-snap pre-processing) ---
    print("[phase4] 10/13 wall buffering...")
    wall_geom = buffer_wall_chains(
        wall_edges=trimmed.wall_edges,
        scale_info=scale_info,
        preview_half_width_px=preview_half_width_px,
    )
    result.wall_geometry = wall_geom
    print(f"  wall chains: {wall_geom.chain_count}, thickness_mm={wall_geom.wall_thickness_mm}, "
          f"disconnected_endpoints={wall_geom.disconnected_endpoint_count}")

    # --- Part B: evidence-based door geometry ---
    door_arc_mask = seg_masks.get("door_arc")
    door_leaf_mask = seg_masks.get("door_leaf")
    door_arc_comps = {c.component_id: c for c in components.get("door_arc", [])}

    door_geometries: list[DoorGeometry] = []
    for door in final_doors:
        # Use the component-local mask when available (restrict to component bbox)
        comp = door_arc_comps.get(door.source_component_id)
        local_arc_mask = None
        if door_arc_mask is not None and comp is not None:
            # Crop mask to component bbox (±8px margin) to avoid evidence bleed from nearby doors
            x0, y0, x1, y1 = comp.bbox
            pad = 8
            h, w = door_arc_mask.shape[:2]
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
            local_arc_mask = np.zeros_like(door_arc_mask)
            local_arc_mask[cy0:cy1, cx0:cx1] = door_arc_mask[cy0:cy1, cx0:cx1]
        elif door_arc_mask is not None:
            local_arc_mask = door_arc_mask
        geom = compute_door_geometry(
            door,
            door_arc_mask=local_arc_mask,
            door_leaf_mask=door_leaf_mask,
        )
        door_geometries.append(geom)

    evidence_count = sum(1 for g in door_geometries if "evidence" in g.hinge_source)
    print(f"  door direction: {evidence_count}/{len(door_geometries)} from evidence, "
          f"{len(door_geometries) - evidence_count} fallback")

    # --- 12. Export final SVG ---
    print("[phase4] 11/13 exporting final SVG...")
    svg_content = build_final_svg(
        scale_info=scale_info,
        wall_geometry=wall_geom,
        hosted_doors=final_doors,
        hosted_windows=final_windows,
        door_geometries=door_geometries,
    )
    result.final_vector_svg = svg_content
    write_final_svg(svg_content, output_dir / "final_vector.svg")

    # --- 13. Export final JSON ---
    print("[phase4] 12/13 exporting final JSON...")
    elapsed = time.perf_counter() - t0
    metrics = {
        "elapsed_s": round(elapsed, 2),
        "r2g_nodes": len(raw_graph.get("nodes", [])),
        "r2g_edges": len(raw_graph.get("edges", [])),
        "aligned_edges": len(aligned_edges),
        "trimmed_wall_edges": len(trimmed.wall_edges),
        "opening_gaps": len(trimmed.opening_gaps),
        "hosted_doors": len(hosted_doors),
        "hosted_windows": len(hosted_windows),
        "rejected_total": len(all_rejected),
        "wall_chains": wall_geom.chain_count,
        "scale_status": scale_info.scale_status,
        "px_to_mm": scale_info.px_to_mm,
        "wall_thickness_mm": wall_geom.wall_thickness_mm,
        "scale_blocked": wall_geom.scale_blocked,
    }
    result.metrics = metrics

    final_json = build_final_vector_json(
        preprocessing_manifest=manifest,
        scale_info=scale_info,
        raw_graph=raw_graph,
        aligned_graph=aligned_graph,
        trimmed_graph=trimmed,
        hosted_doors=final_doors,
        hosted_windows=final_windows,
        rejected_openings=all_rejected,
        wall_geometry=wall_geom,
        metrics=metrics,
        door_geometries=door_geometries,
    )
    result.final_vector_json = final_json
    write_final_vector_json(final_json, output_dir / "final_vector.json")

    # --- Debug overlay ---
    print("[phase4] 13/13 writing debug overlay...")
    overlay = build_debug_overlay(
        input_pil=input_pil,
        aligned_edges=aligned_edges,
        door_candidates=door_acc,
        window_candidates=win_acc,
        hosted_doors=final_doors,
        hosted_windows=final_windows,
        rejected_openings=all_rejected,
        door_arc_components=components.get("door_arc", []),
        scale_info=scale_info,
        trimmed_graph=trimmed,
        door_geometries=door_geometries,
    )
    write_debug_overlay(overlay, output_dir / "image_debug_overlay.png")

    result.elapsed_s = elapsed
    print(f"\n[phase4] done in {elapsed:.1f}s -> {output_dir}")
    return result
