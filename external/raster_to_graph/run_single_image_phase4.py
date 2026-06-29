"""Apply phase4 raster-to-graph inference to a single arbitrary image.

Reuses all functions from run_inference_generous_phase4.py unchanged.
Outputs written alongside the input image.

Usage (from project root, floorplan-cad env):
    conda activate floorplan-cad
    python external/raster_to_graph/run_single_image_phase4.py \
        --image outputs/vectorization/phase4_raster2graph_generous_inference/sample_001/image.png \
        --sample-id sample_001
"""

from PIL import Image

import argparse
import copy
import cv2
import gc
import json
import math
import sys
import time
from collections import defaultdict
import numpy as np
import torch
from pathlib import Path

# Import everything from the phase4 pipeline script
sys.path.insert(0, str(Path(__file__).parent))

from args import get_args_parser
from models.build import build_model, build_postprocessor
from util.random_utils import set_random_seed
from util.data_utils import (
    initialize_tensors, random_keep, is_stop, draw_preds_on_tensors,
    edge_inside, point_inside, remove_points, remove_edge, get_edges_amount,
)
from util.metric_utils import get_results
from util.edges_utils import get_edges_alldirections
from util.misc import NestedTensor
from util.mean_std import mean, std

# Reuse all constants and functions from the main phase4 script by importing them
from run_inference_generous_phase4 import (
    CANVAS_SIZE, DARK_THRESH, STANDARDIZED_MARGIN,
    GENEROUS, MASK_RERUN, MERGE, FILTERS, SCORING,
    preprocess_crop512_margin20_truepad,
    normalize_pil,
    run_mc_inference,
    run_generous_multistart,
    merge_components,
    apply_light_post_merge_filter,
    compute_soft_scores,
    compute_candidate_score,
    make_svg_merged,
    make_overlay_normal,
    make_overlay_components,
    _find_components,
    _agg_filter_stats,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT   = PROJECT_ROOT / "checkpoints_Raster2Graph/checkpoint0299.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True, help="Path to input image")
    parser.add_argument("--sample-id", default=None,  help="Sample ID for output naming (default: image stem)")
    cli = parser.parse_args()

    img_path = Path(cli.image).resolve()
    if not img_path.exists():
        sys.exit(f"ERROR: image not found: {img_path}")

    sid     = cli.sample_id or img_path.stem
    out_dir = img_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available. This script requires a GPU.")

    gc.collect()
    torch.cuda.empty_cache()

    args_p = get_args_parser()
    args   = args_p.parse_args([])
    args.device = "cuda"
    device = torch.device("cuda")
    set_random_seed(args)

    model = build_model(args).to(device)
    postprocessor = build_postprocessor()
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Checkpoint epoch  : {ckpt['epoch']}")
    print(f"Input image       : {img_path}")
    print(f"Output dir        : {out_dir}")
    print(f"Sample ID         : {sid}")
    print()

    t0 = time.perf_counter()

    base_pil, preproc_metrics = preprocess_crop512_margin20_truepad(img_path)
    touches = preproc_metrics["content_touches_edge"]
    margins = preproc_metrics["final_canvas_margins_px"]
    if touches:
        print(f"  [WARN] content touches canvas edge after true-pad preprocessing")
    else:
        print(f"  margins: L={margins['left']} T={margins['top']} "
              f"R={margins['right']} B={margins['bottom']} px")

    gray_arr = np.asarray(base_pil.convert("L"))

    accepted, discarded = run_generous_multistart(
        model, postprocessor, base_pil, device, GENEROUS, MASK_RERUN
    )

    comp_node_count = sum(c["num_points"] for c in accepted)
    comp_edge_count = sum(c["num_edges"]  for c in accepted)

    merged_pts_raw, merged_edges_raw = merge_components(
        accepted,
        snap_tol  = MERGE["node_snap_tolerance_px"],
        inter_tol = MERGE["edge_intersection_tolerance_px"],
        col_tol   = MERGE["collinear_overlap_tolerance_px"],
    )
    merged_raw_node_count = len(merged_pts_raw)
    merged_raw_edge_count = len(merged_edges_raw)

    merged_pts, merged_edges, light_fstats = apply_light_post_merge_filter(
        merged_pts_raw, merged_edges_raw, gray_arr, FILTERS
    )

    merged_scores = compute_soft_scores(merged_pts, merged_edges, gray_arr, SCORING)
    elapsed = time.perf_counter() - t0
    empty   = len(merged_pts) == 0
    agg_fstats = _agg_filter_stats(accepted)

    # Write outputs alongside the input image
    base_pil.save(str(out_dir / "input.png"))

    graph_json = {
        "nodes": [[int(x), int(y)] for x, y in merged_pts],
        "edges": [[int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])] for p1, p2 in merged_edges],
    }
    (out_dir / "graph_pred.json").write_text(json.dumps(graph_json, indent=2), encoding="utf-8")
    (out_dir / "graph_pred.svg").write_text(make_svg_merged(merged_pts, merged_edges), encoding="utf-8")

    make_overlay_components(base_pil, accepted).save(str(out_dir / "graph_overlay_components.png"))
    make_overlay_normal(base_pil, merged_pts_raw, merged_edges_raw).save(str(out_dir / "graph_overlay_merged.png"))
    make_overlay_normal(base_pil, merged_pts, merged_edges).save(str(out_dir / "graph_overlay.png"))

    comp_out = {
        "sample_id": sid,
        "components_before_merge": [
            {
                "component_id":     c["component_id"],
                "source":           c["source"],
                "num_points":       c["num_points"],
                "num_edges":        c["num_edges"],
                "stop_code":        c["stop_code"],
                "candidate_score":  c["candidate_score"],
                "selected_attempt": c["selected_attempt"],
                "filter_stats":     c["filter_stats"],
                "scores": {k: v for k, v in c["scores"].items() if k != "per_edge_evidence"},
                "accepted": True,
            }
            for c in accepted
        ],
        "discarded_components": [
            {
                "component_id": c["component_id"],
                "source":       c["source"],
                "num_points":   c["num_points"],
                "num_edges":    c["num_edges"],
                "stop_code":    c["stop_code"],
                "accepted":     False,
            }
            for c in discarded
        ],
        "merged_graph": {
            "num_nodes_after_merge":        merged_raw_node_count,
            "num_edges_after_merge":        merged_raw_edge_count,
            "num_nodes_after_light_filter": len(merged_pts),
            "num_edges_after_light_filter": len(merged_edges),
            "light_post_merge_filter_stats": light_fstats,
        },
    }
    (out_dir / "components.json").write_text(json.dumps(comp_out, indent=2), encoding="utf-8")

    metrics = {
        "sample_id":              sid,
        "source_image":           str(img_path),
        "source_variant":         preproc_metrics["source_variant"],
        "standardized_margin":    preproc_metrics["standardized_margin"],
        "content_bbox_original":  preproc_metrics["content_bbox_original"],
        "content_bbox_after_preprocess": preproc_metrics["content_bbox_after_preprocess"],
        "final_canvas_margins_px": preproc_metrics["final_canvas_margins_px"],
        "content_touches_edge":   touches,
        "stage_counts": {
            "components_nodes": comp_node_count,
            "components_edges": comp_edge_count,
            "merged_nodes":     merged_raw_node_count,
            "merged_edges":     merged_raw_edge_count,
            "final_nodes":      len(merged_pts),
            "final_edges":      len(merged_edges),
        },
        "soft_scores": {
            "wall_evidence_alignment_score": merged_scores["wall_evidence_alignment_score"],
            "rectangle_cycle_count":         merged_scores["rectangle_cycle_count"],
            "rectangle_cycle_score":         merged_scores["rectangle_cycle_score"],
            "dangling_node_count":           merged_scores["dangling_node_count"],
            "dangling_penalty":              merged_scores["dangling_penalty"],
            "unsupported_edge_ratio":        merged_scores["unsupported_edge_ratio"],
            "small_component_count":         merged_scores["small_component_count"],
            "small_component_penalty":       merged_scores["small_component_penalty"],
            "candidate_score": compute_candidate_score(
                merged_scores, len(merged_edges), max(len(merged_edges), 1), SCORING
            ),
        },
        "components_before_merge": len(accepted),
        "components_after_merge":  len(_find_components(merged_pts, merged_edges)),
        "final_num_points":        len(merged_pts),
        "final_num_edges":         len(merged_edges),
        "empty":                   empty,
        "elapsed_s":               round(elapsed, 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if empty:
        note = "Empty — no significant component survived filters."
    elif len(accepted) > 1:
        note = (f"{len(accepted)} components. "
                f"Components: {comp_node_count}pts/{comp_edge_count}edges → "
                f"Merged: {merged_raw_node_count}pts/{merged_raw_edge_count}edges → "
                f"Final: {len(merged_pts)}pts/{len(merged_edges)}edges. "
                f"wall_ev={merged_scores['wall_evidence_alignment_score']:.2f}.")
    else:
        note = (f"1 component. "
                f"Components: {comp_node_count}pts/{comp_edge_count}edges → "
                f"Merged: {merged_raw_node_count}pts/{merged_raw_edge_count}edges → "
                f"Final: {len(merged_pts)}pts/{len(merged_edges)}edges. "
                f"wall_ev={merged_scores['wall_evidence_alignment_score']:.2f}.")
    (out_dir / "notes.txt").write_text(note + "\n", encoding="utf-8")

    status = "EMPTY" if empty else f"{len(merged_pts)}pt {len(merged_edges)}ed"
    print(f"\n  {status}  ({elapsed:.1f}s)")
    print(f"  wall_evidence = {merged_scores['wall_evidence_alignment_score']:.3f}")
    print(f"  cycles        = {merged_scores['rectangle_cycle_count']}")
    print(note.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
