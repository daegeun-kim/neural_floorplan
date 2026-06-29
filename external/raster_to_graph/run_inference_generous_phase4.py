"""Raster-to-Graph fast runtime-limited smoke test — Task 30.

Changes from Task 29:
  - preprocessing: true 20% white-canvas padding (crop512_margin20_truepad)
    - crops content bbox exactly, creates new white image with 20% padding,
      no clamped-bbox expansion that fails when content touches original edge
  - monte_times restored to 4, max_candidates_per_step restored to 40
  - node_snap_tolerance_px reduced 10 → 6 (less destructive geometry shift)
  - post-merge filter is now light: angle filter + dedup only
    - no tiny/one-edge/short-dangling deletion after merge
  - three overlays: graph_overlay_components.png, graph_overlay_merged.png, graph_overlay.png
  - metrics.json records stage-by-stage node/edge counts and preprocessing margins

Keeps from Task 29:
  - all hard filters before candidate reranking (per-MC-attempt)
  - soft scoring + candidate validity reranking
  - mask-and-rerun multi-start
  - merge-on-intersection

Does NOT modify model weights. Does NOT train.

Usage (from project root, floorplan-cad env):
    conda activate floorplan-cad
    python external/raster_to_graph/run_inference_generous_phase4.py [--max-samples N] [--no-cleanup]
"""

from PIL import Image

import argparse
import copy
import cv2
import gc
import json
import math
import shutil
import sys
import time
from collections import defaultdict
import numpy as np
import torch
from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_BASE  = PROJECT_ROOT / "outputs/vectorization/phase4_raster2graph_generous_inference"
CHECKPOINT   = PROJECT_ROOT / "checkpoints_Raster2Graph/checkpoint0299.pth"
MANIFEST     = PROJECT_ROOT / "data/raster2graph/preprocess_test_samples.json"
CANVAS_SIZE  = 512
DARK_THRESH  = 180
STANDARDIZED_MARGIN = 0.20

OLD_PHASE4_FOLDERS = [
    "outputs/vectorization/phase4_raster2graph_finetuning",
    "outputs/vectorization/phase4_raster2graph_preprocessing_test",
    "outputs/vectorization/phase4_raster2graph_permissive_inference",
    "outputs/vectorization/phase4_raster2graph_tuned_inference",
    "outputs/vectorization/phase4_raster2graph_recovery_inference",
    "outputs/vectorization/phase4_raster2graph_multistart_inference",
]

# Runtime settings (Task 30 — restored Task 29 spec values)
GENEROUS = {
    "first_step_threshold":    0.02,
    "later_step_threshold":    0.02,
    "first_step_force_best":   True,
    "edge_search_threshold":   50,
    "monte_times":             4,
    "max_candidates_per_step": 40,
}

MASK_RERUN = {
    "covered_node_radius_px":   24,
    "covered_edge_width_px":    30,
    "covered_mask_dilation_px": 20,
    "min_component_points":     3,
    "min_component_edges":      2,
    "max_new_starts":           2,
}

MERGE = {
    "node_snap_tolerance_px":          6,   # reduced from 10 (Task 30: less destructive geometry shift)
    "edge_intersection_tolerance_px":   8,
    "collinear_overlap_tolerance_px":   8,
}

# Hard filters (applied per MC attempt — unchanged from Task 29)
FILTERS = {
    "angle_tol_deg":                10,
    "min_component_points":          3,
    "min_component_edges":           2,
    "one_edge_min_length_px":       80,
    "one_edge_min_evidence":        0.65,
    "dangling_short_edge_max_px":   35,
    "dangling_edge_min_evidence":   0.45,
    "wall_evidence_band_px":        10,
    "dark_pixel_threshold":        180,
}

# Soft scoring
SCORING = {
    "wall_evidence_band_px":       10,
    "dark_pixel_threshold":       180,
    "min_cycle_area_px2":         400,
    "max_cycle_aspect_ratio":     8.0,
    "dangling_ratio_soft_limit":  0.35,
    "unsupported_edge_threshold": 0.35,
    "max_expected_cycles":         5.0,
    # Candidate score weights
    "w_wall_evidence":    3.0,
    "w_cycle_score":      2.0,
    "w_dangling_penalty": 1.5,
    "w_unsupported":      2.0,
    "w_small_comp":       1.0,
    "w_edge_count":       0.2,
}

COMPONENT_COLORS = [
    ((255, 0, 0),   (0, 0, 255)),
    ((0, 200, 0),   (0, 130, 0)),
    ((180, 0, 200), (130, 0, 150)),
    ((255, 140, 0), (200, 100, 0)),
    ((0, 200, 200), (0, 140, 140)),
]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _content_bbox(img: Image.Image):
    gray = np.asarray(img.convert("L"))
    dark = gray < DARK_THRESH
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return (0, 0, img.size[0], img.size[1])
    return (int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1))


def _to_512_canvas(img: Image.Image) -> Image.Image:
    sf = CANVAS_SIZE / max(img.size)
    nw, nh = int(img.size[0] * sf), int(img.size[1] * sf)
    scaled = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    canvas.paste(scaled, ((CANVAS_SIZE - nw) // 2, (CANVAS_SIZE - nh) // 2))
    return canvas


def preprocess_crop512_margin20_truepad(img_path: Path):
    """True-pad preprocessing: crop content, add 20% white margin, scale to 512.

    Steps:
      1. Detect dark-content bbox in original image
      2. Crop exactly to content bbox
      3. Create new white image with 20% padding on each side
      4. Scale padded image so long edge = 512px
      5. Center on 512x512 white canvas

    Returns (pil_image, preproc_metrics_dict).

    Unlike the old clamped-bbox expansion, this guarantees margin even when
    content touches the original image boundary.
    """
    img = Image.open(str(img_path)).convert("RGB")
    orig_bbox = _content_bbox(img)
    x0, y0, x1, y1 = orig_bbox

    content_w = max(x1 - x0, 1)
    content_h = max(y1 - y0, 1)

    # Crop exactly to content
    content_crop = img.crop((x0, y0, x1, y1))

    # Compute true padding as fraction of content size
    pad_x = max(int(content_w * STANDARDIZED_MARGIN), 1)
    pad_y = max(int(content_h * STANDARDIZED_MARGIN), 1)
    new_w = content_w + 2 * pad_x
    new_h = content_h + 2 * pad_y

    # Paste content onto white canvas with padding
    padded = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    padded.paste(content_crop, (pad_x, pad_y))

    # Scale to 512x512 canvas
    final = _to_512_canvas(padded)

    # Measure content bbox in final 512x512 canvas
    final_bbox = _content_bbox(final)
    fx0, fy0, fx1, fy1 = final_bbox

    touches = content_touches_edge(final)

    preproc_metrics = {
        "source_variant": "crop512_margin20_truepad",
        "standardized_margin": STANDARDIZED_MARGIN,
        "content_bbox_original": [int(x0), int(y0), int(x1), int(y1)],
        "content_bbox_after_preprocess": [int(fx0), int(fy0), int(fx1), int(fy1)],
        "final_canvas_margins_px": {
            "left":   int(fx0),
            "top":    int(fy0),
            "right":  int(CANVAS_SIZE - fx1),
            "bottom": int(CANVAS_SIZE - fy1),
        },
        "content_touches_edge": touches,
    }

    return final, preproc_metrics


def content_touches_edge(pil_img: Image.Image, dark_thresh: int = DARK_THRESH,
                          border_px: int = 3) -> bool:
    """Return True if any dark content pixel lies within border_px of the canvas edge."""
    gray = np.asarray(pil_img.convert("L"))
    H, W = gray.shape
    bp = border_px
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[:bp, :] = True
    border_mask[-bp:, :] = True
    border_mask[:, :bp] = True
    border_mask[:, -bp:] = True
    return bool((gray < dark_thresh)[border_mask].any())


def normalize_pil(pil_img: Image.Image) -> torch.Tensor:
    if pil_img.size != (CANVAS_SIZE, CANVAS_SIZE):
        pil_img = pil_img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS)
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t  = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mt = torch.tensor(mean, dtype=t.dtype).view(-1, 1, 1)
    st = torch.tensor(std,  dtype=t.dtype).view(-1, 1, 1)
    return (t - mt) / st


# ---------------------------------------------------------------------------
# Single autoregressive attempt
# ---------------------------------------------------------------------------

@torch.no_grad()
def _single_attempt(model, postprocessor, tensor, device, base):
    """One autoregressive pass. Returns (stop_code, preds, fallback_used, fallback_score)."""
    samples_batch = tensor.unsqueeze(0).to(device)
    mask = torch.zeros(1, CANVAS_SIZE, CANVAS_SIZE, dtype=torch.bool, device=device)
    tensors, _ = initialize_tensors(samples_batch)
    tensors_mc = copy.deepcopy(tensors)

    fst_thr    = base["first_step_threshold"]
    lat_thr    = base["later_step_threshold"]
    force_best = base.get("first_step_force_best", True)
    t          = base["edge_search_threshold"]
    max_cands  = base["max_candidates_per_step"]

    preds   = []
    fb_used = False
    fb_score = None

    for iter_time in range(9999999):
        threshold = fst_thr if iter_time == 0 else lat_thr

        samples_iter = NestedTensor(tensors_mc, mask)
        outputs      = model(samples_iter)
        target_sizes = torch.tensor([[CANVAS_SIZE, CANVAS_SIZE]], device=device)
        results      = postprocessor(outputs, target_sizes)[0]

        valid_edge  = torch.where(results["edges"] != 0)[0]
        valid_score = torch.where(results["scores"] > threshold)[0]
        valid_idx   = torch.tensor(
            list(set(valid_edge.tolist()) & set(valid_score.tolist())),
            dtype=valid_edge.dtype, device=device,
        )
        if len(valid_idx) > 0:
            order     = results["scores"][valid_idx].argsort(descending=True)
            valid_idx = valid_idx[order[:max_cands]]

        def _mk(vi):
            v = {k: results[k][vi] for k in (
                "scores", "points", "last_edges", "this_edges", "edges",
                "semantic_left_up", "semantic_right_up",
                "semantic_right_down", "semantic_left_down",
            )}
            v["size"] = torch.tensor([CANVAS_SIZE, CANVAS_SIZE], device=device)
            return v

        this_preds = [_mk(vi) for vi in valid_idx]

        if iter_time == 0 and not this_preds and force_best:
            ci = torch.where((results["edges"] != 0) & (results["edges"] != 16))[0]
            if len(ci) == 0:
                ci = torch.where(results["edges"] != 0)[0]
            if len(ci) > 0:
                bvi = ci[results["scores"][ci].argmax()]
                this_preds = [_mk(bvi)]
                fb_used  = True
                fb_score = round(float(results["scores"][bvi].item()), 4)

        this_preds = random_keep(this_preds) if len(this_preds) > 1 else this_preds

        sc = is_stop(this_preds)
        if sc in (1, 2):
            return sc, preds, fb_used, fb_score

        last_edges = []
        if preds:
            all_given = [p for (pts_, le, te) in preds for p in pts_]
            need_remove = []
            for tp in this_preds:
                pt  = tp["points"].tolist()
                bds = (pt[1]-t, pt[0]-t, pt[1]+t, pt[0]+t)
                dirs = get_edges_alldirections(tp["last_edges"].item())
                cand_fns = [
                    lambda p, bds=bds, pt=pt: bds[1]<=p["points"].tolist()[0]<=bds[3] and p["points"].tolist()[1]<pt[1],
                    lambda p, bds=bds, pt=pt: bds[0]<=p["points"].tolist()[1]<=bds[2] and p["points"].tolist()[0]<pt[0],
                    lambda p, bds=bds, pt=pt: bds[1]<=p["points"].tolist()[0]<=bds[3] and p["points"].tolist()[1]>pt[1],
                    lambda p, bds=bds, pt=pt: bds[0]<=p["points"].tolist()[1]<=bds[2] and p["points"].tolist()[0]>pt[0],
                ]
                dist_fns = [
                    lambda p, pt=pt: pt[1]-p["points"].tolist()[1],
                    lambda p, pt=pt: pt[0]-p["points"].tolist()[0],
                    lambda p, pt=pt: p["points"].tolist()[1]-pt[1],
                    lambda p, pt=pt: p["points"].tolist()[0]-pt[0],
                ]
                skip = False
                for d in range(4):
                    if not int(dirs[d]): continue
                    cands = [p for p in all_given if cand_fns[d](p)]
                    if not cands: skip = True; break
                    nearest = min(cands, key=dist_fns[d])
                    if not (edge_inside((nearest, tp), last_edges) or edge_inside((tp, nearest), last_edges)):
                        last_edges.append((tp, nearest))
                if skip:
                    need_remove.append(tp)
            this_preds = remove_points(need_remove, this_preds)
            for p1, p2 in copy.deepcopy(last_edges):
                if point_inside(p1, need_remove) or point_inside(p2, need_remove):
                    last_edges = remove_edge((p1, p2), last_edges)

        sc = is_stop(this_preds)
        if sc in (1, 2):
            return sc, preds, fb_used, fb_score

        this_edges = []
        if len(this_preds) > 1:
            need_remove2 = []
            opp = [2, 3, 0, 1]
            for tp in this_preds:
                pt  = tp["points"].tolist()
                bds = (pt[1]-t, pt[0]-t, pt[1]+t, pt[0]+t)
                dirs = get_edges_alldirections(tp["this_edges"].item())
                cand_fns = [
                    lambda p, bds=bds, pt=pt: bds[1]<=p["points"].tolist()[0]<=bds[3] and p["points"].tolist()[1]<pt[1],
                    lambda p, bds=bds, pt=pt: bds[0]<=p["points"].tolist()[1]<=bds[2] and p["points"].tolist()[0]<pt[0],
                    lambda p, bds=bds, pt=pt: bds[1]<=p["points"].tolist()[0]<=bds[3] and p["points"].tolist()[1]>pt[1],
                    lambda p, bds=bds, pt=pt: bds[0]<=p["points"].tolist()[1]<=bds[2] and p["points"].tolist()[0]>pt[0],
                ]
                dist_fns = [
                    lambda p, pt=pt: pt[1]-p["points"].tolist()[1],
                    lambda p, pt=pt: pt[0]-p["points"].tolist()[0],
                    lambda p, pt=pt: p["points"].tolist()[1]-pt[1],
                    lambda p, pt=pt: p["points"].tolist()[0]-pt[0],
                ]
                skip = False
                for d in range(4):
                    if not int(dirs[d]): continue
                    cands = [p for p in this_preds if cand_fns[d](p)]
                    if not cands: skip = True; break
                    nearest = min(cands, key=dist_fns[d])
                    if not int(get_edges_alldirections(nearest["this_edges"].item())[opp[d]]):
                        skip = True; break
                    if not (edge_inside((nearest, tp), this_edges) or edge_inside((tp, nearest), this_edges)):
                        this_edges.append((tp, nearest))
                if skip:
                    need_remove2.append(tp)
            this_preds = remove_points(need_remove2, this_preds)
            for p1, p2 in copy.deepcopy(last_edges):
                if point_inside(p1, need_remove2) or point_inside(p2, need_remove2):
                    last_edges = remove_edge((p1, p2), last_edges)
            for p1, p2 in copy.deepcopy(this_edges):
                if point_inside(p1, need_remove2) or point_inside(p2, need_remove2):
                    this_edges = remove_edge((p1, p2), this_edges)

        sc = is_stop(this_preds)
        if sc in (1, 2):
            return sc, preds, fb_used, fb_score

        preds.append((this_preds, last_edges, this_edges))
        tensors_mc, _ = draw_preds_on_tensors(preds, tensors_mc)

        if iter_time > 80:
            return 1, preds, fb_used, fb_score

    return 0, [], fb_used, fb_score


# ---------------------------------------------------------------------------
# Hard filters (used per MC attempt — all four filters active)
# ---------------------------------------------------------------------------

def _is_orthogonal(p1, p2, tol_deg=10):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return False
    angle = math.degrees(math.atan2(abs(dy), abs(dx)))  # 0 = horizontal, 90 = vertical
    return angle <= tol_deg or angle >= (90 - tol_deg)


def _edge_length(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def _edge_wall_evidence(p1, p2, gray_arr, band_px=10, dark_thresh=180):
    """Fraction of pixels near this edge that are wall-dark."""
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    x2, y2 = int(round(p2[0])), int(round(p2[1]))
    H, W = gray_arr.shape
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return 0.0

    n_samples = max(int(length / 3), 2)
    dark_count = 0
    total_count = 0
    for i in range(n_samples + 1):
        t = i / n_samples
        cx = int(round(x1 + t * dx))
        cy = int(round(y1 + t * dy))
        xlo = max(0, cx - band_px)
        xhi = min(W, cx + band_px + 1)
        ylo = max(0, cy - band_px)
        yhi = min(H, cy + band_px + 1)
        patch = gray_arr[ylo:yhi, xlo:xhi]
        total_count += patch.size
        dark_count  += int((patch < dark_thresh).sum())

    return dark_count / max(total_count, 1)


def _find_components(pts, edges):
    """Return list of (comp_pts, comp_edges) for each connected component."""
    adj = defaultdict(set)
    all_pts = set(pts)
    for p1, p2 in edges:
        adj[p1].add(p2)
        adj[p2].add(p1)
        all_pts.add(p1)
        all_pts.add(p2)

    visited = set()
    components = []
    for start in all_pts:
        if start in visited:
            continue
        comp = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            for nb in adj[node]:
                if nb not in visited:
                    stack.append(nb)
        comp_edges = [(p1, p2) for p1, p2 in edges if p1 in comp and p2 in comp]
        components.append((list(comp), comp_edges))
    return components


def apply_hard_filters(pts, edges, gray_arr, cfg):
    """Full hard filter for per-MC-attempt filtering. All four filters active."""
    stats = {
        "edges_removed_angle_filter":         0,
        "components_removed_tiny":            0,
        "points_removed_tiny_components":     0,
        "edges_removed_tiny_components":      0,
        "one_edge_components_removed":        0,
        "one_edge_components_kept_by_evidence": 0,
        "dangling_edges_removed":             0,
    }

    # 1. Angle filter
    kept = []
    for e in edges:
        if _is_orthogonal(e[0], e[1], cfg["angle_tol_deg"]):
            kept.append(e)
        else:
            stats["edges_removed_angle_filter"] += 1
    edges = kept
    touched = set(p for e in edges for p in e)
    pts = [p for p in pts if p in touched]

    # 2+3. Tiny and one-edge component filters
    comps = _find_components(pts, edges)
    final_comps = []
    bpx = cfg["wall_evidence_band_px"]
    dt  = cfg["dark_pixel_threshold"]

    for comp_pts, comp_edges in comps:
        n_pts = len(comp_pts)
        n_ed  = len(comp_edges)

        if n_pts < cfg["min_component_points"] or n_ed < cfg["min_component_edges"]:
            stats["components_removed_tiny"] += 1
            stats["points_removed_tiny_components"]  += n_pts
            stats["edges_removed_tiny_components"]   += n_ed
            continue

        if n_ed == 1:
            e = comp_edges[0]
            length   = _edge_length(e[0], e[1])
            evidence = _edge_wall_evidence(e[0], e[1], gray_arr, bpx, dt)
            if length >= cfg["one_edge_min_length_px"] and evidence >= cfg["one_edge_min_evidence"]:
                final_comps.append((comp_pts, comp_edges))
                stats["one_edge_components_kept_by_evidence"] += 1
            else:
                stats["one_edge_components_removed"] += 1
            continue

        final_comps.append((comp_pts, comp_edges))

    pts_set  = set()
    all_pts  = []
    all_edges = []
    for cp, ce in final_comps:
        for p in cp:
            if p not in pts_set:
                all_pts.append(p)
                pts_set.add(p)
        all_edges.extend(ce)

    # 4. Short dangling edge filter
    degree = defaultdict(int)
    for p1, p2 in all_edges:
        degree[p1] += 1
        degree[p2] += 1

    to_rm = set()
    for e in all_edges:
        p1, p2 = e
        if degree[p1] == 1 or degree[p2] == 1:
            if _edge_length(p1, p2) <= cfg["dangling_short_edge_max_px"]:
                ev = _edge_wall_evidence(p1, p2, gray_arr, bpx, dt)
                if ev < cfg["dangling_edge_min_evidence"]:
                    to_rm.add((min(p1, p2), max(p1, p2)))

    stats["dangling_edges_removed"] = len(to_rm)
    all_edges = [e for e in all_edges if (min(e[0], e[1]), max(e[0], e[1])) not in to_rm]

    touched2 = set(p for e in all_edges for p in e)
    all_pts  = [p for p in all_pts if p in touched2]

    return all_pts, all_edges, stats


def apply_light_post_merge_filter(pts, edges, gray_arr, cfg):
    """Light post-merge filter: angle filter and dedup only.

    Does NOT apply tiny component, one-edge component, or short dangling
    edge deletion. Those filters run before candidate reranking (per MC attempt).
    After merge, edge splits can create short fragments that are still valid
    wall segments — removing them is too destructive.
    """
    stats = {
        "edges_removed_angle_filter": 0,
        "duplicate_edges_removed":    0,
        "self_loop_edges_removed":    0,
    }

    # 1. Angle filter (only severe violations)
    kept = []
    for e in edges:
        if _is_orthogonal(e[0], e[1], cfg["angle_tol_deg"]):
            kept.append(e)
        else:
            stats["edges_removed_angle_filter"] += 1
    edges = kept

    # 2. Remove self-loops
    no_loops = []
    for p1, p2 in edges:
        if p1 != p2:
            no_loops.append((p1, p2))
        else:
            stats["self_loop_edges_removed"] += 1
    edges = no_loops

    # 3. Remove exact duplicate edges
    seen = set()
    dedup = []
    for e in edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in seen:
            seen.add(key)
            dedup.append(e)
        else:
            stats["duplicate_edges_removed"] += 1
    edges = dedup

    touched = set(p for e in edges for p in e)
    pts = [p for p in pts if p in touched]

    return pts, edges, stats


# ---------------------------------------------------------------------------
# Soft scoring
# ---------------------------------------------------------------------------

def _compute_wall_evidence_all_edges(edges, gray_arr, band_px=10, dark_thresh=180):
    scores = []
    for p1, p2 in edges:
        scores.append(_edge_wall_evidence(p1, p2, gray_arr, band_px, dark_thresh))
    return scores


def _count_valid_cycles(pts, edges, min_area=400, max_aspect=8.0):
    if len(edges) < 3:
        return 0

    adj = defaultdict(list)
    for p1, p2 in edges:
        adj[p1].append(p2)
        adj[p2].append(p1)

    high_deg = [p for p in adj if len(adj[p]) >= 2]
    if not high_deg:
        return 0

    seen  = set()
    count = [0]
    calls = [0]

    def shoelace(verts):
        n = len(verts)
        a = 0
        for i in range(n):
            j = (i + 1) % n
            a += verts[i][0] * verts[j][1] - verts[j][0] * verts[i][1]
        return abs(a) / 2

    def bbox_aspect(verts):
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        w = max(xs) - min(xs) + 1
        h = max(ys) - min(ys) + 1
        return max(w, h) / max(min(w, h), 1)

    def dfs(start, cur, path, in_path):
        calls[0] += 1
        if calls[0] > 6000 or len(path) > 10:
            return
        for nxt in adj[cur]:
            if nxt == start and len(path) >= 4:
                key = frozenset(path)
                if key not in seen:
                    seen.add(key)
                    if shoelace(path) >= min_area and bbox_aspect(path) <= max_aspect:
                        count[0] += 1
            elif nxt not in in_path:
                path.append(nxt)
                in_path.add(nxt)
                dfs(start, nxt, path, in_path)
                path.pop()
                in_path.remove(nxt)

    for pt in sorted(high_deg):
        if calls[0] > 6000:
            break
        dfs(pt, pt, [pt], {pt})

    return count[0]


def compute_soft_scores(pts, edges, gray_arr, cfg):
    if not pts or not edges:
        return {
            "wall_evidence_alignment_score": 0.0,
            "unsupported_edge_ratio":        0.0,
            "rectangle_cycle_count":         0,
            "rectangle_cycle_score":         0.0,
            "dangling_node_count":           0,
            "dangling_node_ratio":           0.0,
            "dangling_penalty":              0.0,
            "small_component_count":         0,
            "small_component_penalty":       0.0,
            "per_edge_evidence":             [],
        }

    bpx = cfg["wall_evidence_band_px"]
    dt  = cfg["dark_pixel_threshold"]

    ev_scores = _compute_wall_evidence_all_edges(edges, gray_arr, bpx, dt)
    wall_ev   = float(np.mean(ev_scores)) if ev_scores else 0.0

    unsup_thr   = cfg["unsupported_edge_threshold"]
    unsup_ratio = sum(1 for s in ev_scores if s < unsup_thr) / max(len(ev_scores), 1)

    cycle_count = _count_valid_cycles(pts, edges, cfg["min_cycle_area_px2"], cfg["max_cycle_aspect_ratio"])
    cycle_score = min(cycle_count / max(cfg["max_expected_cycles"], 1), 1.0)

    degree = defaultdict(int)
    for p1, p2 in edges:
        degree[p1] += 1
        degree[p2] += 1
    dangling_n     = sum(1 for d in degree.values() if d == 1)
    dangling_ratio = dangling_n / max(len(pts), 1)
    soft_lim       = cfg["dangling_ratio_soft_limit"]
    dangling_pen   = max(0.0, (dangling_ratio - soft_lim) / max(1.0 - soft_lim, 0.01))
    dangling_pen   = min(dangling_pen, 1.0)

    comps     = _find_components(pts, edges)
    n_comps   = len(comps)
    small_pen = min(max(n_comps - 1, 0) / 4.0, 1.0)

    return {
        "wall_evidence_alignment_score": round(wall_ev, 4),
        "unsupported_edge_ratio":        round(unsup_ratio, 4),
        "rectangle_cycle_count":         cycle_count,
        "rectangle_cycle_score":         round(cycle_score, 4),
        "dangling_node_count":           dangling_n,
        "dangling_node_ratio":           round(dangling_ratio, 4),
        "dangling_penalty":              round(dangling_pen, 4),
        "small_component_count":         n_comps,
        "small_component_penalty":       round(small_pen, 4),
        "per_edge_evidence":             [round(s, 4) for s in ev_scores],
    }


def compute_candidate_score(scores, edge_count, max_edges, cfg):
    norm_edges = edge_count / max(max_edges, 1)
    s = (cfg["w_wall_evidence"]    * scores["wall_evidence_alignment_score"]
       + cfg["w_cycle_score"]      * scores["rectangle_cycle_score"]
       - cfg["w_dangling_penalty"] * scores["dangling_penalty"]
       - cfg["w_unsupported"]      * scores["unsupported_edge_ratio"]
       - cfg["w_small_comp"]       * scores["small_component_penalty"]
       + cfg["w_edge_count"]       * norm_edges)
    return round(s, 4)


# ---------------------------------------------------------------------------
# Monte Carlo wrapper with scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_mc_inference(model, postprocessor, tensor, device, base, gray_arr):
    """Run monte_times attempts; pick best by graph validity score."""
    monte_times = base["monte_times"]
    attempts = []

    for attempt_i in range(monte_times):
        sc, preds, fb, fb_score = _single_attempt(model, postprocessor, tensor, device, base)
        ec = get_edges_amount(preds)
        if not preds:
            raw_pts, raw_edges = [], []
        else:
            raw_pts, raw_edges = get_results((sc, ec, preds))

        f_pts, f_edges, fstats = apply_hard_filters(raw_pts, raw_edges, gray_arr, FILTERS)
        s = compute_soft_scores(f_pts, f_edges, gray_arr, SCORING)

        attempts.append({
            "attempt_i": attempt_i,
            "stop_code": sc,
            "pts":    f_pts,
            "edges":  f_edges,
            "filter_stats": fstats,
            "scores": s,
            "raw_edge_count": len(raw_edges),
            "raw_node_count": len(raw_pts),
            "fallback_used":  fb,
            "fallback_score": fb_score,
        })

    max_ed = max((len(a["edges"]) for a in attempts), default=1)
    for a in attempts:
        a["candidate_score"] = compute_candidate_score(
            a["scores"], len(a["edges"]), max_ed, SCORING
        )

    best = max(attempts, key=lambda a: (a["candidate_score"], len(a["edges"])))

    attempt_summaries = [
        {
            "attempt": a["attempt_i"],
            "stop_code": a["stop_code"],
            "raw_edges": a["raw_edge_count"],
            "filtered_edges": len(a["edges"]),
            "filtered_nodes": len(a["pts"]),
            "candidate_score": a["candidate_score"],
            "wall_evidence": a["scores"]["wall_evidence_alignment_score"],
            "cycle_score":   a["scores"]["rectangle_cycle_score"],
            "dangling_pen":  a["scores"]["dangling_penalty"],
        }
        for a in attempts
    ]

    return (
        best["pts"], best["edges"],
        best["stop_code"], best["filter_stats"], best["scores"],
        best["candidate_score"], best["attempt_i"],
        attempt_summaries,
    )


# ---------------------------------------------------------------------------
# Mask-and-rerun multi-start
# ---------------------------------------------------------------------------

def build_covered_mask(pts, edges, node_r=24, edge_w=30, dilation=20):
    mask = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    for (x1, y1), (x2, y2) in edges:
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, edge_w)
    for x, y in pts:
        cv2.circle(mask, (int(x), int(y)), node_r, 255, -1)
    if dilation > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation*2+1, dilation*2+1))
        mask = cv2.dilate(mask, k)
    return mask > 0


def apply_white_suppression(pil_img: Image.Image, covered_mask) -> Image.Image:
    arr = np.asarray(pil_img.convert("RGB")).copy()
    arr[covered_mask] = 255
    return Image.fromarray(arr)


@torch.no_grad()
def run_generous_multistart(model, postprocessor, base_pil: Image.Image, device, base, mr):
    """Mask-and-rerun multi-start with scoring. Returns (accepted, discarded)."""
    min_pts    = mr["min_component_points"]
    min_edges  = mr["min_component_edges"]
    max_starts = mr["max_new_starts"]
    node_r     = mr["covered_node_radius_px"]
    edge_w     = mr["covered_edge_width_px"]
    dilation   = mr["covered_mask_dilation_px"]

    accepted    = []
    discarded   = []
    all_pts     = []
    all_edges   = []
    current_pil = base_pil
    gray_arr    = np.asarray(base_pil.convert("L"))

    for comp_idx in range(max_starts + 1):
        tensor = normalize_pil(current_pil)
        (pts, edges, sc, fstats, scores,
         cand_score, sel_attempt, attempt_summ) = run_mc_inference(
            model, postprocessor, tensor, device, base, gray_arr
        )

        significant = len(pts) >= min_pts and len(edges) >= min_edges

        comp_info = {
            "component_id":     comp_idx,
            "source":           "initial" if comp_idx == 0 else "mask_rerun",
            "num_points":       len(pts),
            "num_edges":        len(edges),
            "stop_code":        sc,
            "filter_stats":     fstats,
            "scores":           scores,
            "candidate_score":  cand_score,
            "selected_attempt": sel_attempt,
            "attempt_summaries": attempt_summ,
            "pts":   pts,
            "edges": edges,
        }

        if significant:
            accepted.append(comp_info)
            all_pts.extend(pts)
            all_edges.extend(edges)
        else:
            comp_info["accepted"] = False
            discarded.append(comp_info)
            break

        if comp_idx == max_starts:
            break

        covered = build_covered_mask(all_pts, all_edges, node_r, edge_w, dilation)
        if covered.mean() > 0.95:
            break
        current_pil = apply_white_suppression(base_pil, covered)

    return accepted, discarded


# ---------------------------------------------------------------------------
# Graph merge: snap + intersections + collinear
# ---------------------------------------------------------------------------

def _classify_edge(p1, p2, tol):
    if abs(p1[1] - p2[1]) <= tol:
        return 'H'
    if abs(p1[0] - p2[0]) <= tol:
        return 'V'
    return 'D'


def _snap_nodes(pts, edges, tol):
    if not pts:
        return [], []
    seen = set()
    dedup = []
    for p in pts:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    pts = dedup
    n = len(pts)

    parent = list(range(n))

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(xs[i] - xs[j]) <= tol and abs(ys[i] - ys[j]) <= tol:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    old_to_new = {}
    new_pts = []
    for r in sorted(groups):
        members = groups[r]
        cx = round(sum(xs[m] for m in members) / len(members))
        cy = round(sum(ys[m] for m in members) / len(members))
        new_idx = len(new_pts)
        new_pts.append((cx, cy))
        for m in members:
            old_to_new[m] = new_idx

    coord_to_old = {pts[i]: i for i in range(n)}
    new_edges_set = set()
    for (x1, y1), (x2, y2) in edges:
        i1 = coord_to_old.get((x1, y1))
        i2 = coord_to_old.get((x2, y2))
        if i1 is None or i2 is None:
            continue
        n1, n2 = old_to_new[i1], old_to_new[i2]
        if n1 != n2:
            new_edges_set.add((min(n1, n2), max(n1, n2)))

    new_edges = [(new_pts[a], new_pts[b]) for a, b in sorted(new_edges_set)]
    return new_pts, new_edges


def _insert_intersections(pts, edges, tol):
    h_edges = [(p1, p2) for p1, p2 in edges if _classify_edge(p1, p2, tol) == 'H']
    v_edges = [(p1, p2) for p1, p2 in edges if _classify_edge(p1, p2, tol) == 'V']
    other   = [(p1, p2) for p1, p2 in edges if _classify_edge(p1, p2, tol) == 'D']

    crossings = []
    for (hx0, hy0), (hx1, hy1) in h_edges:
        hy  = round((hy0 + hy1) / 2)
        xlo = min(hx0, hx1)
        xhi = max(hx0, hx1)
        for (vx0, vy0), (vx1, vy1) in v_edges:
            vx  = round((vx0 + vx1) / 2)
            ylo = min(vy0, vy1)
            yhi = max(vy0, vy1)
            if xlo - tol <= vx <= xhi + tol and ylo - tol <= hy <= yhi + tol:
                crossings.append((vx, hy))

    pts_set = set(pts)
    for pt in crossings:
        pts_set.add(pt)

    result_edges = list(other)

    for (hx0, hy0), (hx1, hy1) in h_edges:
        hy  = round((hy0 + hy1) / 2)
        xlo = min(hx0, hx1)
        xhi = max(hx0, hx1)
        split_xs = sorted(set(
            [xlo, xhi] +
            [vx for vx, vy in crossings if vy == hy and xlo - tol <= vx <= xhi + tol]
        ))
        for i in range(len(split_xs) - 1):
            p1, p2 = (split_xs[i], hy), (split_xs[i + 1], hy)
            if p1 != p2:
                result_edges.append((p1, p2))
                pts_set.add(p1); pts_set.add(p2)

    for (vx0, vy0), (vx1, vy1) in v_edges:
        vx  = round((vx0 + vx1) / 2)
        ylo = min(vy0, vy1)
        yhi = max(vy0, vy1)
        split_ys = sorted(set(
            [ylo, yhi] +
            [vy for vx_, vy in crossings if vx_ == vx and ylo - tol <= vy <= yhi + tol]
        ))
        for i in range(len(split_ys) - 1):
            p1, p2 = (vx, split_ys[i]), (vx, split_ys[i + 1])
            if p1 != p2:
                result_edges.append((p1, p2))
                pts_set.add(p1); pts_set.add(p2)

    seen = set()
    final_edges = []
    for p1, p2 in result_edges:
        key = (min(p1, p2), max(p1, p2))
        if key not in seen:
            seen.add(key)
            final_edges.append(key)

    return list(pts_set), [(a, b) for a, b in final_edges]


def _merge_collinear(pts, edges, tol):
    h_list, v_list, other = [], [], []

    for p1, p2 in edges:
        cls = _classify_edge(p1, p2, tol)
        if cls == 'H':
            hy = round((p1[1] + p2[1]) / 2)
            h_list.append((hy, min(p1[0], p2[0]), max(p1[0], p2[0])))
        elif cls == 'V':
            vx = round((p1[0] + p2[0]) / 2)
            v_list.append((vx, min(p1[1], p2[1]), max(p1[1], p2[1])))
        else:
            other.append((p1, p2))

    def _group_merge(items, key_i, lo_i, hi_i):
        groups = {}
        for item in items:
            k = item[key_i]
            placed = False
            for gk in list(groups):
                if abs(k - gk) <= tol:
                    groups[gk].append((item[lo_i], item[hi_i]))
                    placed = True
                    break
            if not placed:
                groups[k] = [(item[lo_i], item[hi_i])]
        out = []
        for gk, intervals in groups.items():
            intervals = sorted(intervals)
            merged = [list(intervals[0])]
            for lo, hi in intervals[1:]:
                if lo <= merged[-1][1] + tol:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            for lo, hi in merged:
                out.append((gk, lo, hi))
        return out

    merged_h = _group_merge(h_list, 0, 1, 2)
    merged_v = _group_merge(v_list, 0, 1, 2)

    result_edges = list(other)
    all_pts_set  = set(pts)

    for hy, xlo, xhi in merged_h:
        p1, p2 = (xlo, hy), (xhi, hy)
        if p1 != p2:
            result_edges.append((p1, p2))
            all_pts_set.add(p1); all_pts_set.add(p2)

    for vx, ylo, yhi in merged_v:
        p1, p2 = (vx, ylo), (vx, yhi)
        if p1 != p2:
            result_edges.append((p1, p2))
            all_pts_set.add(p1); all_pts_set.add(p2)

    seen = set()
    final_edges = []
    for p1, p2 in result_edges:
        key = (min(p1, p2), max(p1, p2))
        if key not in seen:
            seen.add(key)
            final_edges.append(key)

    return list(all_pts_set), [(a, b) for a, b in final_edges]


def merge_components(components, snap_tol=6, inter_tol=8, col_tol=8):
    if not components:
        return [], []
    all_pts, all_edges = [], []
    for comp in components:
        all_pts.extend(comp["pts"])
        all_edges.extend(comp["edges"])
    if not all_pts:
        return [], []

    seen = set()
    dedup_edges = []
    for e in all_edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in seen:
            seen.add(key)
            dedup_edges.append(e)

    pts, edges = _snap_nodes(all_pts, dedup_edges, snap_tol)
    pts, edges = _insert_intersections(pts, edges, inter_tol)
    pts, edges = _merge_collinear(pts, edges, col_tol)
    pts, edges = _snap_nodes(pts, edges, snap_tol)
    edges = [(p1, p2) for p1, p2 in edges if p1 != p2]
    return pts, edges


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_svg_merged(pts, edges, size=CANVAS_SIZE):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
        f'<rect width="{size}" height="{size}" fill="white"/>',
    ]
    for p1, p2 in edges:
        x1, y1 = p1; x2, y2 = p2
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="blue" stroke-width="2"/>')
    for x, y in pts:
        parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="red"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def make_overlay_normal(input_pil, pts, edges):
    arr = np.asarray(input_pil.convert("RGB")).copy()
    for p1, p2 in edges:
        cv2.line(arr, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2)
    for x, y in pts:
        cv2.circle(arr, (int(x), int(y)), 4, (255, 0, 0), -1)
    return Image.fromarray(arr)


def make_overlay_components(input_pil, components):
    arr = np.asarray(input_pil.convert("RGB")).copy()
    for ci, comp in enumerate(components):
        nc, ec = COMPONENT_COLORS[ci % len(COMPONENT_COLORS)]
        for (x1, y1), (x2, y2) in comp["edges"]:
            cv2.line(arr, (int(x1), int(y1)), (int(x2), int(y2)), ec, 2)
        for x, y in comp["pts"]:
            cv2.circle(arr, (int(x), int(y)), 5, nc, -1)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gt_counts(wall_graph_rel):
    p = PROJECT_ROOT / wall_graph_rel
    if not p.exists():
        return 0, 0
    try:
        d = json.loads(p.read_text())
        return len(d.get("nodes", [])), len(d.get("edges", []))
    except Exception:
        return 0, 0


def node_count_bin(n):
    if n <= 50:    return "10-50"
    elif n <= 80:  return "51-80"
    elif n <= 120: return "81-120"
    else:          return "120+"


def cleanup_old_folders():
    for rel in OLD_PHASE4_FOLDERS:
        p = PROJECT_ROOT / rel
        if p.exists():
            shutil.rmtree(str(p))
            print(f"  Removed: {p}")
        else:
            print(f"  Skip (not found): {p}")


def _agg_filter_stats(components):
    """Sum per-MC-attempt filter stats across all accepted components."""
    keys = [
        "edges_removed_angle_filter", "components_removed_tiny",
        "points_removed_tiny_components", "edges_removed_tiny_components",
        "one_edge_components_removed", "one_edge_components_kept_by_evidence",
        "dangling_edges_removed",
    ]
    total = {k: 0 for k in keys}
    for c in components:
        for k in keys:
            total[k] += c["filter_stats"].get(k, 0)
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="True-pad 20% preprocessing + light post-merge filter — Task 30")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Skip removing old Phase 4 output folders")
    cli = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available. This script requires a GPU.")

    if not cli.no_cleanup:
        print("=== Removing old Phase 4 output folders ===")
        cleanup_old_folders()
        print()

    if OUTPUT_BASE.exists():
        shutil.rmtree(str(OUTPUT_BASE))
        print(f"Cleared existing Phase 4 output folder: {OUTPUT_BASE}")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

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
    print(f"Output base       : {OUTPUT_BASE}")
    print(f"Source variant    : crop512_margin20_truepad (20% true-pad)")
    print(f"Thresholds        : {GENEROUS['first_step_threshold']}/{GENEROUS['later_step_threshold']}")
    print(f"edge_search       : {GENEROUS['edge_search_threshold']} px")
    print(f"monte_times       : {GENEROUS['monte_times']}")
    print(f"max_candidates    : {GENEROUS['max_candidates_per_step']}")
    print(f"max_new_starts    : {MASK_RERUN['max_new_starts']}")
    print(f"node_snap_tol     : {MERGE['node_snap_tolerance_px']} px (reduced from 10)")
    print(f"Post-merge filter : light (angle + dedup only, no tiny/one-edge/dangling)")
    print()

    entries = json.loads(MANIFEST.read_text())
    if cli.max_samples:
        entries = entries[:cli.max_samples]
        print(f"[Limited to first {cli.max_samples} samples]")

    all_rows = []

    for entry in entries:
        sid      = entry["sample_id"]
        cat      = entry.get("category", "unknown")
        src_path = PROJECT_ROOT / entry["source_model_clean"]
        wg_path  = entry.get("wall_graph", "")
        gt_n, gt_e = load_gt_counts(wg_path)
        nc_bin   = node_count_bin(gt_n)

        if not src_path.exists():
            print(f"[{sid}] SKIP — source not found: {src_path}")
            continue

        t0 = time.perf_counter()
        print(f"\n[{sid}] ({cat})  gt_nodes={gt_n}  gt_edges={gt_e}  bin={nc_bin}")

        # True-pad preprocessing
        base_pil, preproc_metrics = preprocess_crop512_margin20_truepad(src_path)
        touches = preproc_metrics["content_touches_edge"]
        margins = preproc_metrics["final_canvas_margins_px"]
        if touches:
            print(f"  [BUG] content still touches canvas edge after true-pad preprocessing!")
        else:
            print(f"  margins: L={margins['left']} T={margins['top']} "
                  f"R={margins['right']} B={margins['bottom']} px")

        gray_arr = np.asarray(base_pil.convert("L"))

        accepted, discarded = run_generous_multistart(
            model, postprocessor, base_pil, device, GENEROUS, MASK_RERUN
        )

        # Stage 1: component-level counts (sum across accepted components)
        comp_node_count  = sum(c["num_points"] for c in accepted)
        comp_edge_count  = sum(c["num_edges"]  for c in accepted)

        # Stage 2: merge components (node snap + intersections + collinear)
        merged_pts_raw, merged_edges_raw = merge_components(
            accepted,
            snap_tol  = MERGE["node_snap_tolerance_px"],
            inter_tol = MERGE["edge_intersection_tolerance_px"],
            col_tol   = MERGE["collinear_overlap_tolerance_px"],
        )
        merged_raw_node_count = len(merged_pts_raw)
        merged_raw_edge_count = len(merged_edges_raw)

        # Stage 3: light post-merge filter (angle + dedup only)
        merged_pts, merged_edges, light_fstats = apply_light_post_merge_filter(
            merged_pts_raw, merged_edges_raw, gray_arr, FILTERS
        )

        # Soft score final graph
        merged_scores = compute_soft_scores(merged_pts, merged_edges, gray_arr, SCORING)

        elapsed = time.perf_counter() - t0
        empty   = len(merged_pts) == 0

        # Aggregated per-MC-attempt filter stats
        agg_fstats = _agg_filter_stats(accepted)

        # ---- write outputs -------------------------------------------------
        out_dir = OUTPUT_BASE / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        base_pil.save(str(out_dir / "input.png"))

        graph_json = {
            "nodes": [[int(x), int(y)] for x, y in merged_pts],
            "edges": [[int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])] for p1, p2 in merged_edges],
        }
        (out_dir / "graph_pred.json").write_text(
            json.dumps(graph_json, indent=2), encoding="utf-8"
        )
        (out_dir / "graph_pred.svg").write_text(
            make_svg_merged(merged_pts, merged_edges), encoding="utf-8"
        )

        # Three overlays:
        # 1. components: per-component colored, before merge
        make_overlay_components(base_pil, accepted).save(
            str(out_dir / "graph_overlay_components.png")
        )
        # 2. merged: after merge, before post-merge filter
        make_overlay_normal(base_pil, merged_pts_raw, merged_edges_raw).save(
            str(out_dir / "graph_overlay_merged.png")
        )
        # 3. final: after light post-merge filter
        make_overlay_normal(base_pil, merged_pts, merged_edges).save(
            str(out_dir / "graph_overlay.png")
        )

        # components.json
        comp_out = {
            "sample_id": sid,
            "components_before_merge": [
                {
                    "component_id":    c["component_id"],
                    "source":          c["source"],
                    "num_points":      c["num_points"],
                    "num_edges":       c["num_edges"],
                    "stop_code":       c["stop_code"],
                    "candidate_score": c["candidate_score"],
                    "selected_attempt": c["selected_attempt"],
                    "filter_stats":    c["filter_stats"],
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
                "num_nodes_after_merge":       merged_raw_node_count,
                "num_edges_after_merge":       merged_raw_edge_count,
                "num_nodes_after_light_filter": len(merged_pts),
                "num_edges_after_light_filter": len(merged_edges),
                "light_post_merge_filter_stats": light_fstats,
            },
        }
        (out_dir / "components.json").write_text(
            json.dumps(comp_out, indent=2), encoding="utf-8"
        )

        # metrics.json — includes preprocessing metrics and stage counts
        metrics = {
            "sample_id":    sid,
            "category":     cat,
            # Preprocessing
            "source_variant":         preproc_metrics["source_variant"],
            "standardized_margin":    preproc_metrics["standardized_margin"],
            "content_bbox_original":  preproc_metrics["content_bbox_original"],
            "content_bbox_after_preprocess": preproc_metrics["content_bbox_after_preprocess"],
            "final_canvas_margins_px": preproc_metrics["final_canvas_margins_px"],
            "content_touches_edge":   touches,
            # GT info
            "gt_node_count":  gt_n,
            "gt_edge_count":  gt_e,
            "node_count_bin": nc_bin,
            # Generation settings
            "first_step_threshold":    GENEROUS["first_step_threshold"],
            "later_step_threshold":    GENEROUS["later_step_threshold"],
            "first_step_force_best":   GENEROUS["first_step_force_best"],
            "edge_search_threshold":   GENEROUS["edge_search_threshold"],
            "monte_times":             GENEROUS["monte_times"],
            "max_candidates_per_step": GENEROUS["max_candidates_per_step"],
            "max_new_starts":          MASK_RERUN["max_new_starts"],
            # Stage-by-stage counts
            "stage_counts": {
                "components_nodes": comp_node_count,
                "components_edges": comp_edge_count,
                "merged_nodes":     merged_raw_node_count,
                "merged_edges":     merged_raw_edge_count,
                "final_nodes":      len(merged_pts),
                "final_edges":      len(merged_edges),
                "nodes_removed_by_post_merge_filter": merged_raw_node_count - len(merged_pts),
                "edges_removed_by_post_merge_filter": merged_raw_edge_count - len(merged_edges),
            },
            # Per-MC-attempt hard filter stats (aggregated across components)
            "hard_filters_per_attempt": {
                "edges_removed_angle_filter":            agg_fstats["edges_removed_angle_filter"],
                "components_removed_tiny":               agg_fstats["components_removed_tiny"],
                "points_removed_tiny_components":        agg_fstats["points_removed_tiny_components"],
                "edges_removed_tiny_components":         agg_fstats["edges_removed_tiny_components"],
                "one_edge_components_removed":           agg_fstats["one_edge_components_removed"],
                "one_edge_components_kept_by_evidence":  agg_fstats["one_edge_components_kept_by_evidence"],
                "dangling_edges_removed":                agg_fstats["dangling_edges_removed"],
            },
            # Light post-merge filter stats
            "light_post_merge_filter": light_fstats,
            # Soft scores on final graph
            "soft_scores": {
                "wall_evidence_alignment_score": merged_scores["wall_evidence_alignment_score"],
                "rectangle_cycle_count":         merged_scores["rectangle_cycle_count"],
                "rectangle_cycle_score":         merged_scores["rectangle_cycle_score"],
                "dangling_node_count":           merged_scores["dangling_node_count"],
                "dangling_penalty":              merged_scores["dangling_penalty"],
                "unsupported_edge_ratio":        merged_scores["unsupported_edge_ratio"],
                "small_component_count":         merged_scores["small_component_count"],
                "small_component_penalty":       merged_scores["small_component_penalty"],
                "candidate_score":               compute_candidate_score(
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
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

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

        all_rows.append(metrics)

        comp_tag = f" [{len(accepted)}comp]" if len(accepted) > 1 else ""
        status   = "EMPTY" if empty else f"{len(merged_pts)}pt {len(merged_edges)}ed"
        ev_tag   = f"  ev={merged_scores['wall_evidence_alignment_score']:.2f}" if not empty else ""
        cyc_tag  = f"  cyc={merged_scores['rectangle_cycle_count']}" if not empty else ""
        print(f"  {status}{comp_tag}{ev_tag}{cyc_tag}  ({elapsed:.1f}s)")

    _write_summary(all_rows)
    print(f"\nDone. Outputs under: {OUTPUT_BASE}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _write_summary(rows):
    if not rows:
        print("No rows to summarize.")
        return

    n        = len(rows)
    empty    = sum(1 for r in rows if r["empty"])
    nonempty = n - empty

    def _mean(field):
        vals = [r[field] for r in rows if not r["empty"]]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def _smean(key):
        vals = [r["soft_scores"][key] for r in rows if not r["empty"]]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def _stage_mean(stage_key):
        vals = [r["stage_counts"][stage_key] for r in rows if not r["empty"]]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    avg_nodes = _mean("final_num_points")
    avg_edges = _mean("final_num_edges")
    avg_wall  = _smean("wall_evidence_alignment_score")
    avg_cycle = _smean("rectangle_cycle_score")
    avg_dang  = _smean("dangling_penalty")
    avg_unsup = _smean("unsupported_edge_ratio")

    total_edge_touch = sum(1 for r in rows if r.get("content_touches_edge", False))

    total_angle_rm   = sum(r["hard_filters_per_attempt"]["edges_removed_angle_filter"] for r in rows)
    total_tiny_rm    = sum(r["hard_filters_per_attempt"]["components_removed_tiny"] for r in rows)
    total_oneedge_rm = sum(r["hard_filters_per_attempt"]["one_edge_components_removed"] for r in rows)
    total_dangle_rm  = sum(r["hard_filters_per_attempt"]["dangling_edges_removed"] for r in rows)

    total_light_angle = sum(r["light_post_merge_filter"]["edges_removed_angle_filter"] for r in rows)
    total_light_dedup = sum(r["light_post_merge_filter"]["duplicate_edges_removed"] for r in rows)

    nc_bins = ["10-50", "51-80", "81-120", "120+"]
    by_bin  = {}
    for b in nc_bins:
        sub = [r for r in rows if r["node_count_bin"] == b]
        if sub:
            by_bin[b] = {
                "total":     len(sub),
                "empty":     sum(1 for r in sub if r["empty"]),
                "avg_nodes": round(sum(r["final_num_points"] for r in sub) / len(sub), 1),
                "avg_wall":  round(sum(r["soft_scores"]["wall_evidence_alignment_score"]
                                       for r in sub if not r["empty"]) /
                                   max(sum(1 for r in sub if not r["empty"]), 1), 3),
            }

    summary_data = {
        "task": "Task 30",
        "generation_settings":  GENEROUS,
        "mask_rerun_settings":  MASK_RERUN,
        "merge_settings":       MERGE,
        "filter_settings":      FILTERS,
        "scoring_settings":     SCORING,
        "total_samples":        n,
        "non_empty":            nonempty,
        "empty":                empty,
        "non_empty_rate":       round(nonempty / n, 3) if n > 0 else 0,
        "avg_final_nodes":      avg_nodes,
        "avg_final_edges":      avg_edges,
        "avg_wall_evidence":    avg_wall,
        "avg_cycle_score":      avg_cycle,
        "avg_dangling_penalty": avg_dang,
        "avg_unsupported_ratio": avg_unsup,
        "samples_content_touches_edge": total_edge_touch,
        "by_node_count_bin": by_bin,
    }

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / "summary.json").write_text(
        json.dumps({"rows": rows, "summary": summary_data}, indent=2), encoding="utf-8"
    )

    ne_rate = f"{nonempty/n:.1%}" if n > 0 else "N/A"
    md = [
        "# Task 30 — True 20% Padding and Less Destructive Final Filtering",
        "",
        "## Samples Tested",
        "",
        "Three samples from `data/raster2graph/preprocess_test_samples.json` (same as Task 29):",
    ]
    for r in rows:
        md.append(f"- Sample `{r['sample_id']}` (category={r['category']}, bin={r['node_count_bin']})")

    md += [
        "",
        "## Preprocessing: crop512_margin20_truepad",
        "",
        "- **Source variant**: `crop512_margin20_truepad`",
        "- **Standardized margin**: 20% of content bbox on each side",
        "- **Method**: crop content bbox exactly → paste into new white image with 20% padding → scale long edge to 512px → center on 512×512 white canvas",
        "- **Previous method (retired)**: clamped bbox expansion — `crop(max(0, x0-pad), min(W, x1+pad))` failed when content touched the original image boundary",
        f"- Samples where content still touches 512px canvas edge after true padding: **{total_edge_touch}/{n}**",
    ]
    if total_edge_touch > 0:
        md += ["- **BUG**: True padding did NOT prevent edge-touching content. Investigate source images."]
    else:
        md += ["- **All samples**: no wall content touching canvas edge after 20% true-pad preprocessing. Fix confirmed."]

    md += ["", "### Final Canvas Margins Per Sample", ""]
    md += ["| Sample | Left px | Top px | Right px | Bottom px | Edge-Touch |"]
    md += ["|--------|---------|--------|----------|-----------|------------|"]
    for r in rows:
        m = r["final_canvas_margins_px"]
        et = "YES-BUG" if r.get("content_touches_edge") else "ok"
        md.append(f"| {r['sample_id']} | {m['left']} | {m['top']} | {m['right']} | {m['bottom']} | {et} |")

    md += [
        "",
        "## Generation Settings",
        "",
        "| Parameter | Value | Change from Task 29 |",
        "|-----------|-------|---------------------|",
        f"| source_variant | crop512_margin20_truepad | changed (was margin10) |",
        f"| first_step_threshold | {GENEROUS['first_step_threshold']} | unchanged |",
        f"| later_step_threshold | {GENEROUS['later_step_threshold']} | unchanged |",
        f"| first_step_force_best | {GENEROUS['first_step_force_best']} | unchanged |",
        f"| edge_search_threshold | {GENEROUS['edge_search_threshold']} px | unchanged |",
        f"| monte_times | {GENEROUS['monte_times']} | restored to Task 29 spec (was 3 in script) |",
        f"| max_candidates_per_step | {GENEROUS['max_candidates_per_step']} | restored to Task 29 spec (was 15 in script) |",
        f"| max_new_starts | {MASK_RERUN['max_new_starts']} | unchanged |",
        f"| node_snap_tolerance_px | {MERGE['node_snap_tolerance_px']} | reduced from 10 (less geometry shift) |",
        "",
        "## Stage-by-Stage Node/Edge Counts",
        "",
        "| Sample | Comp Nodes | Comp Edges | Merged Nodes | Merged Edges | Final Nodes | Final Edges | Removed by Filter |",
        "|--------|-----------|-----------|-------------|-------------|------------|------------|-------------------|",
    ]
    for r in rows:
        sc = r["stage_counts"]
        removed = f"{sc['nodes_removed_by_post_merge_filter']}n/{sc['edges_removed_by_post_merge_filter']}e"
        md.append(f"| {r['sample_id']} | {sc['components_nodes']} | {sc['components_edges']} "
                  f"| {sc['merged_nodes']} | {sc['merged_edges']} "
                  f"| {sc['final_nodes']} | {sc['final_edges']} | {removed} |")

    md += [
        "",
        "## Post-Merge Filtering: Less Destructive",
        "",
        "**Task 29 behavior**: full hard filter applied after merge (angle + tiny + one-edge + dangling).",
        "**Task 30 behavior**: light post-merge filter only (angle violations + exact duplicate edges).",
        "",
        "Rationale: merge splits edges at intersections creating short fragments that are still valid",
        "wall segments. Tiny/one-edge/dangling deletion after merge removes these valid fragments.",
        "",
        f"- Light post-merge angle violations removed (total): {total_light_angle}",
        f"- Light post-merge duplicate edges removed (total): {total_light_dedup}",
        "",
        "### Whether graph_overlay preserves component graph better",
        "",
        "Compare `graph_overlay_components.png` vs `graph_overlay.png` per sample.",
        "Stage counts above show how many nodes/edges survived from component → merge → final.",
    ]

    for r in rows:
        sc = r["stage_counts"]
        if sc["components_edges"] > 0:
            retention = sc["final_edges"] / sc["components_edges"] * 100
        else:
            retention = 0.0
        md.append(f"- Sample {r['sample_id']}: {sc['components_edges']} component edges "
                  f"→ {sc['final_edges']} final edges ({retention:.0f}% retained)")

    md += [
        "",
        "## Hard Filter Removals Per MC Attempt (before candidate reranking)",
        "",
        f"- Edges removed by angle filter: {total_angle_rm}",
        f"- Tiny components removed: {total_tiny_rm}",
        f"- One-edge components removed: {total_oneedge_rm}",
        f"- Short dangling edges removed: {total_dangle_rm}",
        "",
        "## Final Results",
        "",
        f"- Total samples processed: {n}",
        f"- Non-empty graph rate: {nonempty}/{n} = {ne_rate}",
        f"- Average final nodes: {avg_nodes:.1f}",
        f"- Average final edges: {avg_edges:.1f}",
        f"- Average wall evidence score: {avg_wall:.3f}",
        f"- Average rectangle cycle score: {avg_cycle:.3f}",
        f"- Average dangling penalty: {avg_dang:.3f}",
        f"- Average unsupported edge ratio: {avg_unsup:.3f}",
        "",
        "## Per-Sample Results",
        "",
        "| Sample | Category | Bin | Nodes | Edges | Wall Ev | Cycles | Edge-Touch | Empty |",
        "|--------|----------|-----|-------|-------|---------|--------|------------|-------|",
    ]
    for r in rows:
        ss = r["soft_scores"]
        et = "YES-BUG" if r.get("content_touches_edge", False) else "ok"
        md.append(
            f"| {r['sample_id']} | {r['category']} | {r['node_count_bin']} "
            f"| {r['final_num_points']} | {r['final_num_edges']} "
            f"| {ss['wall_evidence_alignment_score']:.2f} "
            f"| {ss['rectangle_cycle_count']} "
            f"| {et} "
            f"| {'YES' if r['empty'] else 'no'} |"
        )

    md += [
        "",
        "## Analysis: What Was The Main Improvement?",
        "",
        "**Preprocessing fix** (true 20% padding):",
        f"  Task 29: 3/3 samples had content touching canvas edge.",
        f"  Task 30: {total_edge_touch}/3 samples have content touching canvas edge.",
        "",
        "**Filtering fix** (less destructive post-merge):",
        "  Compare stage counts: if final nodes/edges are closer to component counts than in Task 29,",
        "  the light filter is preserving more useful geometry.",
        "",
        "**Whether filtering is still too destructive**:",
        "  Check `graph_overlay_components.png` vs `graph_overlay.png` visually.",
        "  If graph_overlay.png still looks much sparser, the merge stage itself may be",
        "  collapsing geometry via snapping (node_snap_tolerance_px reduced to 6 in Task 30).",
        "",
        "## Three Visual Overlays",
        "",
        "Each sample now has three overlays:",
        "- `graph_overlay_components.png` — per-component colored, before merge",
        "- `graph_overlay_merged.png` — after merge, before light post-merge filter",
        "- `graph_overlay.png` — final after light post-merge filter",
        "",
    ]

    (OUTPUT_BASE / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nSummary JSON -> {OUTPUT_BASE / 'summary.json'}")
    print(f"Summary MD   -> {OUTPUT_BASE / 'summary.md'}")

    print(f"\n=== RESULTS ===")
    print(f"  Non-empty:       {nonempty}/{n} ({ne_rate})")
    print(f"  Avg nodes:       {avg_nodes:.1f}")
    print(f"  Avg edges:       {avg_edges:.1f}")
    print(f"  Avg wall_ev:     {avg_wall:.3f}")
    print(f"  Edge-touch:      {total_edge_touch}/{n} (should be 0 with true padding)")


if __name__ == "__main__":
    main()
