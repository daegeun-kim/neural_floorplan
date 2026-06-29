"""Serialize Phase 4 pipeline results to final_vector.json (spec_v008 §11).

task34:
    - doors/windows now record snapped_points_original, snapped_points_adjusted,
      final_points so the JSON captures the full adjustment audit trail (Part A).
    - build_final_vector_json() accepts optional door_geometries list for
      evidence-based hinge/swing serialization (Part B).
    - metrics now include topology-snap fields from WallGeometry (Part C).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .door_geometry import DoorGeometry, compute_door_geometry, door_geometry_to_dict
from .opening_hosting import HostedOpening, RejectedOpening
from .wall_interval_editing import TrimmedGraph
from ..primitives.scale import ScaleInfo


def _scale_to_dict(scale: ScaleInfo) -> dict:
    return {
        "status": scale.scale_status,
        "px_to_mm": scale.px_to_mm,
        "source": scale.scale_source,
        "confidence": round(scale.confidence, 4),
        "evidence": scale.diagnostics if scale.diagnostics else [],
    }


def _hosted_opening_to_dict(op: HostedOpening, gap: Optional[dict] = None) -> dict:
    """Serialize a hosted opening; include interval + point adjustment fields.

    op.snapped_points are the FINAL (adjusted) endpoints used for wall trimming
    and primitives (after apply_adjusted_intervals_to_hosted_openings).

    gap carries original_interval / adjusted_interval so the dict can also
    reconstruct snapped_points_original from the original t values.
    """
    final_pts = [[round(p[0], 2), round(p[1], 2)] for p in op.snapped_points]
    d: dict = {
        "source_component_id": op.source_component_id,
        "opening_type": op.opening_type,
        "host_edge_idx": op.host_edge_idx,
        "host_edge_raw": op.host_edge_raw,
        "raw_points": [[round(p[0], 2), round(p[1], 2)] for p in op.raw_points],
        "snapped_points": final_pts,          # == snapped_points_adjusted == final_points
        "final_points": final_pts,            # explicit "source of truth" alias
        "width_px": round(op.width_px, 2),
        "width_mm": round(op.width_mm, 1) if op.width_mm is not None else None,
        "snapped_module_mm": op.snapped_module_mm,
        "confidence": round(op.confidence, 4),
    }
    if gap is not None:
        orig_interval = gap.get("original_interval", [])
        adj_interval  = gap.get("adjusted_interval",  [])
        d["original_interval"]        = orig_interval
        d["adjusted_interval"]        = adj_interval
        d["snapped_points_adjusted"]  = final_pts
        d["was_adjusted"]             = gap.get("was_adjusted", False)
        d["adjustment_reason"]        = gap.get("adjustment_reason", "")
        d["adjustment_px"]            = gap.get("adjustment_px", 0.0)
        d["adjustment_mm"]            = gap.get("adjustment_mm")
        d["overlap_resolution_priority"] = gap.get("overlap_resolution_priority", "not_needed")
        # Reconstruct original snapped points from the edge + original interval
        hr = gap.get("host_edge_raw", op.host_edge_raw)
        if hr and len(orig_interval) == 2:
            x1, y1, x2, y2 = hr[0], hr[1], hr[2], hr[3]
            ox = x1 + orig_interval[0] * (x2 - x1)
            oy = y1 + orig_interval[0] * (y2 - y1)
            ox2 = x1 + orig_interval[1] * (x2 - x1)
            oy2 = y1 + orig_interval[1] * (y2 - y1)
            d["snapped_points_original"] = [[round(ox, 2), round(oy, 2)],
                                             [round(ox2, 2), round(oy2, 2)]]
        else:
            d["snapped_points_original"] = final_pts
    return d


def _rejected_opening_to_dict(op: RejectedOpening) -> dict:
    return {
        "opening_type": op.opening_type,
        "source_component_id": op.source_component_id,
        "raw_points": [[round(p[0], 2), round(p[1], 2)] for p in op.raw_points],
        "rejection_reason": op.rejection_reason,
        "debug_confidence": round(op.debug_confidence, 4),
    }


def _find_gap(op: HostedOpening, trimmed: TrimmedGraph) -> Optional[dict]:
    """Find the opening_gap dict that corresponds to a hosted opening."""
    for gap in trimmed.opening_gaps:
        if (gap.get("source_component_id") == op.source_component_id
                and gap.get("opening_type") == op.opening_type
                and gap.get("host_edge_idx") == op.host_edge_idx):
            return gap
    return None


def build_final_vector_json(
    preprocessing_manifest: dict,
    scale_info: ScaleInfo,
    raw_graph: dict,
    aligned_graph: dict,
    trimmed_graph: TrimmedGraph,
    hosted_doors: list[HostedOpening],
    hosted_windows: list[HostedOpening],
    rejected_openings: list[RejectedOpening],
    wall_geometry: Any,  # WallGeometry from wall_buffering
    metrics: Optional[dict] = None,
    door_geometries: Optional[list[DoorGeometry]] = None,
) -> dict:
    """Build the complete final_vector.json structure (spec_v008 §11 + task34).

    Args:
        door_geometries: pre-computed evidence-based DoorGeometry per door (Part B);
                         if None, fallback compute_door_geometry() is used.
    """
    walls_geom = []
    if wall_geometry is not None and wall_geometry.polygon is not None:
        poly = wall_geometry.polygon
        if hasattr(poly, "geoms"):
            polys = list(poly.geoms)
        else:
            polys = [poly]
        for p in polys:
            if hasattr(p, "exterior"):
                walls_geom.append({
                    "type": "Polygon",
                    "exterior": [[round(x, 2), round(y, 2)] for x, y in p.exterior.coords],
                    "interiors": [
                        [[round(x, 2), round(y, 2)] for x, y in ring.coords]
                        for ring in p.interiors
                    ],
                })

    # Merge topology-snap metrics from WallGeometry into the metrics dict
    merged_metrics: dict = dict(metrics or {})
    if wall_geometry is not None:
        merged_metrics.setdefault("pre_buffer_node_count", wall_geometry.pre_buffer_node_count)
        merged_metrics.setdefault("post_snap_node_count", wall_geometry.post_snap_node_count)
        merged_metrics.setdefault("pre_buffer_edge_count", wall_geometry.pre_buffer_edge_count)
        merged_metrics.setdefault("wall_chain_count", wall_geometry.chain_count)
        merged_metrics.setdefault("disconnected_endpoint_count", wall_geometry.disconnected_endpoint_count)

    def _door_geom_for(idx: int, door: HostedOpening) -> DoorGeometry:
        if door_geometries and idx < len(door_geometries):
            return door_geometries[idx]
        return compute_door_geometry(door)

    return {
        "coordinate_space": "preprocessed_512",
        "preprocessing": preprocessing_manifest,
        "scale": _scale_to_dict(scale_info),
        "wall_graph": {
            "raw": {
                "nodes": raw_graph.get("nodes", []),
                "edges": raw_graph.get("edges", []),
            },
            "aligned": {
                "nodes": aligned_graph.get("aligned_nodes", aligned_graph.get("nodes", [])),
                "edges": aligned_graph.get("aligned_edges", aligned_graph.get("edges", [])),
            },
            "trimmed": {
                "edges": trimmed_graph.wall_edges,
                "opening_gaps": trimmed_graph.opening_gaps,
            },
        },
        "openings": {
            "doors": [
                _hosted_opening_to_dict(d, _find_gap(d, trimmed_graph))
                for d in hosted_doors
            ],
            "windows": [
                _hosted_opening_to_dict(w, _find_gap(w, trimmed_graph))
                for w in hosted_windows
            ],
            "rejected": (
                [_rejected_opening_to_dict(r) for r in rejected_openings]
                + trimmed_graph.last_resort_rejected
            ),
        },
        "geometry": {
            "walls": walls_geom,
            "doors": [
                {
                    "source_component_id": d.source_component_id,
                    "final_points": [[round(p[0], 2), round(p[1], 2)] for p in d.snapped_points],
                    "snapped_points": [[round(p[0], 2), round(p[1], 2)] for p in d.snapped_points],
                    "width_px": round(d.width_px, 2),
                    "width_mm": round(d.width_mm, 1) if d.width_mm is not None else None,
                    "door_geometry": door_geometry_to_dict(
                        _door_geom_for(i, d), width_mm=d.width_mm
                    ),
                }
                for i, d in enumerate(hosted_doors)
            ],
            "windows": [
                {
                    "source_component_id": w.source_component_id,
                    "final_points": [[round(p[0], 2), round(p[1], 2)] for p in w.snapped_points],
                    "points": [[round(p[0], 2), round(p[1], 2)] for p in w.snapped_points],
                    "width_px": round(w.width_px, 2),
                    "width_mm": round(w.width_mm, 1) if w.width_mm is not None else None,
                }
                for w in hosted_windows
            ],
        },
        "metrics": merged_metrics,
    }


def write_final_vector_json(data: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
