"""Debug overlay and metrics for the v008 point-graph pipeline (spec_v008 SS15).

Rejected/unresolved evidence and searched points belong only here and in
metrics.json - never in vector.svg.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

from .graph_types import ComponentRecord, DoorCandidateRecord, GraphEdge, GraphPoint, RejectedEvidence, ValidationIssue

POINT_COLORS: dict[str, tuple[int, int, int]] = {
    "1_wall_point": (80, 180, 80),
    "2_wall_point": (80, 80, 200),
    "3_wall_point": (200, 160, 40),
    "4_wall_point": (200, 40, 200),
    "wall_window_point": (60, 120, 220),
    "wall_door_hinge_point": (235, 140, 80),
    "wall_door_end_point": (160, 70, 180),
}

EDGE_COLORS: dict[str, tuple[int, int, int]] = {
    "wall": (80, 80, 200),
    "window": (60, 120, 220),
    "door_origin": (160, 70, 180),
}

REJECTED_COLOR = (160, 160, 160)


POINT_LABELS: dict[str, str] = {
    "1_wall_point": "1 wall end",
    "2_wall_point": "2 wall corner",
    "3_wall_point": "3 wall T-junction",
    "4_wall_point": "4 wall cross",
    "wall_window_point": "wall-window end",
    "wall_door_hinge_point": "wall-door hinge",
    "wall_door_end_point": "wall-door end",
}


def _add_legend(img: Image.Image, scale_info) -> Image.Image:
    legend_width = 230
    row_h = 18
    pad = 8
    rows = len(POINT_COLORS) + len(EDGE_COLORS) + 2
    out = Image.new("RGB", (img.width + legend_width, max(img.height, pad * 2 + rows * row_h)), (245, 245, 245))
    out.paste(img, (0, 0))

    draw = ImageDraw.Draw(out)
    x0 = img.width + pad
    y = pad
    draw.text((x0, y), "Debug legend", fill=(0, 0, 0))
    y += row_h
    draw.text((x0, y), f"scale: {scale_info.scale_status}", fill=(180, 0, 0))
    y += row_h

    for point_type, color in POINT_COLORS.items():
        cy = y + row_h // 2
        draw.ellipse([x0, cy - 4, x0 + 8, cy + 4], outline=color, width=2)
        draw.text((x0 + 16, y), POINT_LABELS[point_type], fill=(0, 0, 0))
        y += row_h

    for edge_type, color in EDGE_COLORS.items():
        cy = y + row_h // 2
        draw.line([(x0, cy), (x0 + 10, cy)], fill=color, width=2)
        draw.text((x0 + 16, y), f"{edge_type} edge", fill=(0, 0, 0))
        y += row_h

    draw.rectangle([x0, y + 3, x0 + 10, y + 13], outline=REJECTED_COLOR, width=1)
    draw.text((x0 + 16, y), "rejected evidence", fill=(0, 0, 0))
    return out


DOOR_CANDIDATE_HIGH_CONFIDENCE_COLOR = (220, 30, 30)
DOOR_CANDIDATE_LOW_CONFIDENCE_COLOR = (220, 150, 30)
DOOR_CANDIDATE_CONFIDENCE_THRESHOLD = 0.75


def build_debug_overlay(
    rgb: np.ndarray,
    points: list[GraphPoint],
    edges: list[GraphEdge],
    rejected_evidence: list[RejectedEvidence],
    scale_info,
    door_candidates: Optional[list[DoorCandidateRecord]] = None,
) -> Image.Image:
    """Render searched points by type, graph edges, and rejected/unresolved
    evidence (spec_v008 SS15), plus every red door_arc candidate bbox with
    its inferred hinge/end points (task13 "Debug Overlay Requirements") -
    low-confidence candidates are drawn in a distinct color from
    high-confidence ones."""
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)

    for edge in edges:
        color = EDGE_COLORS.get(edge.edge_type, (120, 120, 120))
        draw.line([edge.start, edge.end], fill=color, width=1)

    for rej in rejected_evidence:
        if rej.bbox is not None:
            x0, y0, x1, y1 = rej.bbox
            draw.rectangle([x0, y0, x1, y1], outline=REJECTED_COLOR, width=1)
        elif rej.centroid is not None:
            cx, cy = rej.centroid
            draw.rectangle([cx - 5, cy - 5, cx + 5, cy + 5], outline=REJECTED_COLOR, width=1)

    for p in points:
        cx, cy = p.coordinate
        color = POINT_COLORS.get(p.point_type, (255, 255, 255))
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], outline=color, width=2)

    hinge_by_arc = {p.source_component_ids[0]: p for p in points if p.point_type == "wall_door_hinge_point" and p.source_component_ids}
    end_by_arc = {p.source_component_ids[0]: p for p in points if p.point_type == "wall_door_end_point" and p.source_component_ids}
    for cand in door_candidates or []:
        if not cand.created_door_candidate:
            continue
        color = (
            DOOR_CANDIDATE_HIGH_CONFIDENCE_COLOR
            if cand.door_confidence >= DOOR_CANDIDATE_CONFIDENCE_THRESHOLD
            else DOOR_CANDIDATE_LOW_CONFIDENCE_COLOR
        )
        x0, y0, x1, y1 = cand.red_bbox
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        hinge = hinge_by_arc.get(cand.red_component_id)
        end = end_by_arc.get(cand.red_component_id)
        if hinge is not None and end is not None:
            draw.line([hinge.coordinate, end.coordinate], fill=color, width=1)
        if hinge is not None:
            hx, hy = hinge.coordinate
            draw.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], outline=color, width=2)
        if end is not None:
            ex, ey = end.coordinate
            draw.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], outline=color, width=2)

    label = (
        f"unit={scale_info.unit} status={scale_info.scale_status} "
        f"px_to_mm={scale_info.px_to_mm} conf={scale_info.confidence:.2f}"
    )
    draw.text((4, 4), label, fill=(255, 0, 0))
    return _add_legend(img, scale_info)


def build_metrics(
    *,
    image_name: str,
    components: dict[str, list[ComponentRecord]],
    rejected_evidence: list[RejectedEvidence],
    points: list[GraphPoint],
    edges: list[GraphEdge],
    validation_issues: list[ValidationIssue],
    scale_info,
    door_candidates: Optional[list[DoorCandidateRecord]] = None,
) -> dict[str, Any]:
    point_counts: dict[str, int] = {}
    for p in points:
        point_counts[p.point_type] = point_counts.get(p.point_type, 0) + 1

    rejected_by_kind: dict[str, int] = {}
    for r in rejected_evidence:
        rejected_by_kind[r.kind] = rejected_by_kind.get(r.kind, 0) + 1

    return {
        "image": image_name,
        "components": {cls: len(records) for cls, records in components.items()},
        "points": point_counts,
        "edges": {
            "wall": sum(1 for e in edges if e.edge_type == "wall"),
            "window": sum(1 for e in edges if e.edge_type == "window"),
            "door_origin": sum(1 for e in edges if e.edge_type == "door_origin"),
        },
        "rejected_evidence": rejected_by_kind,
        "validation_issues": [
            {"rule": v.rule, "message": v.message, "severity": v.severity} for v in validation_issues
        ],
        "scale": {
            "unit": scale_info.unit,
            "px_to_mm": scale_info.px_to_mm,
            "scale_status": scale_info.scale_status,
            "scale_source": scale_info.scale_source,
            "confidence": scale_info.confidence,
            "diagnostics": getattr(scale_info, "diagnostics", {}),
        },
        "door_candidates": [
            {
                "red_component_id": c.red_component_id,
                "red_bbox": c.red_bbox,
                "red_bbox_long_edge_px": c.red_bbox_long_edge_px,
                "created_door_candidate": c.created_door_candidate,
                "scale_candidate_px_to_mm": c.scale_candidate_px_to_mm,
                "hinge_candidate_support_classes": c.hinge_candidate_support_classes,
                "end_candidate_support_classes": c.end_candidate_support_classes,
                "hinge_distance_to_red_bbox_mm": c.hinge_distance_to_red_bbox_mm,
                "end_distance_to_red_bbox_mm": c.end_distance_to_red_bbox_mm,
                "door_confidence": c.door_confidence,
                "door_inference_notes": c.door_inference_notes,
            }
            for c in (door_candidates or [])
        ],
    }


def write_metrics(output_path: str | Path, metrics: dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
