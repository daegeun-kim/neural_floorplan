"""Debug overlay and metrics for the v008 point-graph pipeline (spec_v008 SS15).

Searched points and graph edges belong only here and in metrics.json - never
in vector.svg. Rejected/unresolved evidence is reported in metrics.json only
(task19: dropped from the overlay image itself - it cluttered the render
without being needed to read the final graph).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

from .graph_types import ComponentRecord, DoorCandidateRecord, GraphEdge, GraphPoint, RejectedEvidence, ValidationIssue

POINT_COLORS: dict[str, tuple[int, int, int]] = {
    "wall_point": (80, 80, 200),
    "wall_window_point": (60, 120, 220),
    "wall_door_hinge_point": (235, 140, 80),
    "wall_door_end_point": (160, 70, 180),
}

EDGE_COLORS: dict[str, tuple[int, int, int]] = {
    "wall": (80, 80, 200),
    "window": (60, 120, 220),
    "door_origin": (160, 70, 180),
}

DOOR_CANDIDATE_HIGH_CONFIDENCE_COLOR = (220, 30, 30)
DOOR_CANDIDATE_LOW_CONFIDENCE_COLOR = (220, 150, 30)
DOOR_CANDIDATE_CONFIDENCE_THRESHOLD = 0.75


POINT_LABELS: dict[str, str] = {
    "wall_point": "wall point (generic - end/corner/T/cross)",
    "wall_window_point": "wall-window end (final, hosted on wall)",
    "wall_door_hinge_point": "wall-door hinge (final, hosted on wall)",
    "wall_door_end_point": "wall-door end (final, hosted on wall)",
}

EDGE_LABELS: dict[str, str] = {
    "wall": "wall edge (final wall skeleton/polygon line)",
    "window": "window edge (final, between paired window points)",
    "door_origin": "door_origin edge (final hinge-to-end line)",
}

# (label, color) rows describing every red door_arc candidate overlay drawn
# in build_debug_overlay's door_candidates loop - the accepted bbox, its
# hinge/end markers (larger circles than the final POINT_COLORS dots), and
# the hinge-to-end connector line all share this same per-candidate color.
DOOR_CANDIDATE_LEGEND_ROWS: list[tuple[str, tuple[int, int, int]]] = [
    ("door_arc candidate, accepted (conf>=0.75)", DOOR_CANDIDATE_HIGH_CONFIDENCE_COLOR),
    ("door_arc candidate, accepted (conf<0.75)", DOOR_CANDIDATE_LOW_CONFIDENCE_COLOR),
]


def _add_legend(img: Image.Image, scale_info) -> Image.Image:
    legend_width = 280
    row_h = 18
    pad = 8
    rows = len(POINT_COLORS) + len(EDGE_COLORS) + len(DOOR_CANDIDATE_LEGEND_ROWS) + 4
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
        draw.text((x0 + 16, y), EDGE_LABELS[edge_type], fill=(0, 0, 0))
        y += row_h

    for label, color in DOOR_CANDIDATE_LEGEND_ROWS:
        draw.rectangle([x0, y + 2, x0 + 10, y + 14], outline=color, width=2)
        draw.text((x0 + 16, y), label, fill=(0, 0, 0))
        y += row_h

    draw.text((x0, y), "scale bar (bottom-left of image):", fill=(0, 0, 0))
    y += row_h
    draw.text((x0, y), f"length = {SCALE_BAR_REFERENCE_MM:.0f}mm at inferred scale", fill=(0, 0, 0))
    return out


SCALE_BAR_REFERENCE_MM = 1000.0


def _draw_scale_bar(draw: ImageDraw.ImageDraw, img_width: int, img_height: int, scale_info) -> None:
    """A ground-truth scale bar (default 1000mm) sized from the same
    px_to_mm the rest of the pipeline used, so a viewer can sanity-check the
    inferred scale directly against the image instead of trusting the
    px_to_mm number alone."""
    margin = 10
    y = img_height - 18
    if scale_info.px_to_mm is None or scale_info.scale_status not in ("resolved", "estimated"):
        draw.text((margin, y - 8), "scale bar: unavailable (scale unknown)", fill=(180, 0, 0))
        return

    bar_px = SCALE_BAR_REFERENCE_MM / scale_info.px_to_mm
    max_bar_px = max(img_width - 2 * margin, 1)
    label_mm = SCALE_BAR_REFERENCE_MM
    if bar_px > max_bar_px:
        # Reference length doesn't fit at this image's resolution - show
        # however many mm the available width actually represents instead of
        # silently clipping a 1000mm bar into a shorter, misleading one.
        bar_px = max_bar_px
        label_mm = bar_px * scale_info.px_to_mm

    x0 = margin
    x1 = x0 + bar_px
    label = f"{label_mm:.0f} mm"
    text_w = draw.textlength(label) if hasattr(draw, "textlength") else len(label) * 6
    backing_w = max(bar_px, text_w) + 6
    draw.rectangle([x0 - 3, y - 21, x0 - 3 + backing_w, y + 6], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.line([(x0, y), (x1, y)], fill=(0, 0, 0), width=2)
    tick_h = 5
    draw.line([(x0, y - tick_h), (x0, y + tick_h)], fill=(0, 0, 0), width=2)
    draw.line([(x1, y - tick_h), (x1, y + tick_h)], fill=(0, 0, 0), width=2)
    draw.text((x0, y - 18), label, fill=(0, 0, 0))


def build_debug_overlay(
    rgb: np.ndarray,
    points: list[GraphPoint],
    edges: list[GraphEdge],
    rejected_evidence: list[RejectedEvidence],
    scale_info,
    door_candidates: Optional[list[DoorCandidateRecord]] = None,
) -> Image.Image:
    """Render searched points by type and graph edges, plus every accepted
    red door_arc candidate's bbox and hinge-to-end connector (task13 "Debug
    Overlay Requirements") - low-confidence candidates are drawn in a
    distinct color from high-confidence ones. ``wall_door_hinge_point``/
    ``wall_door_end_point`` markers themselves stay in their own
    orange/purple ``POINT_COLORS`` (not the candidate's confidence color),
    so the door's two final points always read the same way the other final
    point types do. A bottom-left scale bar shows the inferred px_to_mm as a
    physical reference length (default 1000mm).

    ``rejected_evidence`` is accepted for signature/call-site stability and
    still reported in ``metrics.json`` (``build_metrics``), but is no longer
    drawn here - rejected/unresolved evidence cluttered the overlay without
    being needed to read the final graph.
    """
    del rejected_evidence
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)

    for edge in edges:
        color = EDGE_COLORS.get(edge.edge_type, (120, 120, 120))
        draw.line([edge.start, edge.end], fill=color, width=1)

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
        # The hinge/end points themselves are drawn once already, above, in
        # their own orange/purple POINT_COLORS - not re-drawn here in the
        # candidate's confidence color, so they read the same way every
        # other final point type does instead of looking like red circles.
        hinge = hinge_by_arc.get(cand.red_component_id)
        end = end_by_arc.get(cand.red_component_id)
        if hinge is not None and end is not None:
            draw.line([hinge.coordinate, end.coordinate], fill=color, width=1)

    label = (
        f"unit={scale_info.unit} status={scale_info.scale_status} "
        f"px_to_mm={scale_info.px_to_mm} conf={scale_info.confidence:.2f}"
    )
    draw.text((4, 4), label, fill=(255, 0, 0))
    _draw_scale_bar(draw, img.width, img.height, scale_info)
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
                "all_four_bbox_vertices": c.all_four_bbox_vertices,
                "selected_hinge_vertex": c.selected_hinge_vertex,
                "selected_end_vertex": c.selected_end_vertex,
                "hinge_vertex_score": c.hinge_vertex_score,
                "end_vertex_score": c.end_vertex_score,
                "selected_bbox_edge": c.selected_bbox_edge,
                "host_wall_alignment_score": c.host_wall_alignment_score,
                "door_width_mm": c.door_width_mm,
            }
            for c in (door_candidates or [])
        ],
    }


def write_metrics(output_path: str | Path, metrics: dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
