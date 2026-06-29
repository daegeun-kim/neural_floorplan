"""Shared data structures for the v008 orthogonal point-graph pipeline.

Per spec_v008 SS6, every stage of the pipeline (components -> point search ->
alignment -> connection -> door geometry -> final geometry) communicates
through these types, and ``MaskToVectorResult`` exposes every intermediate
artifact so tests can assert on each stage independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

Direction = Literal["left", "right", "up", "down"]
AttachmentType = Literal["wall", "window", "door_origin"]
EdgeType = Literal["wall", "window", "door_origin"]

# task15: the four degree-classified wall-point subtypes
# (1/2/3/4_wall_point) are collapsed into one generic "wall_point" - wall
# graph construction must not depend on accurate pre-classification of a
# point's eventual degree (task15 problem 1). This is a deliberate, explicit
# deviation from must-rule 22's literal seven-type enumeration; see
# spec_v008_phase3_mask_to_vector.md's task15 notes.
PointType = Literal[
    "wall_point",
    "wall_window_point",
    "wall_door_hinge_point",
    "wall_door_end_point",
]

ALL_POINT_TYPES: tuple[PointType, ...] = (
    "wall_point",
    "wall_window_point",
    "wall_door_hinge_point",
    "wall_door_end_point",
)

OPPOSITE_DIRECTION: dict[Direction, Direction] = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}


@dataclass
class Attachment:
    """One directional attachment at a GraphPoint (spec_v008 SS4/SS9)."""

    type: AttachmentType
    direction: Direction
    source: str
    evidence_length_px: float = 0.0
    confidence: float = 1.0
    host_thickness_px: float = 0.0


@dataclass
class GraphPoint:
    """A searched/aligned architectural point - exactly one of the seven
    allowed point types, with a local attachment table (spec_v008 SS9)."""

    id: str
    point_type: PointType
    coordinate: tuple[float, float]
    attachments: list[Attachment] = field(default_factory=list)
    source_component_ids: list[int] = field(default_factory=list)
    host_wall_edge_id: Optional[str] = None
    """The WallSkeletonEdge.id this point was projected onto, when known
    (window/door points). Lets point_connection.py connect the wall edge to
    this exact point unambiguously instead of guessing by coordinate
    distance - the projected boundary of a long/tall opening commonly lands
    well outside any fixed pixel tolerance from the host chain's own
    skeleton-pixel endpoint (rules 75/78/82/83: host every opening on wall
    topology)."""

    def directions(self) -> set[Direction]:
        return {a.direction for a in self.attachments}

    def attachment_of(self, attachment_type: AttachmentType) -> Optional[Attachment]:
        for a in self.attachments:
            if a.type == attachment_type:
                return a
        return None


@dataclass
class GraphEdge:
    """A final wall/window/door-origin connection between two GraphPoints
    (spec_v008 SS12)."""

    id: str
    edge_type: EdgeType
    point_a_id: str
    point_b_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    source_component_ids: list[int] = field(default_factory=list)
    length_mm: Optional[float] = None
    thickness_px: Optional[float] = None

    @property
    def length_px(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return (dx * dx + dy * dy) ** 0.5


@dataclass
class ComponentRecord:
    """Cleaned connected-component evidence for one class (spec_v008 SS8).

    ``rect_size`` is ``(long_axis_px, short_axis_px)`` from
    ``cv2.minAreaRect`` - the long axis is used as door-origin/window length
    evidence, the short axis as wall-thickness evidence. ``skeleton_points``
    and ``endpoints`` are only populated for line-like classes (wall) where
    point_detection.py needs them to build the skeleton graph.
    """

    class_name: str
    component_id: int
    area_px: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    centroid: tuple[float, float]
    rect_size: Optional[tuple[float, float]] = None
    rect_angle: Optional[float] = None
    skeleton_points: list[tuple[int, int]] = field(default_factory=list)
    endpoints: list[tuple[int, int]] = field(default_factory=list)
    mask: Optional[np.ndarray] = None


@dataclass
class RejectedEvidence:
    """Evidence dropped during cleanup or point search - debug/metrics only,
    never surfaced in vector.svg (spec_v008 SS15)."""

    kind: str
    reason: str
    class_name: Optional[str] = None
    bbox: Optional[tuple[int, int, int, int]] = None
    centroid: Optional[tuple[float, float]] = None
    component_id: Optional[int] = None


@dataclass
class ValidationIssue:
    """A point/graph invariant violation (spec_v008 SS10/SS17.9) - rejects
    only the affected component/region, never the whole graph."""

    rule: str
    message: str
    related_ids: list[str] = field(default_factory=list)
    severity: str = "error"


@dataclass
class DoorCandidateRecord:
    """Per-red-``door_arc``-cluster door inference report (task13 "Metrics
    Requirements"). One record per accepted red cluster - makes it obvious
    whether the cluster became a door and how its hinge/end points were
    inferred (forced-fallback vs. real orange/purple pairing)."""

    red_component_id: int
    red_bbox: tuple[int, int, int, int]
    red_bbox_long_edge_px: float
    created_door_candidate: bool
    scale_candidate_px_to_mm: Optional[float] = None
    hinge_candidate_support_classes: list[str] = field(default_factory=list)
    end_candidate_support_classes: list[str] = field(default_factory=list)
    hinge_distance_to_red_bbox_mm: Optional[float] = None
    end_distance_to_red_bbox_mm: Optional[float] = None
    door_confidence: float = 0.0
    door_inference_notes: str = ""
    # task17 "Required Metrics": the bbox-vertex selection itself, reported
    # independently of whether a door was ultimately created from it.
    all_four_bbox_vertices: dict[str, tuple[float, float]] = field(default_factory=dict)
    selected_hinge_vertex: Optional[tuple[float, float]] = None
    selected_end_vertex: Optional[tuple[float, float]] = None
    hinge_vertex_score: Optional[int] = None
    end_vertex_score: Optional[int] = None
    selected_bbox_edge: Optional[str] = None
    host_wall_alignment_score: Optional[int] = None
    door_width_mm: Optional[float] = None


@dataclass
class MaskToVectorResult:
    """Every intermediate artifact of one run, exposed for tests
    (spec_v008 SS6: "expose intermediate artifacts for tests")."""

    decoded_masks: dict[str, np.ndarray] = field(default_factory=dict)
    components: dict[str, list[ComponentRecord]] = field(default_factory=dict)
    rejected_evidence: list[RejectedEvidence] = field(default_factory=list)
    scale_info: Any = None
    raw_points: list[GraphPoint] = field(default_factory=list)
    point_validation: list[ValidationIssue] = field(default_factory=list)
    aligned_points: list[GraphPoint] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    graph_validation: list[ValidationIssue] = field(default_factory=list)
    walls: list[Any] = field(default_factory=list)
    windows: list[Any] = field(default_factory=list)
    door_origins: list[Any] = field(default_factory=list)
    door_leaves: list[Any] = field(default_factory=list)
    door_arcs: list[Any] = field(default_factory=list)
    door_candidates: list[DoorCandidateRecord] = field(default_factory=list)
    svg: Optional[str] = None

    @property
    def validation_issues(self) -> list[ValidationIssue]:
        return self.point_validation + self.graph_validation
