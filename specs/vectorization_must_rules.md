# Vectorization Must Rules

These rules assume the vectorizer is given an already segmented 7-class raster image. The job of vectorization is to convert that raster evidence into clean architectural vector output plus a debug overlay/metrics that explain the decisions.

The rules are written so they can be checked from `vector.svg`, `debug_overlay.png`, and `metrics.json`.

## Input Assumption

1. The vectorizer must start from an already segmented raster with these semantic classes:
   - background
   - floor
   - wall
   - window
   - door_arc
   - door_leaf
   - door_origin
2. The raster classes are evidence. They must not be copied as noisy pixel contours into the final vector output.
3. Floor/background pixels must not define wall linework or the outer wall.
4. Wall, window, door_arc, door_leaf, and door_origin pixels must be the main evidence for wall/opening/door vectorization.

## Scale And Dimensions

5. Scale must always be inferred from connected red `door_arc` pixel clusters.
6. The long edge of each red `door_arc` cluster bounding box must be treated as a candidate real door width.
7. Each red cluster must evaluate exactly these scale candidates:
   - `px_to_mm = 700 / red_cluster_bbox_long_edge_px`
   - `px_to_mm = 900 / red_cluster_bbox_long_edge_px`
8. If multiple red clusters exist, the vectorizer must choose the globally most consistent `px_to_mm` by robust voting/clustering and use the median of the winning group.
9. Obvious red-cluster scale outliers may be rejected, but ordinary noisy red clusters must still be reported in debug/metrics.
10. Door-origin width must be either `700 mm` or `900 mm`.
11. Door leaf length must equal the door-origin width.
12. Door arc radius must equal the door-origin width.
13. Wall thickness must be either `100 mm` or `200 mm`.
14. Window width must be at least `300 mm`.
15. Window total thickness must be `100 mm`.
16. Default point merge and axis-alignment tolerance must be `500 mm` (task17: reverted from the `1000 mm` task15 experiment - door-derived axes are the trusted anchors now, so the goal is no longer to over-merge distant geometry by raw distance alone).
17. Door hinge and door end points must each be within `200 mm` of the associated red `door_arc` bounding box.
18. Door-origin and wall-thickness evidence may be used only as checks/debug evidence for scale; they must not override red-bbox scale inference.
19. If no usable red `door_arc` cluster exists, metric scale must be reported as unresolved or scale-blocked rather than invented from other evidence.
20. Pixel units must never be silently labeled as millimeters.

## Point Graph Rules

21. Final geometry must be built from an architectural point graph.
22. Every final detected point must be exactly one of these seven point types:
   - `1_wall_point`
   - `2_wall_point`
   - `3_wall_point`
   - `4_wall_point`
   - `wall_window_point`
   - `wall_door_hinge_point`
   - `wall_door_end_point`
23. There must be no unresolved, generic, or untyped final point category.
24. Every point must store local attachments.
25. Every attachment must have a semantic type: `wall`, `window`, or `door_origin`.
26. Every attachment direction must be cardinal: `left`, `right`, `up`, or `down`.
27. Door leaf and door arc must not be point-search attachment types; they are generated after the door-origin segment is accepted.
28. A `1_wall_point` must be a true free-standing wall end.
29. A `1_wall_point` must not be created where window, door_arc, door_leaf, or door_origin evidence touches or sits near the wall end.
30. A wall end touching blue window evidence must become `wall_window_point`, not `1_wall_point`.
31. A wall end near red/orange/purple door evidence must become `wall_door_hinge_point` or `wall_door_end_point`, not `1_wall_point`.
32. A `2_wall_point` must represent exactly two wall attachments at a 90-degree corner.
33. A `3_wall_point` must represent a T-junction or branch with three wall attachments.
34. A `4_wall_point` must require wall evidence in all four cardinal directions.
35. Every final window must have exactly two compatible `wall_window_point` endpoints.
36. `wall_window_point` endpoints must face each other through window attachments.
37. Every accepted red `door_arc` cluster must produce exactly one `wall_door_hinge_point`.
38. Every accepted red `door_arc` cluster must produce exactly one `wall_door_end_point`.
39. Door hinge count must equal door end count.
40. Door hinge count must equal accepted red `door_arc` cluster count.
1
## Red Door-Arc Rules

41. A connected red `door_arc` pixel cluster must become a door object once it survives component cleanup.
42. Red `door_arc` clusters must define door count.
43. Red `door_arc` clusters must define door location.
44. Red `door_arc` clusters must define the hinge/end search area.
45. Red `door_arc` clusters must drive door-origin, door-leaf, and door-arc generation.
46. Purple `door_origin` and orange `door_leaf` evidence must refine geometry, but must not decide whether a red cluster is a door.
47. A red cluster must not be rejected because purple evidence is fragmented, missing, too short, or noisy.
48. A red cluster must not be rejected because orange evidence is fragmented, missing, or noisy.
49. A red cluster must not be rejected because the snapped `700 mm` or `900 mm` endpoint differs from noisy purple pixels.
50. Weak or missing purple/orange evidence must lower confidence and appear in debug/metrics, not delete the door.
51. A red cluster may be rejected only if it is below the minimum component area, its bounding box aspect ratio exceeds `2:1`, or no plausible nearby wall/door geometry can be inferred after fallback.
52. Purple `door_origin` evidence without a red arc must not create a final door.
53. Orange `door_leaf` evidence without a red arc must not create a final door.
54. If no red `door_arc` evidence exists, no final door may be created.
55. Door hinge inference should prefer the orange/purple intersection when present.
56. If orange/purple intersection is missing, the hinge must be inferred from red arc geometry and nearest plausible host wall.
57. Hinge inference must prefer support from red, orange, purple, and black evidence.
58. Door end inference must prefer support from red, purple, and orange evidence.
59. If one evidence class is missing, hinge/end inference must use the strongest available subset.

## Orthogonality

60. Final wall, window, and door-origin edges must be orthogonal only: `0`, `90`, `180`, or `270` degrees.
61. Final vector output must not contain 45-degree wall geometry.
62. Diagonal or arbitrary-angle evidence must be snapped to the strongest supported orthogonal interpretation or rejected into debug output.
63. Door leaf must be perpendicular to the door-origin segment.
64. Door arc must span exactly `90` degrees.

## Wall Rules

65. Wall construction must use structural/opening evidence, not the floor/background border.
66. Final wall geometry must be black closed filled polygon geometry.
67. Wall centerlines may exist only as internal construction helpers.
68. Walls must not be represented in final SVG as single lines with `stroke-width`.
69. Wall graph edges sharing endpoints must be merged into connected wall geometry before polygon generation.
70. Connected wall geometry must avoid duplicate internal caps at shared vertices.
71. Wall polygon joins must be clean at corners and junctions.
72. Outer-wall evidence must not be duplicated as inner-wall geometry.
73. Inner walls should connect to outer walls or other inner walls when evidence supports connection.
74. Free wall endpoints must be preserved only when source evidence clearly stops.

## Window Rules

75. Every final window must be hosted by wall topology.
76. A window must not float independently from a wall.
77. A window must replace the wall interval at its location.
78. Window endpoints must connect exactly to adjacent wall geometry after graph construction.
79. Window geometry must be a blue closed filled polygon.
80. Window final geometry must not be a raw contour trace or stroke-only line.

## Door Geometry Rules

81. Door output must consist of `door_origin`, `door_leaf`, and `door_arc`.
82. Door origin must be hosted on wall topology.
83. Door origin must replace the wall interval at its location.
84. Door origin must be a thin symbolic purple SVG line.
85. Door origin must not be offset into a wall-like polygon.
86. Door leaf must start at the `wall_door_hinge_point`.
87. Door leaf must be generated procedurally from the door origin.
88. Door leaf must be a thin symbolic orange SVG line.
89. Door leaf must not trace raw orange pixel contours.
90. Door arc must be generated procedurally from the hinge point.
91. Door arc center must be the hinge point where door origin and opened leaf meet.
92. Door arc must connect the closed-door origin direction to the opened leaf direction.
93. Door arc side must follow red `door_arc` and orange `door_leaf` evidence.
94. Door arc must be a thin symbolic red SVG arc.
95. Door arc must not trace raw red pixel contours.
96. SVG arc flags must be computed from hinge, origin-end, and leaf-end geometry so the arc remains centered on the hinge for every orthogonal wall orientation.

## Final SVG Rules

97. Final SVG must contain only these visible groups: `wall`, `window`, and `door`.
98. Required visible drawing order must be `wall`, `window`, then `door`.
99. Required final colors must be:
   - wall: black `#000000`
   - window: blue `#3c78dc`
   - door_origin: purple `#a046b4`
   - door_leaf: orange `#eb8c50`
   - door_arc: red `#dc5a5a`
100. Final SVG must contain no debug group.
101. Final SVG must contain no dashed unresolved marker.
102. Final SVG must contain no unidentified primitive groups.
103. Final SVG must contain no generic `opening` group.
104. Final SVG must contain no free-floating windows.
105. Final SVG must contain no free-floating doors.
106. Final SVG root attributes must include:
   - `data-unit`
   - `data-scale-status`
   - `data-px-to-mm`
   - `data-scale-source`

## Debug Overlay And Metrics Rules

107. Rejected evidence must appear only in metrics, never in the debug overlay image or final SVG (task19: dropped from the overlay - it cluttered the render without being needed to read the final graph; still fully reported in `metrics.json`).
108. Low-confidence or force-inferred door candidates must appear only in debug overlay and metrics, never as debug markers in final SVG.
109. Debug overlay must show searched points by type.
110. Debug overlay must show wall/window/door-origin graph edges.
111. Debug overlay must show every red `door_arc` candidate bbox.
112. Debug overlay must show each red candidate's inferred `wall_door_hinge_point` and `wall_door_end_point`, each in its own final point-type color (orange hinge / purple end - task19: not the candidate's confidence color, so they read the same way every other final point type does).
113. Debug overlay must visually distinguish low-confidence inferred door candidate bboxes from high-confidence ones.
114. Metrics must include scale diagnostics:
   - `red_arc_bbox_long_edges_px`
   - `red_arc_px_to_mm_candidates`
   - `red_arc_selected_modules_mm`
   - `selected_px_to_mm`
   - `scale_source`
   - `scale_rejected_outliers`
115. Metrics must include one door-candidate record per red cluster:
   - `red_component_id`
   - `red_bbox`
   - `red_bbox_long_edge_px`
   - `created_door_candidate`
   - `scale_candidate_px_to_mm`
   - `hinge_candidate_support_classes`
   - `end_candidate_support_classes`
   - `hinge_distance_to_red_bbox_mm`
   - `end_distance_to_red_bbox_mm`
   - `door_confidence`
   - `door_inference_notes`

## Observable Failure Rules

116. The output fails if it traces raw pixel contours as final geometry.
117. The output fails if it exports diagonal or 45-degree wall geometry.
118. The output fails if it creates doors without red arc evidence.
119. The output fails if an accepted red arc cluster does not become a door object.
120. The output fails if final windows or doors are not hosted on wall topology.
121. The output fails if debug or unidentified visible groups appear in `vector.svg`.
122. The output fails if wall thickness is not `100 mm` or `200 mm` when scale is resolved.
123. The output fails if door width is not `700 mm` or `900 mm` when scale is resolved.
