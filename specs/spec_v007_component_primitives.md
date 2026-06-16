# Spec v007: Component Primitives for Neural Floorplan

## 0. Purpose

This document defines the reusable CAD-like component primitive system used after CNN segmentation.

The purpose of this stage is not to train a neural network and not to perform full mask-to-vector reconstruction. It defines the clean geometric building blocks that the next stage can place, scale, rotate, and export.

Pipeline position:

```text
CNN prediction
→ component primitive definitions   ← this spec
→ mask-to-vector reconstruction     ← spec v008
→ debug SVG output
```

## 1. Scope

This spec covers:

```text
- primitive object definitions
- primitive parameters
- allowed transformations
- SVG drawing behavior
- folder/module organization under src/
- extensibility for later component variants
```

This spec does not cover:

```text
- CNN training
- CNN evaluation
- mask cleanup
- object detection from masks
- opening classification from masks
- final topology validation
- room adjacency graph construction
- DXF export
- Grasshopper integration
```

## 2. Design Principle

The CNN output is expected to be noisy but spatially meaningful. Therefore, the vector output should not directly trace noisy mask contours.

Instead, the system should reconstruct the plan using clean parametric primitives.

```text
noisy semantic mask
→ rough evidence
→ clean primitive parameters
→ generated CAD-like geometry
```

A primitive is not a static SVG block. It is a code-defined parametric object that can generate SVG geometry when given position, scale, rotation, and other parameters.

## 3. Active Semantic Classes

The CNN currently predicts 5 classes:

| ID | Class |
|---:|---|
| 0 | background |
| 1 | wall |
| 2 | opening |
| 3 | room |
| 4 | icon |

Spec v007 does not change the CNN class count.

No new CNN classes should be introduced here.

## 4. Primitive Types

Initial primitive types:

```text
WallPrimitive
OpeningPrimitive
DoorPrimitive
WindowPrimitive
RoomPrimitive
```

Icon output is ignored in the first vectorization stage even though the CNN still predicts the icon class.

## 5. Static vs Parametric Blocks

Do not store doors, windows, or walls as fixed SVG template files as the source of truth.

Use code-defined parametric primitives.

SVG is an output format, not the primitive definition format.

Acceptable:

```text
DoorPrimitive(width, hinge_point, orientation, swing_side)
→ generate SVG line + quarter-circle arc
```

Not preferred:

```text
door.svg
→ scale/rotate pasted SVG block
```

Reason:

```text
- every component has different position
- wall-hosted objects need accurate alignment
- width and rotation must be computed from the predicted plan
- later variants are easier to support through class inheritance or parameters
```

## 6. Coordinate and Unit System

The preferred output coordinate system is real-world units.

However, a raster image alone cannot reliably reveal absolute scale. The system must use a scale resolver.

Scale priority:

```text
1. explicit scale metadata, if available
2. SVG-derived dataset metadata, if available
3. dimension labels, if available and parsed elsewhere
4. standard architectural assumptions, such as typical door width
5. fallback to pixel units with scale = 1.0
```

For the first implementation, support both:

```text
pixel coordinates internally
real-world scale factor if known
```

The primitive classes should store:

```text
x, y coordinates in working units
scale metadata
unit label: "px", "mm", "m", or "unknown"
```

If no reliable scale is available, export SVG in pixel-space and record that scale is unknown.

## 7. Primitive Transformation Rules

Each primitive must define its allowed degrees of freedom.

| Primitive | Allowed transformation | Not allowed initially |
|---|---|---|
| WallPrimitive | translate, rotate, length scale, thickness parameter | arbitrary bending |
| OpeningPrimitive | translate along host wall, 1D width scale, rotate with host wall | free-floating placement |
| DoorPrimitive | translate, rotate, width parameter, swing side parameter | arbitrary shape deformation |
| WindowPrimitive | translate along host wall, 1D width scale, rotate with host wall | arbitrary shape deformation |
| RoomPrimitive | polygon simplification and snapping | free-form decorative curves |

## 8. WallPrimitive

### 8.1 Purpose

Represents a clean wall as a CAD-like linear object.

### 8.2 Internal representation

Use centerline + thickness as the main representation.

```text
WallPrimitive:
    id
    start: (x, y)
    end: (x, y)
    thickness
    orientation_angle
    confidence
```

Reason:

```text
- easier to fit from skeletonized wall masks
- easier to extend, merge, and snap
- easier to host openings
- can generate wall polygons later
```

### 8.3 SVG generation

For SVG preview, generate either:

```text
- a thick line using stroke-width = thickness
- or a rectangle/polygon generated from centerline and thickness
```

The first implementation may use thick SVG strokes for simplicity.

## 9. OpeningPrimitive

### 9.1 Purpose

Represents a generic interruption or hosted opening on a wall.

Because the CNN predicts one `opening` class, all predicted openings should first become `OpeningPrimitive` candidates before optional door/window classification.

### 9.2 Internal representation

```text
OpeningPrimitive:
    id
    host_wall_id
    center: (x, y)
    width
    orientation_angle
    start: (x, y)
    end: (x, y)
    confidence
    opening_type: "generic" | "door_candidate" | "window_candidate"
```

### 9.3 Required rule

An opening should be hosted by a wall.

Do not generate free-floating openings unless explicitly marked as unresolved for debugging.

## 10. DoorPrimitive

### 10.1 Purpose

Generates a clean CAD-like hinged door symbol from an opening candidate.

### 10.2 Initial door type

Only one door primitive is required in v007:

```text
single hinged door
```

No sliding door, double door, folding door, or variant library is required yet.

However, the primitive structure must allow later variants.

Recommended extension path:

```text
DoorPrimitive
  HingedDoorPrimitive
  SlidingDoorPrimitive
  DoubleDoorPrimitive
```

### 10.3 Internal representation

```text
DoorPrimitive:
    id
    host_wall_id
    hinge_point: (x, y)
    width
    orientation_angle
    swing_direction
    confidence
```

### 10.4 Shape rule

The generated door symbol should preserve exact CAD shape logic:

```text
- straight door leaf line
- quarter-circle swing arc
- hosted opening segment
```

The door symbol may be scaled by width, translated, and rotated, but its geometric logic should not be deformed.

## 11. WindowPrimitive

### 11.1 Purpose

Represents a clean linear window hosted by a wall.

### 11.2 Initial window type

Only one generic window primitive is required in v007.

### 11.3 Internal representation

```text
WindowPrimitive:
    id
    host_wall_id
    center: (x, y)
    width
    orientation_angle
    confidence
```

### 11.4 Shape rule

A window should behave as a 1D scalable wall-hosted object.

Allowed:

```text
- width scaling along wall
- rotation with wall
- translation along host wall
```

Not allowed:

```text
- arbitrary free-form deformation
```

## 12. RoomPrimitive

### 12.1 Purpose

Represents a room or space region as a simplified polygon.

### 12.2 Internal representation

```text
RoomPrimitive:
    id
    polygon: [(x, y), ...]
    area
    confidence
```

Room type classification is not required in v007.

### 12.3 Shape rule

Room polygons may be simplified and snapped to wall geometry in v008.

Room adjacency and full topology validation are not part of v007.

## 13. Icon Handling

The CNN still predicts `icon`, but v007 primitives do not need to support furniture or fixture icons.

The first mask-to-vector implementation should ignore icon pixels.

Future option:

```text
IconPrimitive:
    bbox
    contour
    generic symbol type
```

But this is not required now.

## 14. Door vs Window Concern

The current CNN does not separate doors and windows. Both are included in the `opening` class.

Recommendation:

```text
Do not retrain the CNN to 6 classes yet.
```

Reason:

```text
- current CNN is already stable
- vector stage can first test whether geometric heuristics are sufficient
- retraining should be considered only if heuristic separation is consistently unreliable
```

Initial heuristic direction:

```text
long, narrow, wall-aligned opening → window_candidate
compact opening with aspect ratio closer to 1 → door_candidate
uncertain case → generic opening
```

The primitive library must support DoorPrimitive and WindowPrimitive, but v008 may choose to output GenericOpening when classification confidence is low.

## 15. Folder Structure

All code should stay under `src/`.

Recommended structure:

```text
src/
  primitives/
    __init__.py
    base.py
    wall.py
    opening.py
    door.py
    window.py
    room.py
    svg.py
```

Alternative acceptable structure:

```text
src/vectorization/primitives/
```

Choose one structure and keep it consistent. Since v008 belongs to vectorization, the recommended final structure is:

```text
src/vectorization/
  primitives/
    __init__.py
    base.py
    wall.py
    opening.py
    door.py
    window.py
    room.py
```

## 16. SVG Output Contract

Each primitive should support conversion to SVG elements.

Minimum methods:

```text
to_svg()
bounds()
transform()
```

The SVG output should support debug layers:

```text
WALL
OPENING
DOOR
WINDOW
ROOM
```

The first milestone only requires SVG export.

JSON export is not required in v007.

## 17. Acceptance Criteria

v007 is complete when:

```text
1. Primitive classes exist under src/vectorization/primitives/.
2. WallPrimitive can generate SVG geometry.
3. OpeningPrimitive can generate SVG geometry.
4. DoorPrimitive can generate a clean hinged door symbol.
5. WindowPrimitive can generate a clean wall-hosted window symbol.
6. RoomPrimitive can generate SVG polygon geometry.
7. Primitive transformations are parametric, not static SVG pasting.
8. Icon class is ignored without breaking the system.
9. The primitive system is extensible for later component variants.
```

## 18. Non-Goals

v007 does not:

```text
- detect objects from CNN masks
- clean noisy masks
- classify openings from masks
- enforce room topology
- produce final CAD output from a full prediction image
- retrain the CNN
- add new semantic classes
```

Those belong to v008 or later specs.
