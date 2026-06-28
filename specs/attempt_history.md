# Attempt History: CNN Training and Vectorization

## 0. Purpose

This document records the major attempts, failures, and direction changes made so far in the neural floorplan project.

The focus is on two parts of the workflow:

1. CNN semantic segmentation training
2. vectorization from CNN prediction to architectural SVG

The goal is to preserve why each phase existed, what it tried to solve, why it was not enough, and what changed next.

## 1. Overall Workflow

The intended project workflow is:

```txt
CubiCasa SVG / raster data
-> semantic mask generation
-> CNN segmentation training
-> CNN prediction masks
-> vectorization
-> clean architectural SVG / CAD-like output
```

The project does not try to train a model that directly outputs SVG. Instead, the CNN predicts semantic raster classes, and a separate vectorization process converts those predicted pixels into architectural geometry.

This separation created two major development tracks:

- improving what the CNN predicts
- improving how predicted pixels become vectors

Most failures so far came from the gap between those two tracks. A CNN prediction can look semantically reasonable but still be hard to convert into correct walls, windows, and doors if the vectorization logic is not structured well.

## 2. Phase 1: Initial 5-Class Segmentation

### Intention

The first segmentation approach used a simple semantic class structure:

```txt
background
wall
opening
room
icon
```

The intention was to keep the CNN task broad and simple. Instead of making the model learn many architectural details, the model would predict general regions, and vectorization would later interpret those regions.

This phase assumed:

- walls could be recovered from the `wall` class
- openings could be handled as one generic `opening` class
- rooms/floor could help define spatial regions
- icons/furniture could be separated from architecture

### What Changed In CNN Training

The CNN was trained as a semantic segmentation model with five output classes.

The label-generation process grouped architectural features into broad categories. Doors and windows were not separated enough for downstream geometry. Door origin, door leaf, and door arc were not available as distinct evidence.

### Failure

The 5-class CNN output was not good enough for vectorization.

The important realization was that generic `opening` information was too weak. Openings are not just visual gaps. They are spatial constraints:

- windows must replace a wall segment
- doors must have a hinge, origin, leaf, and swing side
- openings determine where wall topology is interrupted
- doors and windows must be hosted by walls

A generic `opening` class did not provide enough information to identify door count, door direction, hinge location, or window endpoints.

### Resulting Direction Change

The project moved away from broad semantic classes and toward more vectorization-aware classes.

The key lesson was:

```txt
The CNN class design must support vectorization, not only visual segmentation.
```

## 3. Phase 2: 7-Class Segmentation With Door and Window Evidence

### Intention

The second major CNN phase added more classes for openings.

The active 7-class scheme became:

```txt
0 background
1 floor
2 wall
3 window
4 door_arc
5 door_leaf
6 door_origin
```

The intention was to make the CNN output directly useful for vectorization.

Instead of asking vectorization to infer all door/window structure from a generic opening region, the CNN would predict:

- where walls are
- where windows are
- where door swing arcs are
- where door leaves are
- where door origins/thresholds are

This was expected to make door and window reconstruction more reliable.

### What Changed In CNN Training

The segmentation target changed from five broad classes to seven vectorization-oriented classes.

Specific CNN label changes:

- `opening` was removed
- `room` became `floor`
- `icon` / furniture supervision was removed
- windows became their own class
- doors were split into three classes:
  - `door_origin`
  - `door_leaf`
  - `door_arc`

The door labels were designed for geometry:

- `door_origin` should identify the wall-aligned threshold segment
- `door_leaf` should identify the opened panel direction
- `door_arc` should identify the swing area and help count doors

The training run identity became:

```txt
segformer_b0_run3
```

This required:

- new semantic masks
- new output class count
- new class palette
- new training config
- new feature cache
- new checkpoint/output directories

### What Improved

The CNN output became more semantically useful than the 5-class output.

The model now produced separate evidence for:

- wall
- window
- door arc
- door leaf
- door origin

This made it possible to reason about door count and door geometry in principle.

### Failure

The main problem became vectorization, not only CNN prediction.

Even with better semantic classes, the vectorizer still had to convert noisy raster evidence into clean architectural geometry. The 7-class output did not automatically solve:

- wall topology
- wall thickness
- window endpoint pairing
- door hinge pairing
- door-origin length normalization
- scale estimation
- avoiding contour-like walls
- avoiding duplicated or disconnected wall segments

The CNN could provide useful evidence, but the vectorizer still needed strong architectural rules.

The key failure was:

```txt
Better semantic classes did not guarantee correct vector geometry.
```

### Resulting Direction Change

The project shifted from "improve the CNN classes until vectorization works" to "redesign vectorization so it uses CNN classes as evidence, not as final geometry."

The 7-class CNN remains useful, but vectorization needs its own graph logic.

## 4. Phase 3: Early Vectorization From Semantic Classes

### Intention

The first vectorization attempts used the predicted semantic classes directly.

The rough intention was:

```txt
wall pixels -> wall vectors
window pixels -> window vectors
door pixels -> door vectors
floor pixels -> floor fill
```

This approach treated class masks as the main geometric source.

### What Changed In Vectorization

Vectorization started to include separate logic for:

- outer walls
- inner walls
- hosted windows
- door origins
- door leaves
- door arcs
- floor output

The output goal became a clean SVG with:

- black walls
- blue windows
- purple door origins
- orange door leaves
- red door arcs
- optional white floor

### Failure

The early vectorization output did not match architectural expectations.

Observed problems included:

- walls looked like raster contours instead of architectural geometry
- wall thickness was inconsistent or visually wrong
- outer walls could be derived from floor/background borders
- inner walls could disappear or duplicate outer walls
- windows and doors could fail to connect cleanly to wall endpoints
- door arcs could be reversed or poorly placed
- debug-like elements could appear in final SVG
- scale assumptions could make geometry too thick or too thin

The major issue was that semantic masks are not the same thing as architectural topology.

For example:

- a wall mask blob is not automatically a wall graph
- a door arc mask is not automatically a door primitive
- a window mask component is not automatically two correct window endpoints
- a floor/background boundary is not reliable wall evidence

### Resulting Direction Change

The vectorization strategy moved toward stronger architectural primitives:

- wall centerlines
- connected wall curves
- hosted openings
- replacement of wall spans by windows/door origins
- procedural door leaf and arc generation

This improved the conceptual structure, but the implementation still struggled because it was mostly segment/component driven rather than point-graph driven.

## 5. Phase 4: Vectorization Refinement Around Walls and Openings

### Intention

After the first vectorization failures, the goal became to repair specific geometry problems without throwing away the whole process.

The intention was to make the output more architectural by enforcing:

- wall polygons instead of stroke-thickness lines
- connected wall curves before polygon generation
- clean wall joins
- hosted windows and doors
- door arcs centered on hinge points
- no debug elements in final SVG

### What Changed In Vectorization

Several rules were added:

- walls and windows should be closed filled polygons
- door origin, door leaf, and door arc should be thin symbolic SVG elements
- wall centerline segments sharing endpoints should merge before buffering
- outer wall evidence should not be duplicated as inner wall evidence
- inner wall branches connected to the outer wall should be preserved
- door count should be driven by red `door_arc` components
- door origin/leaf length should snap to `700 mm` or `900 mm`
- window minimum width should be `300 mm`
- metric rules should not silently fall back to arbitrary pixel values

### Failure

This refinement still did not fully solve vectorization.

The process remained too dependent on class components and wall extraction heuristics. It still had difficulty with:

- scale estimation from noisy components
- wall thickness measurement from connected blobs
- recognizing correct topological junctions
- pairing windows and doors reliably
- explaining debug output clearly

The recent sample-level issue showed this clearly:

- scale could remain unknown because door and wall measurements conflict
- wall thickness could become visually too thick because component measurements are not true local wall thickness
- debug overlay points could be hard to interpret without a legend

### Resulting Direction Change

The project moved toward a full restart of vectorization.

The key lesson was:

```txt
Vectorization should not start from wall/floor/object classes as final things.
It should start by identifying architectural points, then connect them.
```

## 6. Phase 5: Vectorization Restart as Orthogonal Point Graph

### Intention

The current vectorization restart changes the main representation.

Instead of directly converting semantic classes into wall/floor/window/door vectors, the new process searches for architectural points first.

The seven allowed point types are:

```txt
1_wall_point
2_wall_point
3_wall_point
4_wall_point
wall_window_point
wall_door_hinge_point
wall_door_end_point
```

The intention is:

```txt
detect points
-> align points orthogonally
-> connect compatible points
-> generate walls, windows, and doors from graph edges
```

This makes topology explicit.

### What Changed In Vectorization

The active v008 process became:

```txt
7-class prediction
-> class masks
-> connected components
-> direct search for seven point types
-> point alignment
-> point connection
-> wall/window/door-origin graph
-> procedural door leaf and arc
-> SVG export
```

Important changes:

- floor generation is excluded for now
- final geometry is orthogonal only
- 45-degree walls are no longer supported
- every final point must be one of the seven point types
- door existence is driven by red `door_arc` evidence
- purple/orange door evidence without red arc is rejected
- missing/noisy purple door-origin evidence may be inferred when red arc evidence exists
- `1_wall_point` is a valid free wall end and should not be forcibly connected
- windows and doors must be represented as point pairs before final edges are created

### Why This Direction Is Different

The previous vectorization approach asked:

```txt
What object does this class component represent?
```

The new approach asks:

```txt
What architectural point is this evidence supporting?
Which other point should it connect to?
```

This is a major shift.

The output is no longer expected to emerge from raw class regions. It should emerge from a clean graph of endpoints, corners, T-junctions, window endpoints, and door endpoints.

### Current Known Risks

The point-graph restart is more promising, but it still has unresolved technical risks:

- noisy CNN predictions can create fragmented components
- scale estimation can fail when wall and door measurements disagree
- wall thickness must be measured locally, not from large connected-component bounding boxes
- debug overlays must clearly explain point types and rejected evidence
- point detection rules must avoid overfitting to one sample shape
- window and door pairing must be robust to missing or imperfect pixels

### Latest Door Recognition Refinement

The latest work within this same point-graph phase focuses on door recognition.

The current implementation still treats red `door_arc` clusters too much like optional evidence. A red cluster may be used for scale, but then the same cluster can be rejected later if purple/orange pairing, endpoint distance, or origin span checks fail.

The revised intention is stricter:

```txt
connected red door_arc cluster = door object
```

Once a red cluster is detected, it should always create one door candidate. The red cluster determines:

- door existence
- door count
- door location
- scale inference
- hinge/end search region

The vectorizer should then infer:

- one `wall_door_hinge_point`
- one `wall_door_end_point`

The hinge point should be selected by highest combined proximity to:

```txt
red + orange + purple + black
```

The end point should be selected by highest combined proximity to:

```txt
red + purple + orange
```

If one evidence type is missing, the point should still be inferred from the strongest available subset. Weak or fragmented orange/purple evidence should reduce confidence, not delete the door.

This is still part of the current Phase 5 point-inference direction. It does not change the overall vectorization concept. It clarifies that door objects begin from red clusters first, and point inference follows from that door object.

## 7. Phase 6: Settled Raster-To-Graph Inference

### Intention

The next vectorization direction after the point-graph restart was to use the external Raster-to-Graph checkpoint instead of continuing to hand-build wall topology from semantic masks.

The input should be:

```txt
model_clean.png
```

The original possible supervised graph target was:

```txt
masks/wall_graph.json
```

The first direct inference attempts often produced either a very accurate graph or no graph at all. The project therefore shifted from training first to adapting the inference process around this dataset.

### What Changed In Phase 4

```txt
model_clean.png
-> content bbox crop
-> true 20% white padding around content
-> long edge scaled to 512 px
-> centered on a white 512x512 canvas
-> original Raster-to-Graph mean/std normalization
-> pretrained checkpoint0299.pth
-> generous autoregressive inference
-> hard/soft validity scoring and reranking
-> mask-and-rerun multistart recovery
-> merge-on-intersection
-> light post-merge cleanup
```

Important local changes:

- standardized input as `crop512_margin20_truepad`
- lowered graph-generation thresholds enough to reduce empty outputs
- increased pixel tolerance so geometry is judged by architectural logic, not pixel-perfect matching
- added hard angle filtering to keep only near-horizontal or near-vertical edges
- added soft scoring for wall evidence, rectangle cycles, dangling penalties, and unsupported-edge penalties
- added mask-and-rerun multistart so disconnected graph regions can be generated
- merged components by node snapping, H/V intersection insertion, edge splitting, and collinear merge
- reduced final filtering so valid fragments created during merge are not deleted

### Current Status

The current Raster-to-Graph method is satisfactory enough that no fine-tuning is needed for now. Fine-tuning remains a future fallback, not the active plan.

Current output organization:

```txt
outputs/vectorization/phase4_raster2graph_generous_inference/<sample>/
```

Each sample folder contains the preprocessed input, predicted graph JSON/SVG, overlays, metrics, and component diagnostics. The testing-only `outputs/raster2graph/` folder is retired.

## 8. Summary Of Intention Changes

| Phase | CNN Intention | Vectorization Intention | Main Failure | Resulting Change |
|---|---|---|---|---|
| 5-class segmentation | predict broad architectural classes | infer geometry from broad masks | openings were too generic | split openings into richer classes |
| 7-class segmentation | predict vectorization-useful wall/window/door evidence | convert richer masks into vectors | geometry still failed because topology was implicit | redesign vectorization rules |
| early semantic vectorization | keep class-to-object mapping simple | convert masks into walls/floor/openings | output looked contour-like and disconnected | add architectural primitive rules |
| wall/opening refinement | keep 7-class CNN | fix wall connectivity and hosted openings | component heuristics still unstable | restart vectorization from graph points |
| point-graph restart | keep 7-class CNN as evidence source | detect points, align, connect | point/scale detection remained brittle | move to Raster-to-Graph wall graph prediction |
| Raster-to-Graph inference | use clean SVG-rendered raster as graph-model input | adapt pretrained checkpoint with preprocessing, thresholds, multistart, scoring, and merge cleanup | direct checkpoint inference was too often empty | settled on generous inference from `model_clean.png`; no fine-tuning needed for now |

## 9. Current Project Position

The current best understanding is:

```txt
The CNN should predict semantic evidence.
Raster-to-Graph should produce the wall topology.
The vectorizer/CAD stage should attach classification and openings later.
```

The CNN is not expected to solve wall graph vectorization by itself.

The active 7-class CNN is still useful because it provides wall/window/door evidence. However, the current Phase 4 wall graph comes from the pretrained Raster-to-Graph inference pipeline, not from semantic-mask point extraction.

The current vectorization direction is therefore:

```txt
model_clean.png
-> pretrained Raster-to-Graph inference
-> orthogonal wall graph
-> attach door/window evidence from the 7-class segmentation output
-> clean CAD-like SVG / JSON
```

This is the current settled direction. The earlier semantic and point-graph attempts remain important history because they explain why the project moved toward graph prediction.

### Latest Phase 4 Graph-To-Vector Update

The next implementation target is documented in:

```txt
specs/spec_v008_phase4_vectorization.md
```

The updated Phase 4 vectorization process uses both outputs:

```txt
Raster-to-Graph output -> clean wall centerline topology
7-class segmentation   -> scale, doors, windows, and debug evidence
```

Important decisions:

- door/window endpoints are inferred by proximity to the aligned R2G graph, not by black wall pixels alone
- the two endpoints of one door/window must host on the same wall edge or same wall chain interval
- openings are inserted and trimmed in centerline graph space before wall polygons are generated
- remaining wall centerlines are connected into graph chains before buffering, so corners and junctions render cleanly
- final wall thickness is 200mm when scale is resolved, implemented as 100mm buffering on each side of the centerline

This update keeps the successful R2G wall topology work while reusing the 7-class segmentation model only where it is strongest: semantic opening evidence and scale inference.

### Task32 Door Primitive Regression

The first Phase 4 graph-to-vector attempt produced useful wall/window output, but revealed a door primitive regression in:

```txt
outputs/vectorization/phase4_vectorization/1316/final_vector.svg
```

The failure is not door detection itself. The hosted door origin edge exists, but Phase 4 exports it with the wrong primitive semantics:

```txt
current wrong behavior:
  hosted door origin edge is drawn as the red door leaf
  door origin is reduced to a purple circle
  door arc starts at the hinge point

required behavior:
  door_origin = purple line along the hosted wall gap
  door_leaf   = orange perpendicular line from hinge
  door_arc    = red 90-degree arc centered on hinge,
                from origin far point to leaf endpoint
```

This is a regression because Phase 3 already had the correct primitive contract in:

```txt
src/vectorization/primitives/door.py
```

Task32 instructs Claude to reuse or exactly match the existing `DoorOriginPrimitive`, `DoorLeafPrimitive`, and `DoorArcPrimitive` behavior instead of keeping Phase 4's simplified local door drawing.

### Task33 Opening Interval De-Overlap Rule

The next Phase 4 refinement concerns overlapping trim intervals when valid doors and windows sit very close on the same wall segment.

The first idea of preserving the higher-confidence opening and rejecting the lower-confidence one was rejected because the 7-class raster usually detects both openings correctly. Slight overlap is usually a vector-placement problem, not evidence that one opening is invalid.

Updated rule:

```txt
door vs window:
  keep the door fixed
  move or shrink the window away from the door

door vs door:
  keep the higher-confidence door fixed
  move or shrink the lower-confidence door

window vs window:
  keep the higher-confidence window fixed
  move or shrink the lower-confidence window
```

Openings should be rejected only when no feasible non-overlapping interval exists on the host wall chain. Final wall trimming must use the adjusted non-overlapping intervals, and `final_vector.json` should record original and adjusted interval positions for debugging.

### Task34 Pipeline Contract Enforcement

After Task33, the vector output barely changed in the problem areas because the implementation only partially followed the intended process.

Observed issues:

```txt
1. wall trimming used adjusted opening intervals,
   but final window/door primitives still used the original hosted snapped_points

2. door primitive shape contract was fixed,
   but hinge and swing direction still used fallback values:
     hinge_source = fallback_pt0
     swing_source = fallback
     swing_side = fallback_left

3. some buffered wall corners stayed disconnected,
   indicating the pre-buffer wall graph still contains endpoints that are visually close
   but not topologically snapped into shared nodes
```

Task34 therefore requires the adjusted opening geometry to become the single source of truth for wall trim, SVG, JSON, debug overlay, and notebook output. It also requires evidence-driven door direction from the 7-class raster:

```txt
red door_arc pixels    -> primary swing-side / arc quadrant evidence
orange door_leaf pixels -> secondary hinge/leaf support
purple door_origin pixels -> origin edge validation
```

The Phase 4 notebook is now treated as part of the required deliverable for every future vectorization behavior change, because it is the manual sample-testing surface.
