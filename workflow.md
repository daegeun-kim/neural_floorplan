# Neural Floor Plan → Classified CAD (Training Pipeline)

## 0. Scope

Goal:
Controlled raster floor plan (or color-coded sketch)
→ semantic understanding
→ clean, classified CAD-like vector output


Stop before Grasshopper analysis.

---

## 1. Datasets

### 1.1 Primary Dataset (Core)

- :contentReference[oaicite:0]{index=0}  
- https://github.com/cubicasa/cubicasa5k

Use for:
- walls  
- rooms  
- doors  
- windows  
- room types  
- SVG vector ground truth  

---

### 1.2 Secondary Dataset (Topology / Scale)

- :contentReference[oaicite:1]{index=1}  
- https://github.com/ennauata/houseganpp  
- https://github.com/zzilch/RPLAN-Toolbox  

Use for:
- room adjacency  
- layout distribution  
- synthetic raster generation  

---

### 1.3 Optional Dataset (Advanced CAD Detail)

- :contentReference[oaicite:2]{index=2}  
- https://floorplancad.github.io/

Use for:
- detailed symbol detection  
- CAD-like precision  

---

## 2. Data Preparation

### 2.1 Convert SVG → Raster + Masks

From CubiCasa5K:

Input:
- SVG vector plan  

Generate:
- raster image (clean plan)  
- semantic masks:
  - wall  
  - window  
  - door  
  - room boundary  
  - room type  

---

### 2.2 Generate Controlled Hand-Drawn Style Input

Transform clean raster into sketch-like input:

- line jitter (small noise)  
- stroke width variation  
- slight rotation/skew  
- broken edges  
- grayscale or color-coded lines  
- optional paper texture  

Optional color coding:
- wall = red  
- window = blue  
- door = green  
- room = white/black fill  

Output pair:
- Input: sketch-like raster  
- Label: clean semantic masks  

---

## 3. Model Setup

### 3.1 Pretrained Model Options

#### Option A (Recommended)
- :contentReference[oaicite:3]{index=3}  
- https://huggingface.co/docs/transformers/model_doc/segformer  

#### Option B
- U-Net with pretrained encoder (ResNet / EfficientNet)  

#### Option C (AEC-specific)
- :contentReference[oaicite:4]{index=4}  
- https://github.com/zlzeng/DeepFloorplan  

---

### 3.2 Model Structure

Input:
- raster floor plan (H × W × 3)

Backbone:
- pretrained encoder (SegFormer / ResNet)

Head:
- semantic segmentation

Output:
- multi-channel mask:
  - wall  
  - window  
  - door  
  - room  
  - background  
  - (optional room types)  

---

### 3.3 Loss Function

Loss:
- CrossEntropy + DiceLoss  

---

## 4. Training

### 4.1 Input–Output Pair

- X: sketch-like raster image  
- Y: semantic segmentation masks  

---

### 4.2 Training Strategy

- initialize backbone with pretrained weights  
- freeze backbone (optional, early stage)  
- train segmentation head  
- unfreeze and fine-tune full model  

---

### 4.3 Data Augmentation

- rotation (90° only)  
- horizontal/vertical flip  
- brightness/contrast variation  
- line thickness variation  
- noise injection  

---

### 4.4 Output of Training

Model predicts:
- wall mask  
- window mask  
- door mask  
- room mask  
- (optional room type mask)  

---

## 5. Post-Processing → Classified CAD

### 5.1 Mask → Geometry

- extract contours from masks  
- simplify polylines  
- snap to orthogonal grid  
- close polygons  

---

### 5.2 Wall Reconstruction

- convert wall mask → centerlines  
- estimate thickness  
- generate parallel offsets  

---

### 5.3 Door / Window Assignment

- detect line segments from masks  
- attach to nearest wall  
- compute width  

---

### 5.4 Room Reconstruction

- extract room polygons  
- ensure closure  
- assign room type  

---

## 6. Final Output Format (Before GH)

Example JSON:

```json
{
  "walls": [
    {
      "centerline": [[x1,y1],[x2,y2]],
      "thickness": 150
    }
  ],
  "windows": [
    {
      "line": [[x1,y1],[x2,y2]],
      "host_wall_id": 3
    }
  ],
  "doors": [
    {
      "line": [[x1,y1],[x2,y2]],
      "host_wall_id": 5,
      "width": 900
    }
  ],
  "rooms": [
    {
      "polygon": [[...]],
      "type": "bedroom",
      "area": 12.5
    }
  ],
  "adjacency": [
    ["bedroom","living_room"]
  ]
}