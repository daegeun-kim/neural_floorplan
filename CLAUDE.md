# Safety Constraints (Strict)

Scope:
- Only operate within the current project root directory.
- Do NOT read, modify, or access parent directories or external folders.

File Operations:
- Only edit files inside this repository.

Environment:
- You may install Python libraries.
- Only install libraries necessary for the current spec.
- Always list packages before installing and wait for approval.
- Use pyenv and poetry for libraries management

Execution:
- Before running any shell command:
  1. Show the command
  2. Explain why it is needed
  3. Wait for approval

General:
- Do NOT go beyond the current spec.
- Do NOT introduce additional tools, frameworks, or datasets unless requested.

# Project Instructions

Project:
Neural Floor Plan to Classified CAD

Goal:
Convert controlled raster floor plans or color-coded sketches into semantic masks, then into clean classified CAD-like vector geometry.

Pipeline:
1. Dataset loading
2. SVG/raster preprocessing
3. Semantic mask generation
4. Sketch-style augmentation
5. Segmentation model training
6. Evaluation
7. Mask-to-vector post-processing
8. Classified JSON export

Rules:
- Follow specs in /specs.
- Work one spec version at a time.
- Do not implement beyond the active spec.
- Before coding, create a plan.
- After coding, run tests.
- Use feature branches.
- Commit only after successful test/lint.
- Keep experiment outputs out of Git unless explicitly approved.

Environment:
- Python
- PyTorch
- Hugging Face Transformers
- OpenCV
- Shapely
- pytest                                                                                                    

Commands:
- Create env: conda create -n floorplan-cad python=3.11
- Activate env: conda activate floorplan-cad
- Install package: pip install -e or conda install depending on reliability
- Test: pytest
- Format: ruff format .
- Lint: ruff check .