# Detection Module

YOLOv8-Nano pedestrian detection with hybrid crosswalk surface classification on JAAD dataset videos.

## Overview

This module detects pedestrians and classifies them into one of four states based on their location and the surface beneath their feet:

| State | Meaning | Color |
|-------|---------|-------|
| **CROSSWALK** | On a detected crosswalk surface | Green |
| **JAYWALKING** | On road without crosswalk markings | Red |
| **SAFE** | On sidewalk | Cyan |
| **STATIC** | Not actively crossing | Gray |

## How It Works

### 1. Pedestrian Detection
YOLOv8-Nano detects pedestrians with two classes:
- `person_crossing` (class 0) — actively crossing
- `person_static` (class 1) — standing still

### 2. Surface Classification
For each crossing pedestrian, a pretrained SegFormer (`nvidia/segformer-b0-finetuned-ade-512-512`) segments the scene into semantic classes (road, sidewalk, etc.).

### 3. Hybrid Crosswalk Detection
A 40x40 pixel patch at the pedestrian's feet is analyzed with two methods:

- **White stripe detection** — Thresholds bright pixels (>150 intensity) to find painted zebra markings
- **Edge texture analysis** — Canny edge detection to identify patterned surfaces (brick pavers, triangles)

A surface is classified as a crosswalk if either the white pixel ratio exceeds `WHITE_THRESH` (default: 0.08) or the edge density exceeds `EDGE_THRESH` (default: 0.05).

### 4. Temporal Smoothing
A 10-frame majority-voting window per tracked pedestrian prevents state flickering.

## Dataset Preparation

Parse JAAD XML annotations into YOLO format with an 80/20 train/val split:

```bash
python parse_jaad.py
```

This reads from `./JAAD/JAAD_clips/` and `./JAAD/annotations/`, and outputs to `./dataset_yolo/` with the standard YOLO directory structure:
```
dataset_yolo/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Training

```bash
python train_yolo.py
```

Configuration (in `train_yolo.py`):
- Model: YOLOv8-Nano (`yolov8n.pt`)
- Image size: 640
- Batch size: 16
- Early stopping patience: 5 epochs
- Device: MPS (Apple Silicon)
- Dataset config: `jaad_mini.yaml`

Best weights are saved to `runs/detect/jaad_yolo_run/weights/best.pt`.

## Inference

```bash
python main.py
```

Edit the configuration at the top of `main.py`:
- `VIDEO_PATH` — path to input video
- `OUTPUT_PATH` — path for annotated output video
- `YOLO_WEIGHTS` — path to trained YOLO weights

### Tunable Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WHITE_THRESH` | 0.08 | Minimum white pixel ratio for stripe detection |
| `EDGE_THRESH` | 0.05 | Minimum edge density for texture detection |

## Dependencies

- PyTorch (MPS)
- Ultralytics (YOLOv8)
- Hugging Face Transformers (SegFormer)
- OpenCV
