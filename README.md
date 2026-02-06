# Jaywalking Detection Pipeline

A three-module deep learning pipeline for pedestrian detection, crossing intention prediction, and crosswalk surface segmentation — built for the EPFL [CIVIL-459: Deep Learning for Autonomous Vehicles](https://edu.epfl.ch/coursebook/en/deep-learning-for-autonomous-vehicles-CIVIL-459) course.

## Pipeline Overview

The three modules form a complementary pedestrian safety system operating on [JAAD dataset](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/) videos:

```
JAAD videos ─┬─► Detection  (YOLOv8 bboxes + surface classification) ─► pedestrian state labels
             │
             └─► Intention  (OpenPifPaf keypoints → DSTformer transformer) ─► crossing probability

FPV images  ──► Segmentation (SegFormer / DeepLab) ─► crosswalk pixel masks
```

## Modules

| Module | Model | Task |
|--------|-------|------|
| [**detection/**](detection/README.md) | YOLOv8-Nano + SegFormer surface classifier | Pedestrian detection and 4-state classification (STATIC, SAFE, CROSSWALK, JAYWALKING) |
| [**intention/**](intention/README.md) | MotionBERT (DSTformer) | Binary crossing intention prediction from 2D pose keypoints |
| [**segmentation/**](segmentation/README.md) | SegFormer-B0 / DeepLabV3+ | Pixel-level crosswalk segmentation on first-person-view images |

## Repository Structure

```
├── detection/
│   ├── main.py            # Full inference pipeline
│   ├── parse_jaad.py       # JAAD XML → YOLO format conversion
│   ├── train_yolo.py       # YOLOv8-Nano training
│   ├── jaad.yaml           # YOLO dataset config
│   └── jaad_mini.yaml      # Mini dataset config
├── intention/
│   ├── train.py            # Training / evaluation entry point
│   ├── inference.py        # Video inference with OpenPifPaf
│   ├── dataset.py          # JAAD dataset generation
│   ├── configs/            # YAML training & inference configs
│   └── lib/                # Model (DSTformer), data, and utilities
├── segmentation/
│   ├── train.py            # SegFormer-B0 training
│   ├── train_deeplab.py    # DeepLabV3+ alternative
│   ├── check_data.py       # Dataset validation
│   └── visualize.py        # Prediction visualization
└── CLAUDE.md               # Development reference
```

## Quick Start

### Detection
```bash
python detection/parse_jaad.py      # Preprocess JAAD → YOLO format
python detection/train_yolo.py      # Train YOLOv8-Nano
python detection/main.py            # Run inference pipeline
```

### Intention
```bash
pip install -r intention/requirements.txt
python intention/dataset.py --data_path=. --compute_kps --regen
python intention/train.py --config intention/configs/JAAD_train.yaml -f 100
```

### Segmentation
```bash
python segmentation/train.py        # Train SegFormer-B0
python segmentation/train_deeplab.py # Train DeepLabV3+ alternative
```

See each module's README for detailed usage.

## Hardware

All modules target **Apple Silicon (MPS)** with automatic CPU fallback. Training and inference run on Metal Performance Shaders when available.

## Key Dependencies

- PyTorch (MPS-accelerated)
- Ultralytics YOLOv8
- Hugging Face Transformers (SegFormer)
- OpenPifPaf 0.10.1 (2D pose estimation)
- OpenCV, scikit-learn, TensorBoardX

## Dataset

This project uses the [JAAD (Joint Attention in Autonomous Driving)](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/) dataset for detection and intention modules, and the FPVCrosswalk2025 dataset for crosswalk segmentation.
