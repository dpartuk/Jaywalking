# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jaywalking is a monorepo for a three-module deep learning pipeline for pedestrian detection, crossing intention prediction, and crosswalk surface segmentation — built for the EPFL CIVIL-459 Deep Learning for Autonomous Vehicles course. All modules target **Apple Silicon (MPS)** with CPU fallback.

## Architecture

The three modules form a complementary pedestrian safety system operating on JAAD dataset videos:

- **`detection/`** — YOLOv8 pedestrian detection + hybrid texture/edge crosswalk identification. Classifies pedestrians into states: STATIC, SAFE, CROSSWALK, JAYWALKING. Entry point: `main.py`.
- **`intention/`** — MotionBERT (DSTformer) transformer for binary crossing intention prediction from 2D pose keypoints (extracted via OpenPifPaf). Model architecture in `lib/model/` (ActionNet = DSTformer backbone + classification head). Datasets in `lib/data/`, utilities in `lib/utils/`.
- **`segmentation/`** — SegFormer-B0 (or DeepLabV3+ alternative) for pixel-level crosswalk segmentation on FPV images. Dataset: FPVCrosswalk2025 with synthetic and real-world splits.

## Commands

### Detection
```bash
python detection/parse_jaad.py          # Preprocess JAAD → YOLO format (80/20 split)
python detection/train_yolo.py          # Train YOLOv8-nano on JAAD
python detection/main.py                # Run full inference pipeline
```

### Intention
```bash
# Install dependencies
pip install -r intention/requirements.txt

# Generate dataset from JAAD videos
python intention/dataset.py --data_path=. --compute_kps --regen

# Train (30-frame clips, 5-block DSTformer, config in configs/JAAD_train.yaml)
python intention/train.py --config intention/configs/JAAD_train.yaml -f 100

# Resume from checkpoint
python intention/train.py --config intention/configs/JAAD_train.yaml -f 100 -c

# Evaluate
python intention/train.py --config intention/configs/JAAD_train.yaml -f 100 -e

# Inference on video
python intention/inference.py --config intention/configs/inference.yaml --data_path datagen/infer_DB/infer_clips/ --filename <video>

# TensorBoard
tensorboard --logdir=intention/logs/
```

### Segmentation
```bash
python segmentation/train.py            # Train SegFormer-B0
python segmentation/train_deeplab.py    # Train DeepLabV3+ alternative
python segmentation/check_data.py       # Validate dataset stats
python segmentation/visualize.py        # Visualize predictions
```

There is no test suite, linter configuration, or build system.

## Key Configuration

- **`intention/configs/JAAD_train.yaml`** — Training hyperparameters (epochs: 50, batch: 8, lr: 1e-4/1e-3, depth: 5, num_joints: 19, clip_len: 30, dim_feat: 512)
- **`detection/jaad.yaml`** / **`jaad_mini.yaml`** — YOLO dataset configs (classes: person_crossing, person_static)
- **`segmentation/train.py`** — Hardcoded config (batch: 8, lr: 6e-5, epochs: 2, NUM_WORKERS: 0 for macOS stability)

## Key Dependencies

- PyTorch (MPS-accelerated), Ultralytics YOLO, Hugging Face Transformers (SegFormer)
- OpenPifPaf 0.10.1 (2D pose estimation for intention module)
- OpenCV, TensorBoardX, scikit-learn

## Data Flow

JAAD videos → **Detection** (YOLOv8 bounding boxes + surface classification) → pedestrian state labels
JAAD videos → **Intention** (OpenPifPaf keypoints → DSTformer temporal transformer) → crossing probability
FPV images → **Segmentation** (SegFormer/DeepLab) → crosswalk pixel masks
