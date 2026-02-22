# Jaywalking Detection Pipeline

A three-module deep learning pipeline for pedestrian detection, crossing intention prediction, and crosswalk surface segmentation — built for the EPFL [CIVIL-459: Deep Learning for Autonomous Vehicles](https://edu.epfl.ch/coursebook/en/deep-learning-for-autonomous-vehicles-CIVIL-459) course.

## Pipeline Overview

The pipeline detects jaywalking by combining two independent models and merging their outputs:

```
                      JAAD Video (.mp4)
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
     Step 1: Intention           Step 2: Segmentation
     (DSTformer)                 (SegFormer-B3)
                │                       │
   Per-frame pedestrian        Per-frame crosswalk
   crossing predictions        probability metrics
   + bounding boxes            (max_prob, mean_prob)
                │                       │
                └───────────┬───────────┘
                            ▼
                  Step 3: Jaywalking Classification
                  (merge_predictions.py)
                            │
                            ▼
            Per-frame + aggregated jaywalking flags
                      + risk scores
```

Steps 1 and 2 are independent and can run in parallel on the same video. Step 3 merges the outputs without requiring a model.

### Step 1: Intention Prediction

- **Input**: Raw JAAD video frames
- **Process**: OpenPifPaf extracts 2D pose keypoints (17 body joints) → DSTformer transformer predicts crossing intent from 30-frame temporal clips
- **Output**: `datagen/infer_DB/infer_pred/<video_id>.json` — per-frame list of pedestrians with `pred` (0=not crossing, 1=crossing), `confidence`, and `bbox`
- **Note**: First 30 frames have no predictions (model requires full temporal context)

### Step 2: Crosswalk Segmentation

- **Input**: Video frames extracted from the same JAAD videos
- **Process**: SegFormer-B3 produces pixel-level crosswalk probability maps
- **Output**:
  - Binary masks: `segmentation/crosswalk/<video_id>/<frame>.png`
  - Summary metrics: `segmentation/crosswalk/crosswalk_metrics.csv` — per-frame `max_prob`, `mean_prob`, `pct_pixels_over_50`

### Step 3: Jaywalking Classification

- **Input**: Step 1 JSONs + Step 2 metrics CSV
- **Process**: For each frame, a pedestrian is flagged as jaywalking when `pred == 1` (crossing) AND `max_prob < threshold` (no crosswalk detected in the frame). This is a frame-level crosswalk probability check, not a spatial overlap between bounding box and mask.
- **Output**:
  - `jaywalking_results.csv` — per-frame, per-pedestrian jaywalking flags
  - `jaywalking_results_aggregated.csv` — sliding window aggregation (50-frame windows, step 10, min 25 jaywalking frames) with risk scores enriched with JAAD scene attributes (location, weather, time of day, vehicle action, pedestrian age, lane count)

### Risk Score

The aggregated output includes a composite risk score per window:

```
risk = P(crossing) × (1 - P(crosswalk)) × V_risk × E_risk × C_risk × A_risk
```

Where `V_risk` = vehicle action, `E_risk` = environment (time, weather, lanes), `C_risk` = crowd factor, `A_risk` = age vulnerability. Returns 0 for plaza/indoor locations.

## Modules

| Module | Model | Task |
|--------|-------|------|
| [**detection/**](detection/README.md) | YOLOv8-Nano + SegFormer surface classifier | Pedestrian detection and 4-state classification (STATIC, SAFE, CROSSWALK, JAYWALKING) |
| [**intention/**](intention/README.md) | MotionBERT (DSTformer) | Binary crossing intention prediction from 2D pose keypoints |
| [**segmentation/**](segmentation/README.md) | SegFormer-B3 / DeepLabV3+ | Pixel-level crosswalk segmentation on first-person-view images |

## Repository Structure

```
├── detection/
│   ├── main.py              # Full inference pipeline
│   ├── parse_jaad.py        # JAAD XML → YOLO format conversion
│   ├── train_yolo.py        # YOLOv8-Nano training
│   ├── jaad.yaml            # YOLO dataset config
│   └── jaad_mini.yaml       # Mini dataset config
├── intention/
│   ├── train.py             # Training / evaluation entry point
│   ├── inference.py         # Video inference with OpenPifPaf
│   ├── dataset.py           # JAAD dataset generation
│   ├── configs/             # YAML training & inference configs
│   └── lib/                 # Model (DSTformer), data, and utilities
├── segmentation/
│   ├── train.py             # SegFormer-B3 training
│   ├── train_deeplab.py     # DeepLabV3+ alternative
│   ├── infer_jaad.py        # Run segmentation on JAAD video frames
│   ├── check_data.py        # Dataset validation
│   └── visualize.py         # Prediction visualization
├── merge_predictions.py     # Step 3: merge intention + segmentation → jaywalking
├── run_inference_all.sh     # Batch intention inference on all 346 JAAD videos
└── CLAUDE.md                # Development reference
```

## Quick Start

### Step 1: Intention Inference
```bash
pip install -r intention/requirements.txt

# Single video
python intention/inference.py --config intention/configs/inference.yaml \
  --data_path datagen/infer_DB/infer_clips/ --filename <video>

# All 346 JAAD videos
bash run_inference_all.sh
```

### Step 2: Segmentation Inference
```bash
python segmentation/infer_jaad.py --jaad_path <JAAD_clips_dir>
```

### Step 3: Merge & Classify
```bash
python merge_predictions.py \
  --intention_dir datagen/infer_DB/infer_pred/ \
  --segmentation_csv segmentation/crosswalk/crosswalk_metrics.csv \
  --crosswalk_threshold 0.5
```

### Training (optional)
```bash
# Detection
python detection/parse_jaad.py      # Preprocess JAAD → YOLO format
python detection/train_yolo.py      # Train YOLOv8-Nano

# Intention
python intention/dataset.py --data_path=. --compute_kps --regen
python intention/train.py --config intention/configs/JAAD_train.yaml -f 100

# Segmentation
python segmentation/train.py        # Train SegFormer-B3
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
