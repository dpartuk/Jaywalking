# Segmentation Module

Pixel-level crosswalk segmentation using SegFormer-B0 (primary) or DeepLabV3+ (alternative) on first-person-view images.

## Overview

Binary semantic segmentation (background vs. crosswalk) trained on the FPVCrosswalk2025 dataset. Two model options are provided:

| Model | Backbone | Script | Default LR | Default Epochs |
|-------|----------|--------|-----------|----------------|
| **SegFormer-B0** | `nvidia/mit-b0` | `train.py` | 6e-5 | 2 |
| **DeepLabV3+** | ResNet-50 | `train_deeplab.py` | 1e-4 | 10 |

Both models output 2-class predictions and are evaluated with Mean IoU and per-category Crosswalk IoU.

## Dataset

The FPVCrosswalk2025 dataset contains first-person-view images with corresponding binary crosswalk masks.

### Expected Structure

```
FPVCrosswalk2025/
├── <split>/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── ...
│   └── masks/
│       ├── img1_mask.jpg
│       └── ...
└── ...
```

- Masks use the `_mask` suffix naming convention (e.g., `img1.jpg` pairs with `img1_mask.jpg`)
- JPG mask artifacts are handled automatically — pixel values >128 are mapped to crosswalk (class 1), the rest to background (class 0)
- Data is auto-split 80/10/10 into train/val/test

### Validate Dataset

```bash
python check_data.py
```

Scans for valid image-mask pairs and displays a sample with raw and thresholded masks side by side. Update `DATASET_ROOT` in the script if your data is elsewhere.

## Training

### SegFormer-B0 (recommended)

```bash
python train.py
```

Configuration (hardcoded at the top of `train.py`):
- `DATASET_ROOT`: `./FPVCrosswalk2025`
- `BATCH_SIZE`: 8
- `LR`: 6e-5
- `EPOCHS`: 2
- `NUM_WORKERS`: 0 (for macOS stability)

Best weights saved to `best_segformer_crosswalk.pt`.

### DeepLabV3+

```bash
python train_deeplab.py
```

Configuration:
- `BATCH_SIZE`: 8
- `LR`: 1e-4
- `EPOCHS`: 10
- Input resized to 512x512
- Uses ImageNet normalization
- Main loss + 0.4 * auxiliary loss

Best weights saved to `best_deeplab_crosswalk.pt`.

## Evaluation

Both training scripts run validation after each epoch and report:
- **Mean IoU** across both classes
- **Crosswalk IoU** (per-category, class 1)

The best model checkpoint is saved when Mean IoU improves.

## Visualization

```bash
python visualize.py
```

Displays a side-by-side comparison of input image, ground truth mask, and model prediction for a sample from the validation set. Requires a trained SegFormer checkpoint (`best_segformer_crosswalk.pt`).

## Dependencies

- PyTorch (MPS)
- Hugging Face Transformers (SegFormer)
- torchvision (DeepLabV3+)
- scikit-learn (train/val/test splitting)
- evaluate (Mean IoU metric)
- Pillow, matplotlib
