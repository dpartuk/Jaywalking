"""Fine-tune SegFormer crosswalk segmentation on JAAD annotations.

Loads pretrained weights from best_segformer_crosswalk.pt (trained on
FPVCrosswalk2025) and fine-tunes on manually annotated JAAD frames in
segmentation/jaad_finetune/{images,masks}/.

Saves improved weights to best_segformer_crosswalk_jaad.pt.
"""

import os
import random
import numpy as np
import torch
torch.backends.cudnn.enabled = False  # Workaround for ThunderCompute prototyping mode
import evaluate
from pathlib import Path
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm.auto import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JAAD_FINETUNE_DIR = os.path.join(SCRIPT_DIR, "jaad_finetune")
IMAGES_DIR = os.path.join(JAAD_FINETUNE_DIR, "images")
MASKS_DIR = os.path.join(JAAD_FINETUNE_DIR, "masks")

OUTPUT_WEIGHTS = os.path.join(os.path.dirname(SCRIPT_DIR), "best_segformer_b3_crosswalk_jaad.pt")
# Resume from JAAD fine-tuned weights if available, otherwise start from pretrained ImageNet backbone
PRETRAINED_WEIGHTS = OUTPUT_WEIGHTS if os.path.isfile(OUTPUT_WEIGHTS) else None

MODEL_CHECKPOINT = "nvidia/mit-b3"

BATCH_SIZE = 4  # Reduced for larger model
LR = 1e-5             # Low LR for fine-tuning
EPOCHS = 100
VAL_SPLIT = 0.2

# Setup Device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    NUM_WORKERS = 4
    print(f"Using CUDA ({torch.cuda.get_device_name(0)}).")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    NUM_WORKERS = 0  # macOS fork issues with MPS
    print("Using Apple MPS (Metal Performance Shaders) acceleration.")
else:
    DEVICE = torch.device("cpu")
    NUM_WORKERS = 0
    print("MPS/CUDA not available. Using CPU (will be slow).")

# ==========================================
# 2. DATA GATHERING
# ==========================================
def get_image_mask_pairs(images_dir, masks_dir):
    image_paths = []
    mask_paths = []

    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(images_dir, fname)
        stem = Path(fname).stem

        # Match mask by stem (masks are always .png from labelme_to_masks.py)
        mask_path = os.path.join(masks_dir, f"{stem}.png")
        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
        else:
            print(f"  Warning: no mask for {fname}, skipping")

    return image_paths, mask_paths


# ==========================================
# 3. DATASET CLASS
# ==========================================
class CrosswalkDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def _apply_augmentations(self, image, mask):
        # Horizontal flip (50%)
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random resized crop (50%)
        if random.random() > 0.5:
            w, h = image.size
            scale = random.uniform(0.7, 1.0)
            new_h, new_w = int(h * scale), int(w * scale)
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            image = TF.resized_crop(image, top, left, new_h, new_w, (h, w))
            mask = TF.resized_crop(mask, top, left, new_h, new_w, (h, w),
                                   interpolation=T.InterpolationMode.NEAREST)

        # Color jitter (image only, 50%)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

        # Gaussian blur (image only, 20%)
        if random.random() > 0.8:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        return image, mask

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.augment:
            image, mask = self._apply_augmentations(image, mask)

        mask_np = np.array(mask)
        # Threshold: >128 = crosswalk (handles any compression artifacts)
        mask_np = (mask_np > 128).astype(np.uint8)

        encoded_inputs = self.processor(
            images=image,
            segmentation_maps=mask_np,
            return_tensors="pt"
        )

        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.squeeze(0)

        return encoded_inputs


# ==========================================
# 4. MAIN
# ==========================================
def main():
    # Validate paths
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")
    if not os.path.isdir(MASKS_DIR):
        raise FileNotFoundError(f"Masks directory not found: {MASKS_DIR}")

    # Gather data
    print("Scanning fine-tuning dataset...")
    images, masks = get_image_mask_pairs(IMAGES_DIR, MASKS_DIR)

    if len(images) == 0:
        raise ValueError(f"No image/mask pairs found in {JAAD_FINETUNE_DIR}")

    print(f"Found {len(images)} image/mask pairs.")

    # Split
    if len(images) < 5:
        # Too few for a split — use all for training, skip validation
        train_imgs, train_masks = images, masks
        val_imgs, val_masks = [], []
        print("Too few images for val split — using all for training.")
    else:
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            images, masks, test_size=VAL_SPLIT, random_state=42
        )

    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    # Load model with pretrained FPVCrosswalk weights
    print("Loading SegFormer model...")
    id2label = {0: "background", 1: "crosswalk"}
    label2id = {"background": 0, "crosswalk": 1}

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if PRETRAINED_WEIGHTS and os.path.isfile(PRETRAINED_WEIGHTS):
        print(f"Loading pretrained weights from {PRETRAINED_WEIGHTS}")
        model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE, weights_only=True))
    else:
        print(f"Starting from pretrained ImageNet backbone ({MODEL_CHECKPOINT})")

    processor = SegformerImageProcessor.from_pretrained(MODEL_CHECKPOINT)

    # Datasets and loaders
    train_dataset = CrosswalkDataset(train_imgs, train_masks, processor, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_loader = None
    if val_imgs:
        val_dataset = CrosswalkDataset(val_imgs, val_masks, processor)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    model.to(DEVICE)

    # Metrics
    metric = evaluate.load("mean_iou")

    # ==========================================
    # 5. TRAINING LOOP
    # ==========================================
    print("\nStarting Fine-tuning with augmentation...")

    best_cw_iou = 0.0

    # Establish baseline by running validation before training
    if val_loader is not None:
        model.eval()
        print("Running baseline validation...")
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                outputs = model(pixel_values=pixel_values)
                upsampled_logits = torch.nn.functional.interpolate(
                    outputs.logits, size=labels.shape[-2:],
                    mode="bilinear", align_corners=False
                )
                predictions = upsampled_logits.argmax(dim=1)
                metric.add_batch(
                    predictions=predictions.detach().cpu().numpy(),
                    references=labels.detach().cpu().numpy()
                )
        metrics = metric.compute(num_labels=2, ignore_index=255)
        best_cw_iou = metrics["per_category_iou"][1]
        print(f"Baseline Crosswalk IoU: {best_cw_iou:.4f}")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # --- TRAIN ---
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(DEVICE).contiguous()
            labels = batch["labels"].to(DEVICE).contiguous()

            optimizer.zero_grad()

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            # Downsample labels to match logits (avoids MPS interpolate backward crash)
            h, w = logits.shape[-2:]
            labels_small = torch.nn.functional.interpolate(
                labels.unsqueeze(1).float(),
                size=(h, w),
                mode="nearest"
            ).squeeze(1).long()

            loss = torch.nn.functional.cross_entropy(
                logits, labels_small,
                weight=torch.tensor([1.0, 5.0], device=DEVICE)
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = train_loss / len(train_loader)
        scheduler.step()
        print(f"Avg Train Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # --- VALIDATION ---
        if val_loader is None:
            # No validation set — save every epoch
            torch.save(model.state_dict(), OUTPUT_WEIGHTS)
            print(f"Saved weights to {OUTPUT_WEIGHTS}")
            continue

        model.eval()
        print("Running Validation...")
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(pixel_values=pixel_values)

                upsampled_logits = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

                predictions = upsampled_logits.argmax(dim=1)

                metric.add_batch(
                    predictions=predictions.detach().cpu().numpy(),
                    references=labels.detach().cpu().numpy()
                )

        metrics = metric.compute(num_labels=2, ignore_index=255)
        mean_iou = metrics["mean_iou"]
        iou_crosswalk = metrics["per_category_iou"][1]

        print(f"Mean IoU: {mean_iou:.4f} | Crosswalk IoU: {iou_crosswalk:.4f}")

        if iou_crosswalk > best_cw_iou:
            print(f"Crosswalk IoU improved ({best_cw_iou:.4f} -> {iou_crosswalk:.4f}). Saving model...")
            best_cw_iou = iou_crosswalk
            torch.save(model.state_dict(), OUTPUT_WEIGHTS)

    print(f"\nFine-tuning Complete. Best Crosswalk IoU: {best_cw_iou:.4f}. Weights saved to {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()
