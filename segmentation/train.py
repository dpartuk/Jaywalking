import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import glob
import numpy as np
import torch
import evaluate
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm.auto import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Update this path to where your dataset folder actually is
DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FPVCrosswalk2025")

BATCH_SIZE = 8          # Lower to 4 or 2 if you run out of memory
LR = 6e-5               # Learning Rate
EPOCHS = 2               # Number of training epochs
NUM_WORKERS = 0         # Set to 0 for maximum stability on macOS M-chips

# Setup Device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA ({torch.cuda.get_device_name(0)}).")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS (Metal Performance Shaders) acceleration.")
else:
    DEVICE = torch.device("cpu")
    print("MPS/CUDA not available. Using CPU (will be slow).")

# ==========================================
# 2. DATA GATHERING
# ==========================================
# Function to get all image-mask pairs
# Assumes images are in 'images' folder and masks in 'masks' folder with '_mask' suffix
# e.g., images/img1.jpg <-> masks/img1_mask.jpg
# Returns two lists: image paths and corresponding mask paths
#
def get_image_mask_pairs(root_dir):
    image_paths = []
    mask_paths = []
    
    print("Scanning dataset files...")
    for root, dirs, files in os.walk(root_dir):
        if 'images' in os.path.basename(root):
            mask_root = root.replace('/images', '/masks')
            
            if not os.path.exists(mask_root):
                continue
                
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    
                    # Construct mask filename: name.jpg -> name_mask.jpg
                    name_stem = Path(file).stem
                    suffix = Path(file).suffix
                    
                    # Check common naming conventions
                    candidates = [
                        f"{name_stem}_mask{suffix}",
                        f"{name_stem}{suffix}" 
                    ]
                    
                    for mask_name in candidates:
                        mask_path = os.path.join(mask_root, mask_name)
                        if os.path.exists(mask_path):
                            image_paths.append(img_path)
                            mask_paths.append(mask_path)
                            break
    
    return image_paths, mask_paths

# Gather data
images, masks = get_image_mask_pairs(DATASET_ROOT)

if len(images) == 0:
    raise ValueError(f"No images found in {DATASET_ROOT}. Check your path structure.")

print(f"✅ Found {len(images)} image/mask pairs.")

# Splits
train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(images, masks, test_size=0.2, random_state=42)
val_imgs, test_imgs, val_masks, test_masks = train_test_split(temp_imgs, temp_masks, test_size=0.5, random_state=42)

print(f"Data Split -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

# ==========================================
# 3. DATASET CLASS
# ==========================================
# We need to handle the JPG masks. JPG compression turns a perfect binary mask (0, 255)
# into noisy gray values (e.g., 5, 250). We must threshold them.
class CrosswalkDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Open Image and Mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L") 

        # Convert to numpy
        mask_np = np.array(mask)

        # Thresholding: Handle JPG artifacts in masks
        # Everything > 128 is Class 1 (Crosswalk), else 0 (Background)
        mask_np = (mask_np > 128).astype(np.uint8)

        # Preprocess using HuggingFace utility
        encoded_inputs = self.processor(
            images=image,
            segmentation_maps=mask_np,
            return_tensors="pt"
        )

        # Squeeze batch dimension
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.squeeze(0)

        return encoded_inputs

# ==========================================
# 4. MODEL SETUP
# ==========================================
# Load Model for Crosswalk Segmentation
# We have 2 classes: Background and Crosswalk
# We will use SegformerForSemanticSegmentation. 
# The standard checkpoint is usually nvidia/mit-b0 (fastest) or nvidia/segformer-b0-finetuned-ade-512-512.

print("Loading SegFormer model...")
id2label = {0: "background", 1: "crosswalk"}
label2id = {"background": 0, "crosswalk": 1}

# Using mit-b0 for speed.
model_checkpoint = "nvidia/mit-b0"

model = SegformerForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    num_labels=2, 
    id2label=id2label, 
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

processor = SegformerImageProcessor.from_pretrained(model_checkpoint)

# Create Datasets and Loaders
train_dataset = CrosswalkDataset(train_imgs, train_masks, processor)
val_dataset = CrosswalkDataset(val_imgs, val_masks, processor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LR)

# Move model to GPU (MPS)
model.to(DEVICE)

# Metrics
metric = evaluate.load("mean_iou")

# ==========================================
# 5. TRAINING LOOP
# ==========================================
print("\nStarting Training...")

best_iou = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    
    # --- TRAIN ---
    model.train()
    train_loss = 0.0
    
    # tqdm creates the progress bar
    progress_bar = tqdm(train_loader, desc="Training")
    
    # ... (inside the epoch loop) ...
    for batch in progress_bar:
        # 1. Force inputs to be contiguous
        pixel_values = batch["pixel_values"].to(DEVICE).contiguous()
        labels = batch["labels"].to(DEVICE).contiguous()

        optimizer.zero_grad()
        
        # 2. Forward pass (No labels passed to model)
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # Shape: [Batch, 2, 128, 128]
        
        # 3. FIX: Downsample Labels instead of Upsampling Logits
        # This avoids the 'interpolate.backward' crash on MPS.
        
        # Get the resolution of the model output
        h, w = logits.shape[-2:]
        
        # Resize labels to match model output (Nearest Neighbor preserves integer classes)
        # Labels: [B, H, W] -> [B, 1, H, W] for interpolate -> [B, H, W]
        labels_small = torch.nn.functional.interpolate(
            labels.unsqueeze(1).float(), 
            size=(h, w), 
            mode="nearest"
        ).squeeze(1).long()
        
        # 4. Calculate Loss at lower resolution
        loss = torch.nn.functional.cross_entropy(logits, labels_small)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update stats
        train_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # --- VALIDATION ---
    model.eval()
    print("Running Validation...")
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(pixel_values=pixel_values)
            
            # Resize logits to match label size for evaluation
            upsampled_logits = torch.nn.functional.interpolate(
                outputs.logits, 
                size=labels.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
            predictions = upsampled_logits.argmax(dim=1)
            
            # Add batch to metric
            metric.add_batch(
                predictions=predictions.detach().cpu().numpy(), 
                references=labels.detach().cpu().numpy()
            )
    
    # Compute Metrics
    metrics = metric.compute(num_labels=2, ignore_index=255)
    mean_iou = metrics['mean_iou']
    iou_crosswalk = metrics['per_category_iou'][1]
    
    print(f"Mean IoU: {mean_iou:.4f} | Crosswalk IoU: {iou_crosswalk:.4f}")
    
    # Save Best Model
    if mean_iou > best_iou:
        print(f"🚀 IoU improved ({best_iou:.4f} -> {mean_iou:.4f}). Saving model...")
        best_iou = mean_iou
        torch.save(model.state_dict(), "best_segformer_crosswalk.pt")

print("\nTraining Complete.")