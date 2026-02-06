import os
import glob
import numpy as np
import torch
import evaluate
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision import transforms
from tqdm.auto import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_ROOT = "./FPVCrosswalk2025" 
BATCH_SIZE = 8
LR = 1e-4 # DeepLab often likes slightly higher LR than Transformers
EPOCHS = 10
NUM_WORKERS = 0

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using Apple MPS (Metal Performance Shaders) acceleration.")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ MPS not available. Using CPU.")

# ==========================================
# 2. DATA GATHERING
# ==========================================
def get_image_mask_pairs(root_dir):
    image_paths = []
    mask_paths = []
    
    print("Scanning dataset files...")
    for root, dirs, files in os.walk(root_dir):
        if 'images' in os.path.basename(root):
            mask_root = root.replace('/images', '/masks')
            if not os.path.exists(mask_root): continue
                
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    name_stem = Path(file).stem
                    suffix = Path(file).suffix
                    candidates = [f"{name_stem}_mask{suffix}", f"{name_stem}{suffix}"]
                    
                    for mask_name in candidates:
                        mask_path = os.path.join(mask_root, mask_name)
                        if os.path.exists(mask_path):
                            image_paths.append(img_path)
                            mask_paths.append(mask_path)
                            break
    return image_paths, mask_paths

images, masks = get_image_mask_pairs(DATASET_ROOT)
print(f"✅ Found {len(images)} pairs.")
train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(images, masks, test_size=0.2, random_state=42)
val_imgs, test_imgs, val_masks, test_masks = train_test_split(temp_imgs, temp_masks, test_size=0.5, random_state=42)

# ==========================================
# 3. DATASET CLASS (TorchVision Style)
# ==========================================
class CrosswalkDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        # Resize to fixed size for DeepLab (e.g., 512x512)
        target_size = (512, 512)
        image = image.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)

        # Convert mask to numpy for thresholding
        mask_np = np.array(mask)
        mask_np = (mask_np > 128).astype(np.longlong) # Long required for CrossEntropy

        # Transforms
        if self.transform:
            image = self.transform(image)
            
        return image, torch.from_numpy(mask_np)

# Standard ImageNet normalization for DeepLab
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CrosswalkDataset(train_imgs, train_masks, transform=img_transform)
val_dataset = CrosswalkDataset(val_imgs, val_masks, transform=img_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# ==========================================
# 4. MODEL SETUP (DeepLabV3)
# ==========================================
print("Loading DeepLabV3-ResNet50...")
# Load pretrained model
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

# Modify the Classifier Head for 2 classes (Background, Crosswalk)
# DeepLab's classifier is a DeepLabHead -> last layer is a Conv2d
model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

# DeepLab also has an "auxiliary" classifier for training stability
model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
metric = evaluate.load("mean_iou")

# ==========================================
# 5. TRAINING LOOP
# ==========================================
print("\nStarting Training...")
best_iou = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    model.train()
    train_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for images, masks in progress_bar:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        
        # DeepLab outputs an OrderedDict: {'out': tensor, 'aux': tensor}
        outputs = model(images)
        logits = outputs['out']
        aux_logits = outputs['aux']
        
        # Calculate loss (Main + 0.4 * Aux)
        loss_main = torch.nn.functional.cross_entropy(logits, masks)
        loss_aux = torch.nn.functional.cross_entropy(aux_logits, masks)
        loss = loss_main + 0.4 * loss_aux
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    print(f"Average Train Loss: {train_loss / len(train_loader):.4f}")

    # --- VALIDATION ---
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)['out']
            predictions = outputs.argmax(dim=1)
            
            metric.add_batch(
                predictions=predictions.detach().cpu().numpy(), 
                references=masks.detach().cpu().numpy()
            )
    
    metrics = metric.compute(num_labels=2, ignore_index=255)
    mean_iou = metrics['mean_iou']
    print(f"Mean IoU: {mean_iou:.4f}")
    
    if mean_iou > best_iou:
        print(f"🚀 Saved Best Model (IoU: {mean_iou:.4f})")
        best_iou = mean_iou
        torch.save(model.state_dict(), "best_deeplab_crosswalk.pt")

print("Training Complete.")