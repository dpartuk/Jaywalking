import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
# Update this to exact location. Use absolute path if unsure.
DATASET_ROOT = "./FPVCrosswalk2025" 

def check_pairs():
    # 1. Quick File Scan
    print(f"Scanning {DATASET_ROOT}...")
    image_paths = []
    mask_paths = []
    
    for root, dirs, files in os.walk(DATASET_ROOT):
        if 'images' in os.path.basename(root):
            mask_root = root.replace('/images', '/masks')
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(root, file)
                    # Handle naming convention
                    name_stem = Path(file).stem
                    suffix = Path(file).suffix
                    mask_name = f"{name_stem}_mask{suffix}"
                    mask_path = os.path.join(mask_root, mask_name)
                    
                    if os.path.exists(mask_path):
                        image_paths.append(img_path)
                        mask_paths.append(mask_path)

    print(f"Found {len(image_paths)} valid pairs.")
    if len(image_paths) == 0:
        print("ERROR: No data found. Check DATASET_ROOT path.")
        return

    # 2. visual Check of Mask 0
    img = Image.open(image_paths[0])
    mask = Image.open(mask_paths[0]).convert("L")
    mask_np = np.array(mask)
    
    print(f"Mask Shape: {mask_np.shape}")
    print(f"Unique values in raw mask (check for JPG noise): {np.unique(mask_np)}")
    
    # Simulate thresholding
    binary_mask = (mask_np > 128).astype(np.uint8)
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img)
    ax[0].set_title("Image")
    ax[1].imshow(mask_np, cmap='gray')
    ax[1].set_title("Raw Mask (Noisy?)")
    ax[2].imshow(binary_mask, cmap='gray')
    ax[2].set_title("Processed Binary Mask")
    plt.show()

if __name__ == "__main__":
    check_pairs()