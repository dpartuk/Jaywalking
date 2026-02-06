import os
import shutil
import random
from tqdm import tqdm

# Config
SOURCE_DIR = "./dataset_yolo"
TARGET_DIR = "./dataset_mini"
NUM_TRAIN = 2000  # We only need 2000 images to test
NUM_VAL = 500

def create_subset(split, limit):
    print(f"📦 Creating 'Mini' version of {split} set...")
    
    # Source paths
    img_src = os.path.join(SOURCE_DIR, "images", split)
    lbl_src = os.path.join(SOURCE_DIR, "labels", split)
    
    # Target paths
    img_dst = os.path.join(TARGET_DIR, "images", split)
    lbl_dst = os.path.join(TARGET_DIR, "labels", split)
    
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)
    
    # Get all images
    files = [f for f in os.listdir(img_src) if f.endswith(".jpg")]
    
    # Randomly select a small batch
    if len(files) > limit:
        files = random.sample(files, limit)
        
    for f in tqdm(files):
        # Copy Image
        shutil.copy(os.path.join(img_src, f), os.path.join(img_dst, f))
        
        # Copy Label (if it exists)
        label_name = f.replace(".jpg", ".txt")
        if os.path.exists(os.path.join(lbl_src, label_name)):
            shutil.copy(os.path.join(lbl_src, label_name), os.path.join(lbl_dst, label_name))

if __name__ == "__main__":
    create_subset("train", NUM_TRAIN)
    create_subset("val", NUM_VAL)
    print(f"\n✅ Mini Dataset ready at: {TARGET_DIR}")