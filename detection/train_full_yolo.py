from ultralytics import YOLO
import torch

def train_full_scale():
    print("🚀 Starting Full-Scale Training on JAAD (100k+ images)...")
    print("⚠️ This will take several hours. Keep your Mac plugged in!")

    # Load the base model
    model = YOLO('yolov8n.pt') 

    # Train
    results = model.train(
        data='jaad.yaml',      # Pointing to the FULL dataset
        epochs=50,             # Deep training
        patience=10,           # Stop if no improvement for 10 epochs
        imgsz=640,
        device='mps',          # Apple Silicon Acceleration
        batch=16,              
        workers=4,
        name='jaad_yolo_full', # New folder for results
        exist_ok=True,
        save=True,             # Save checkpoints
        save_period=5          # Save every 5 epochs just in case
    )
    
    print("✅ Training Complete!")
    print("Best weights: runs/detect/jaad_yolo_full/weights/best.pt")

if __name__ == '__main__':
    train_full_scale()