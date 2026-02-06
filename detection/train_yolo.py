from ultralytics import YOLO
import torch

def train_model():
    # 1. Load the model
    # 'yolov8n.pt' is the Nano version (fastest)
    # If you want more accuracy at the cost of speed, use 'yolov8s.pt'
    print("Loading YOLOv8-Nano...")
    model = YOLO('yolov8n.pt') 

    # 2. Train on M4 (MPS)
    print("Starting training on Apple Silicon (MPS)...")
    results = model.train(
        data='jaad_mini.yaml',
        epochs=2,             # 30 epochs is usually enough for this dataset
        imgsz=640,             # Standard resolution
        device='mps',          # Force Metal Performance Shaders
        batch=16,              # Adjust if you run out of memory (try 8 or 32)
        workers=4,             # Data loading workers
        name='jaad_yolo_run',  # Name of the output folder
        exist_ok=True,         # Overwrite existing run folder
        patience=5             # Stop early if no improvement for 5 epochs
    )
    
    print("✅ Training Complete!")
    print("Best model saved to: runs/detect/jaad_yolo_run/weights/best.pt")

if __name__ == '__main__':
    # Double check MPS again just to be safe
    if not torch.backends.mps.is_available():
        print("⚠️ Warning: MPS not detected. Training might be slow on CPU.")
    
    train_model()