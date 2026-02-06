from ultralytics import YOLO

def resume_training():
    print("🔄 Resuming training from last checkpoint...")
    
    # Load the specific "last state" weights
    # Make sure this path matches where your training was saving
    model = YOLO('runs/detect/jaad_yolo_full/weights/last.pt') 
    
    # Resume training
    # It will automatically find the dataset and hyperparameters used previously
    model.train(resume=True, batch=8)

if __name__ == '__main__':
    resume_training()