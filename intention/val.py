import torch
import platform

def verify_mps():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")

    is_built = torch.backends.mps.is_built()

    is_available = torch.backends.mps.is_available()

    print(f"MPS Built: {is_built}")
    print(f"MPS Available: {is_available}")

    if is_built and is_available:
        try:
            device = torch.device("mps")
            x = torch.ones(1, device=device)
            print(f"Success: Tensor created on {x.device}")
        except Exception as e:
            print(f"Error: MPS available but tensor creation failed. {e}")
    else:
        print("Warning: MPS not active. System will default to CPU.")

if __name__ == "__main__":
    verify_mps()