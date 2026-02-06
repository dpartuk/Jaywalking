import torch
import math
import time

def check_gpu():
    # 1. Check if MPS is built into PyTorch
    if not torch.backends.mps.is_built():
        print("❌ MPS not built into this PyTorch version!")
        return

    # 2. Check if MPS is available essentially
    if not torch.backends.mps.is_available():
        print("❌ MPS not available (Are you on MacOS 12.3+?)")
        return
    
    # 3. Actual Compute Test
    device = torch.device("mps")
    print(f"✅ Success! Running on: {device}")
    
    try:
        # Create a random tensor on GPU
        x = torch.randn(5000, 5000, device=device)
        
        # Perform a matrix multiplication (Heavy GPU task)
        start = time.time()
        y = torch.matmul(x, x)
        end = time.time()
        
        print(f"🚀 M4 Speed Test: 5000x5000 Matrix Multiplication took {end - start:.4f} seconds.")
        print("Environment is ready for Step 2.")
        
    except Exception as e:
        print(f"❌ Error running on MPS: {e}")

if __name__ == "__main__":
    check_gpu()