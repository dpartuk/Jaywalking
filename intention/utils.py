import torch

def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA (NVIDIA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")

    return device