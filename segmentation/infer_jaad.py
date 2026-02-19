"""Run trained SegFormer crosswalk segmentation on all JAAD frames.

Walks ~/local/JAAD_DS/images/<video_id>/<frame>.png and saves binary
crosswalk masks (0/255 uint8 PNG) to segmentation/crosswalk/<video_id>/<frame>.png.
"""

import os
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm.auto import tqdm


# ==========================================
# CONFIGURATION
# ==========================================
JAAD_IMAGES = os.path.expanduser("~/local/JAAD_DS/images")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "crosswalk")
WEIGHTS_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "best_segformer_crosswalk.pt")
MODEL_CHECKPOINT = "nvidia/mit-b0"

# Setup Device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    NUM_WORKERS = 4
    print(f"Using CUDA ({torch.cuda.get_device_name(0)}).")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    NUM_WORKERS = 0  # macOS fork issues with MPS
    print("Using Apple MPS (Metal Performance Shaders) acceleration.")
else:
    DEVICE = torch.device("cpu")
    NUM_WORKERS = 0
    print("MPS/CUDA not available. Using CPU (will be slow).")


# ==========================================
# DATASET
# ==========================================
class JAADFrameDataset(Dataset):
    """Loads JAAD frames and returns preprocessed tensors + metadata for saving."""

    def __init__(self, image_root, processor):
        self.processor = processor
        self.samples = []  # list of (abs_path, relative_path)

        for video_id in sorted(os.listdir(image_root)):
            video_dir = os.path.join(image_root, video_id)
            if not os.path.isdir(video_dir):
                continue
            for frame in sorted(os.listdir(video_dir)):
                if frame.lower().endswith((".png", ".jpg", ".jpeg")):
                    abs_path = os.path.join(video_dir, frame)
                    rel_path = os.path.join(video_id, frame)
                    self.samples.append((abs_path, rel_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        abs_path, rel_path = self.samples[idx]
        image = Image.open(abs_path).convert("RGB")
        orig_size = image.size  # (W, H)

        encoded = self.processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)

        return pixel_values, rel_path, orig_size[0], orig_size[1]


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="SegFormer inference on JAAD frames")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--weights", type=str, default=WEIGHTS_PATH, help="Path to model weights")
    parser.add_argument("--input_dir", type=str, default=JAAD_IMAGES, help="JAAD images root")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output masks root")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--sample", type=int, default=0,
                        help="Randomly select N frames total from random videos (for testing)")
    parser.add_argument("--video", type=str, default=None,
                        help="Process only this video ID (e.g. video_0001)")
    parser.add_argument("--frame", type=str, default=None,
                        help="Process only this frame within --video (e.g. 00150.png)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-frame softmax stats to diagnose model confidence")
    args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"JAAD images not found at {args.input_dir}")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Model weights not found at {args.weights}")

    # Load model
    print("Loading SegFormer model...")
    id2label = {0: "background", 1: "crosswalk"}
    label2id = {"background": 0, "crosswalk": 1}

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    processor = SegformerImageProcessor.from_pretrained(MODEL_CHECKPOINT)

    # Build dataset
    print(f"Scanning frames in {args.input_dir} ...")
    dataset = JAADFrameDataset(args.input_dir, processor)
    print(f"Found {len(dataset)} frames.")

    if len(dataset) == 0:
        print("No frames found. Exiting.")
        return

    # Filter to specific video/frame for testing
    if args.video or args.frame:
        if args.frame and not args.video:
            parser.error("--frame requires --video")
        filtered = []
        for i, (abs_path, rel_path) in enumerate(dataset.samples):
            vid = rel_path.split(os.sep)[0]
            fname = os.path.basename(rel_path)
            if vid != args.video:
                continue
            if args.frame and fname != args.frame:
                continue
            filtered.append(i)
        if not filtered:
            print(f"No frames matched --video={args.video}" +
                  (f" --frame={args.frame}" if args.frame else ""))
            return
        dataset = Subset(dataset, filtered)
        print(f"Filtered to {len(dataset)} frame(s) from {args.video}.")

    # Random sampling for testing
    if args.sample > 0:
        n = min(args.sample, len(dataset))
        indices = random.sample(range(len(dataset)), n)
        dataset = Subset(dataset, indices)
        print(f"Randomly sampled {n} frames for testing.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Inference
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving masks to {args.output_dir}")

    with torch.no_grad():
        for pixel_values, rel_paths, widths, heights in tqdm(loader, desc="Inference"):
            pixel_values = pixel_values.to(DEVICE)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits  # [B, 2, H/4, W/4]

            # Process each image in the batch
            for i in range(logits.shape[0]):
                w, h = int(widths[i]), int(heights[i])

                # Upsample logits to original resolution
                upsampled = torch.nn.functional.interpolate(
                    logits[i].unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )

                if args.debug:
                    probs = torch.softmax(upsampled, dim=1)
                    cw_prob = probs[0, 1]  # crosswalk channel
                    pct_over_50 = (cw_prob > 0.5).float().mean().item() * 100
                    print(f"  {rel_paths[i]}: crosswalk_prob "
                          f"max={cw_prob.max():.4f} mean={cw_prob.mean():.4f} "
                          f"pixels>0.5={pct_over_50:.2f}%")

                pred = upsampled.argmax(dim=1).squeeze(0)  # [H, W]

                # Convert to 0/255 uint8 mask
                mask = (pred.cpu().numpy() * 255).astype(np.uint8)

                # Save
                out_path = os.path.join(args.output_dir, rel_paths[i])
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                # Force .png extension regardless of input format
                out_path = str(Path(out_path).with_suffix(".png"))
                Image.fromarray(mask, mode="L").save(out_path)

    print(f"Done. Masks saved to {args.output_dir}")


if __name__ == "__main__":
    main()
