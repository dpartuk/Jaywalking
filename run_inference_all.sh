#!/bin/bash
# Run intention inference on all JAAD video clips.
# Usage: bash run_inference_all.sh
# Run from the repository root on a Linux machine with GPU.

set -e

# Install dependencies
# pip install -r intention/requirements.txt

# Verify checkpoint exists
if [ ! -f checkpoints/best_epoch.bin ]; then
    echo "Error: checkpoints/best_epoch.bin not found"
    exit 1
fi

# Create output directory
mkdir -p datagen/infer_DB/infer_pred

total=$(ls datagen/infer_DB/infer_clips/video_*.mp4 | wc -l)
count=0

for vid in datagen/infer_DB/infer_clips/video_*.mp4; do
    name=$(basename "$vid" .mp4)
    count=$((count + 1))
    echo "[$count/$total] Processing $name"

    # Skip if already processed
    if [ -f "datagen/infer_DB/infer_pred/${name}.json" ]; then
        echo "  Skipping (already exists)"
        continue
    fi

    python intention/inference.py \
        --config intention/configs/inference.yaml \
        --data_path . \
        --filename "$name"
done

echo "Done. $count videos processed."
echo "Results in datagen/infer_DB/infer_pred/"
