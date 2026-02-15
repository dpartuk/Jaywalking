"""
Create a mini version of jaad_dataset.pkl for quick training/testing.
Selects a small subset of videos with balanced crossing/not-crossing labels.

Usage:
    python intention/make_mini_dataset.py --input datagen/data/jaad_dataset.pkl --output datagen/data/jaad_mini.pkl --num_videos 10
"""

import pickle
import argparse
import numpy as np


def count_labels(annotations, vid):
    """Count crossing and not-crossing sequences in a video."""
    c, nc = 0, 0
    for ped in annotations[vid]['ped_annotations'].values():
        for sample in ped:
            if sample['cross'] == 1:
                c += 1
            else:
                nc += 1
    return c, nc


def main():
    parser = argparse.ArgumentParser(description="Create a mini JAAD dataset from the full pkl")
    parser.add_argument('--input', type=str, default='datagen/data/jaad_dataset.pkl',
                        help='Path to the full dataset pkl')
    parser.add_argument('--output', type=str, default='datagen/data/jaad_mini.pkl',
                        help='Path to write the mini dataset pkl')
    parser.add_argument('--num_videos', type=int, default=10,
                        help='Number of videos to include')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        dataset = pickle.load(f)

    annotations = dataset['annotations']
    all_vids = sorted(annotations.keys())

    # Score each video: prefer videos that have both C and NC labels
    vid_scores = []
    for vid in all_vids:
        c, nc = count_labels(annotations, vid)
        total = c + nc
        balance = min(c, nc) / max(total, 1)  # 0 = all one class, 0.5 = perfect balance
        vid_scores.append((vid, c, nc, total, balance))

    # Sort by balance (most balanced first), break ties by total sequences (more is better)
    vid_scores.sort(key=lambda x: (x[4], x[3]), reverse=True)

    selected = [v[0] for v in vid_scores[:args.num_videos]]
    selected.sort()  # keep video order

    # Build mini dataset
    mini_annotations = {vid: annotations[vid] for vid in selected}

    # Split: first 20% of selected videos for test, rest for train
    seq_counts = []
    for vid in selected:
        c, nc = count_labels(annotations, vid)
        seq_counts.append(c + nc)

    cumsum = np.cumsum(seq_counts) / sum(seq_counts)
    split_idx = next(i for i, val in enumerate(cumsum) if val > 0.2)

    test_ids = selected[:split_idx]
    train_ids = selected[split_idx:]

    mini_dataset = {
        'ckpt': selected[-1],
        'seq_per_vid': seq_counts,
        'split': {
            'train_ID': train_ids,
            'test_ID': test_ids,
        },
        'annotations': mini_annotations,
    }

    # Print summary
    total_c, total_nc = 0, 0
    for vid in selected:
        c, nc = count_labels(annotations, vid)
        total_c += c
        total_nc += nc
        split = "test" if vid in test_ids else "train"
        print(f"  {vid}  C={c:3d}  NC={nc:3d}  total={c+nc:3d}  [{split}]")

    print(f"\nSelected {len(selected)} videos")
    print(f"Train: {len(train_ids)} videos, Test: {len(test_ids)} videos")
    print(f"Total sequences: {total_c + total_nc}  (C={total_c}, NC={total_nc})")
    print(f"C ratio: {total_c/(total_c+total_nc):.1%}")

    with open(args.output, 'wb') as f:
        pickle.dump(mini_dataset, f, protocol=4)

    print(f"\nMini dataset written to {args.output}")


if __name__ == '__main__':
    main()
