"""
Merge intention prediction JSONs with segmentation crosswalk metrics
to produce a unified CSV flagging jaywalking events.

A pedestrian is flagged as jaywalking when:
  - The intention model predicts them as crossing (pred == 1)
  - The segmentation model finds no crosswalk nearby (max_prob < threshold)
"""

import argparse
import csv
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge intention predictions with segmentation crosswalk metrics"
    )
    parser.add_argument(
        "--intention_dir",
        type=str,
        default="datagen/infer_DB/infer_pred/",
        help="Directory containing intention prediction JSONs",
    )
    parser.add_argument(
        "--segmentation_csv",
        type=str,
        default="segmentation/crosswalk/crosswalk_metrics.csv",
        help="Path to crosswalk metrics CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="jaywalking_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--crosswalk_threshold",
        type=float,
        default=0.5,
        help="max_prob below this means no crosswalk detected (default: 0.5)",
    )
    return parser.parse_args()


def load_segmentation_data(csv_path):
    """Load segmentation CSV into a dict keyed by (video_id, frame_number_int)."""
    seg_data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            # Parse frame filename like "00123.png" -> 123
            frame_num = int(os.path.splitext(row["frame"])[0])
            seg_data[(video_id, frame_num)] = {
                "max_prob": float(row["max_prob"]),
                "mean_prob": float(row["mean_prob"]),
                "pct_pixels_over_50": float(row["pct_pixels_over_50"]),
            }
    return seg_data


def process_intention_json(json_path, video_id, seg_data, crosswalk_threshold):
    """Process a single intention prediction JSON and return result rows."""
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for frame_entry in data["output"]:
        intention_frame = frame_entry["frame"]  # 1-indexed
        seg_frame = intention_frame - 1  # 0-indexed

        seg_key = (video_id, seg_frame)
        seg_metrics = seg_data.get(seg_key)

        for ped_idx, ped in enumerate(frame_entry["predictions"]):
            pred = ped["pred"]
            confidence = ped["confidence"]
            bbox = ped.get("bbox")

            # Skip pedestrians with no prediction (None)
            if pred is None:
                continue

            crossing = int(pred) == 1

            if seg_metrics is not None:
                max_prob = seg_metrics["max_prob"]
                mean_prob = seg_metrics["mean_prob"]
                pct_over_50 = seg_metrics["pct_pixels_over_50"]
                jaywalking = crossing and max_prob < crosswalk_threshold
            else:
                max_prob = None
                mean_prob = None
                pct_over_50 = None
                # Without segmentation data we can't confirm crosswalk presence
                jaywalking = crossing

            rows.append({
                "video_id": video_id,
                "frame": seg_frame,
                "ped_index": ped_idx,
                "crossing_pred": int(pred),
                "crossing_confidence": round(confidence, 4) if confidence is not None else None,
                "bbox": str(bbox) if bbox is not None else "",
                "crosswalk_max_prob": round(max_prob, 4) if max_prob is not None else "",
                "crosswalk_mean_prob": round(mean_prob, 4) if mean_prob is not None else "",
                "crosswalk_pct_over_50": round(pct_over_50, 2) if pct_over_50 is not None else "",
                "jaywalking": jaywalking,
            })

    return rows


def main():
    args = parse_args()

    # Load segmentation data
    if not os.path.isfile(args.segmentation_csv):
        print(f"Error: Segmentation CSV not found: {args.segmentation_csv}")
        return
    print(f"Loading segmentation data from {args.segmentation_csv}")
    seg_data = load_segmentation_data(args.segmentation_csv)
    print(f"  Loaded {len(seg_data)} frame entries")

    # Find intention JSONs
    if not os.path.isdir(args.intention_dir):
        print(f"Error: Intention directory not found: {args.intention_dir}")
        return
    json_files = sorted(
        f for f in os.listdir(args.intention_dir) if f.endswith(".json")
    )
    if not json_files:
        print(f"No JSON files found in {args.intention_dir}")
        return
    print(f"Found {len(json_files)} intention prediction file(s)")

    # Process each video
    all_rows = []
    video_stats = {}
    for json_file in json_files:
        video_id = os.path.splitext(json_file)[0]
        json_path = os.path.join(args.intention_dir, json_file)
        rows = process_intention_json(json_path, video_id, seg_data, args.crosswalk_threshold)
        all_rows.extend(rows)

        crossing_count = sum(1 for r in rows if r["crossing_pred"] == 1)
        jaywalking_count = sum(1 for r in rows if r["jaywalking"])
        video_stats[video_id] = {
            "total_preds": len(rows),
            "crossing": crossing_count,
            "jaywalking": jaywalking_count,
        }

    # Write output CSV
    fieldnames = [
        "video_id", "frame", "ped_index", "crossing_pred", "crossing_confidence",
        "bbox", "crosswalk_max_prob", "crosswalk_mean_prob", "crosswalk_pct_over_50",
        "jaywalking",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Print summary
    total_preds = len(all_rows)
    total_crossing = sum(1 for r in all_rows if r["crossing_pred"] == 1)
    total_jaywalking = sum(1 for r in all_rows if r["jaywalking"])

    print(f"\n{'='*50}")
    print(f"Results written to {args.output}")
    print(f"{'='*50}")
    print(f"Total pedestrian predictions: {total_preds}")
    print(f"Crossing predictions:         {total_crossing}")
    print(f"Jaywalking detections:        {total_jaywalking}")
    print(f"Crosswalk threshold:          {args.crosswalk_threshold}")
    print(f"\nPer-video breakdown:")
    for vid, stats in sorted(video_stats.items()):
        print(
            f"  {vid}: {stats['total_preds']} preds, "
            f"{stats['crossing']} crossing, "
            f"{stats['jaywalking']} jaywalking"
        )


if __name__ == "__main__":
    main()
