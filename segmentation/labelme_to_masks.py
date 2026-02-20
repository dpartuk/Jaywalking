"""Convert LabelMe JSON annotations to binary crosswalk masks.

Reads JSONs from segmentation/jaad_finetune/labels/ and saves
binary PNG masks (0=background, 255=crosswalk) to
segmentation/jaad_finetune/masks/ with matching filenames.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(SCRIPT_DIR, "jaad_finetune", "labels")
MASKS_DIR = os.path.join(SCRIPT_DIR, "jaad_finetune", "masks")


def main():
    if not os.path.isdir(LABELS_DIR):
        raise FileNotFoundError(f"Labels directory not found: {LABELS_DIR}")

    os.makedirs(MASKS_DIR, exist_ok=True)

    jsons = [f for f in sorted(os.listdir(LABELS_DIR)) if f.endswith(".json")]
    if not jsons:
        print(f"No JSON files found in {LABELS_DIR}")
        return

    print(f"Found {len(jsons)} annotation files.")

    for jf in jsons:
        with open(os.path.join(LABELS_DIR, jf)) as f:
            data = json.load(f)

        h = data["imageHeight"]
        w = data["imageWidth"]

        # Black background
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        for shape in data["shapes"]:
            if shape["label"].lower() != "crosswalk":
                continue

            points = shape["points"]
            shape_type = shape.get("shape_type", "polygon")

            if shape_type in ("polygon", "linestrip"):
                # Flatten list of [x, y] to list of tuples
                # linestrip is treated as a closed polygon
                poly = [(p[0], p[1]) for p in points]
                if len(poly) >= 3:
                    draw.polygon(poly, fill=255)
            elif shape_type == "rectangle":
                # Two corner points
                x0, y0 = points[0]
                x1, y1 = points[1]
                draw.rectangle([x0, y0, x1, y1], fill=255)

        # Save mask with same stem as the original image
        # Handle Windows-style backslashes in imagePath from LabelMe annotations
        img_name = Path(data["imagePath"].replace("\\", "/")).stem
        out_path = os.path.join(MASKS_DIR, f"{img_name}.png")
        mask.save(out_path)
        print(f"  {jf} -> {img_name}.png  ({np.array(mask).sum() // 255} crosswalk pixels)")

    print(f"Done. Masks saved to {MASKS_DIR}")


if __name__ == "__main__":
    main()
