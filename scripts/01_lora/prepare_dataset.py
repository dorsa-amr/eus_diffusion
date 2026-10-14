# Preprocesses the Endoscopic Ultrasound Database of the Pancreas for LoRA training.
# Each TIFF frame is resized so its longer side is 512 px, then zero-padded to 512x512
# (aspect ratio preserved) and saved as <label>_<patient>_<frame>.png.

import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SRC_ROOT = PROJECT_ROOT.parent / "Endoscopic_pancreas_data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "images"

TARGET_SIZE = 512

mapping = {
    "Cancer_pancreas": "cancer",
    "Healthy_pancreas": "healthy",
    "Pancreatitis": "pancreatitis",
}

# --------------------------------------------------
# Resize function
# --------------------------------------------------

def resize_with_padding(img, target_size=512):

    h, w = img.shape

    scale = target_size / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)

    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

    return canvas


# --------------------------------------------------
# Dataset construction
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Resize and zero-pad EUS frames to 512x512.")
    parser.add_argument("--src_root", type=Path, default=DEFAULT_SRC_ROOT,
                        help="Root of the raw dataset (contains Cancer_pancreas/, Healthy_pancreas/, Pancreatitis/).")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for folder, label in mapping.items():

        base = args.src_root / folder / "Original_Images"

        for patient in base.iterdir():

            if not patient.is_dir():
                continue

            for img_path in tqdm(list(patient.glob("*.tif")), desc=f"{label}/{patient.name}"):

                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                img = resize_with_padding(img, TARGET_SIZE)

                new_name = f"{label}_{patient.name}_{img_path.stem}.png"

                cv2.imwrite(str(args.out_dir / new_name), img)


if __name__ == "__main__":
    main()
