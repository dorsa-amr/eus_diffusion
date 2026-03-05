from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw" / "eus_images"
PROCESSED_DIR = DATA_DIR / "processed" / "images"

SRC_ROOT = Path("/home/dorsa/Uni-Concordia/Mitacs/Endoscopic_pancreas_data")

TARGET_SIZE = 512

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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

for folder, label in mapping.items():

    base = SRC_ROOT / folder / "Original_Images"

    for patient in base.iterdir():

        if not patient.is_dir():
            continue

        for img_path in tqdm(list(patient.glob("*.tif"))):

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = resize_with_padding(img, TARGET_SIZE)

            new_name = f"{label}_{patient.name}_{img_path.stem}.png"

            cv2.imwrite(str(PROCESSED_DIR / new_name), img)
