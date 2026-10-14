# Extracts a binary fan (field-of-view) mask from each preprocessed EUS image.
# Explored as a conditioning input; not used in the final pipeline.

import cv2
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMG_DIR = str(PROJECT_ROOT / "data" / "processed" / "images")
OUT_DIR = str(PROJECT_ROOT / "data" / "processed" / "conditions" / "fan_mask")


os.makedirs(OUT_DIR, exist_ok=True)

for fname in os.listdir(IMG_DIR):
    img = cv2.imread(os.path.join(IMG_DIR, fname), 0)
    h, w = img.shape

    mask = (img > 5).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15)))

    cv2.imwrite(os.path.join(OUT_DIR, fname), mask)
