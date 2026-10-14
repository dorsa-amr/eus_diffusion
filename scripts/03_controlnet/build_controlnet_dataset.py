# Builds the ControlNet dataset (report §2.1, "Dataset Preparation") from the output of
# scripts/02_needle_simulation/simulate_needles.py and the patient splits from split_by_patients.py.
#
# Every FRAME_STRIDE-th frame of each patient is kept. Targets are the frames with the simulated
# needle; conditioning images are the halo masks converted to RGB.
#
# Output layout (under DEST_ROOT):
#   train/<frame>.png, train/mask_<frame>.png, train/metadata.jsonl   (format read by train_controlnet.py)
#   val/images/<frame>.png,  val/conditioning_images/<frame>.png
#   test/images/<frame>.png, test/conditioning_images/<frame>.png
#   validation_masks/mask<i>_<frame>.png                              (masks used for validation during training)

import os
import json
import shutil
import numpy as np
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = PROJECT_ROOT / "outputs" / "algorithmic_approach" / "output_dataset_v4"
DEST_ROOT = PROJECT_ROOT / "data" / "controlnet_dataset"
SPLIT_DIR = PROJECT_ROOT / "data" / "controlnet_dataset" / "splits"

FRAME_STRIDE = 3
PROMPT = "An endoscopic ultrasound image from pancreas with a needle"

# Fixed validation-set frames used as --validation_image during ControlNet training
VALIDATION_FRAMES = [
    "cancer_C11_V3_0074",
    "cancer_C16_V2_0528",
    "healthy_H18_V1_1081",
    "pancreatitis_P02_V4_0863",
]


def read_split_file(name):

    path = os.path.join(SPLIT_DIR, f"{name}_patients.txt")

    patients = []

    with open(path, "r") as f:
        for line in f:
            category, patient = line.strip().split("/")
            patients.append((category, patient))

    return patients


def convert_to_rgb(path):

    img = Image.open(path).convert("L")
    arr = np.array(img)

    rgb = np.stack([arr, arr, arr], axis=-1)

    return Image.fromarray(rgb)


def iter_patient_frames(patients):
    """Yields (target_path, halo_mask_path, frame_name) for every FRAME_STRIDE-th frame."""

    for category, patient in patients:

        img_dir = os.path.join(SOURCE_ROOT, "images", category, patient)
        mask_dir = os.path.join(SOURCE_ROOT, "halo_masks", category, patient)

        if not os.path.isdir(img_dir):
            continue

        files = sorted(os.listdir(img_dir))

        for i, fname in enumerate(files):

            if i % FRAME_STRIDE != 0:
                continue

            if not fname.endswith(".png"):
                continue

            base = os.path.splitext(fname)[0]
            cond_path = os.path.join(mask_dir, base + "_halo.png")

            if not os.path.exists(cond_path):
                continue

            yield os.path.join(img_dir, fname), cond_path, fname


def build_train(patients):

    out_dir = os.path.join(DEST_ROOT, "train")
    os.makedirs(out_dir, exist_ok=True)

    n = 0
    with open(os.path.join(out_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:

        for target_path, cond_path, fname in iter_patient_frames(patients):

            mask_name = "mask_" + fname

            convert_to_rgb(target_path).save(os.path.join(out_dir, fname))
            convert_to_rgb(cond_path).save(os.path.join(out_dir, mask_name))

            entry = {
                "file_name": fname,
                "conditioning_image": mask_name,
                "text": PROMPT
            }
            f.write(json.dumps(entry) + "\n")
            n += 1

    return n


def build_eval_split(split_name, patients):

    img_dir = os.path.join(DEST_ROOT, split_name, "images")
    cond_dir = os.path.join(DEST_ROOT, split_name, "conditioning_images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(cond_dir, exist_ok=True)

    n = 0
    for target_path, cond_path, fname in iter_patient_frames(patients):

        shutil.copy(target_path, os.path.join(img_dir, fname))
        convert_to_rgb(cond_path).save(os.path.join(cond_dir, fname))
        n += 1

    return n


def build_validation_masks():

    out_dir = os.path.join(DEST_ROOT, "validation_masks")
    os.makedirs(out_dir, exist_ok=True)

    for i, frame in enumerate(VALIDATION_FRAMES, start=1):
        src = os.path.join(DEST_ROOT, "val", "conditioning_images", frame + ".png")
        shutil.copy(src, os.path.join(out_dir, f"mask{i}_{frame}.png"))


def main():

    n = build_train(read_split_file("train"))
    print(f"train set created with {n} pairs.")

    for split in ["val", "test"]:
        n = build_eval_split(split, read_split_file(split))
        print(f"{split} set created with {n} pairs.")

    build_validation_masks()
    print("Validation masks copied.")


if __name__ == "__main__":
    main()
