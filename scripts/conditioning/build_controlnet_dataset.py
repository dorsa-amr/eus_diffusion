import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = PROJECT_ROOT / "outputs" / "algorithmic_approach" / "output_dataset_v4"
DEST_ROOT = PROJECT_ROOT / "data" / "controlnet_dataset"
SPLIT_DIR = PROJECT_ROOT / "data" / "controlnet_dataset" / "splits"


def prepare_dirs():

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DEST_ROOT, split, "conditioning"), exist_ok=True)
        os.makedirs(os.path.join(DEST_ROOT, split, "target"), exist_ok=True)


def read_split_file(name):

    path = os.path.join(SPLIT_DIR, f"{name}_patients.txt")

    patients = []

    with open(path, "r") as f:
        for line in f:
            category, patient = line.strip().split("/")
            patients.append((category, patient))

    return patients


def convert_to_rgb(mask_path):

    img = Image.open(mask_path).convert("L")
    arr = np.array(img)

    rgb = np.stack([arr, arr, arr], axis=-1)

    return Image.fromarray(rgb)


def copy_patient_data(split_name, patients):

    for category, patient in patients:

        img_dir = os.path.join(SOURCE_ROOT, "images", category, patient)
        mask_dir = os.path.join(SOURCE_ROOT, "halo_masks", category, patient)

        for fname in os.listdir(img_dir):

            if not fname.endswith(".png"):
                continue

            base = os.path.splitext(fname)[0]

            target_path = os.path.join(img_dir, fname)
            cond_path = os.path.join(mask_dir, base + "_halo.png")

            if not os.path.exists(cond_path):
                continue

            shutil.copy(
                target_path,
                os.path.join(DEST_ROOT, split_name, "target", fname)
            )

            cond_rgb = convert_to_rgb(cond_path)

            cond_rgb.save(
                os.path.join(DEST_ROOT, split_name, "conditioning", fname)
            )


def main():

    prepare_dirs()

    for split in ["train", "val", "test"]:

        patients = read_split_file(split)

        copy_patient_data(split, patients)

        print(f"{split} set created with {len(patients)} patients.")


if __name__ == "__main__":
    main()