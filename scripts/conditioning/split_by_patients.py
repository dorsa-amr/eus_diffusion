import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = PROJECT_ROOT / "outputs" / "algorithmic_approach" / "output_dataset_v4"
SPLIT_DIR = PROJECT_ROOT / "data" / "controlnet_dataset" / "splits"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)


def collect_patients():

    patients = []

    images_root = os.path.join(SOURCE_ROOT, "images")

    for category in os.listdir(images_root):

        cat_path = os.path.join(images_root, category)

        if not os.path.isdir(cat_path):
            continue

        for patient in os.listdir(cat_path):
            patients.append((category, patient))

    return patients


def save_split_file(name, patients):

    os.makedirs(SPLIT_DIR, exist_ok=True)

    path = os.path.join(SPLIT_DIR, f"{name}_patients.txt")

    with open(path, "w") as f:
        for category, patient in patients:
            f.write(f"{category}/{patient}\n")


def main():

    patients = collect_patients()

    random.shuffle(patients)

    n = len(patients)

    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_patients = patients[:train_end]
    val_patients = patients[train_end:val_end]
    test_patients = patients[val_end:]

    save_split_file("train", train_patients)
    save_split_file("val", val_patients)
    save_split_file("test", test_patients)

    print("Patient splits created.")
    print(f"Train: {len(train_patients)}")
    print(f"Val: {len(val_patients)}")
    print(f"Test: {len(test_patients)}")


if __name__ == "__main__":
    main()