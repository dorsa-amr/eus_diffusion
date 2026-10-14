# Figure 2: one grid per ControlNet validation step. Each row shows the real image with the
# simulated needle, the conditioning mask, and the 4 generated validation images.
#
# Reads <root_dir>/step_<N>/{i}_conditioning.png and {i}_gen_{j}.png (written by the patched
# train_controlnet.py) and <root_dir>/validation_masks_corresponding_images/image{i}_*.png.

import os
import cv2
import matplotlib.pyplot as plt

ROOT_DIR = "outputs/controlnet_eus/validation_images"
REAL_IMG_DIR = os.path.join(ROOT_DIR, "validation_masks_corresponding_images")
SAVE_DIR = os.path.join(ROOT_DIR, "grids")

NUM_MASKS = 4
NUM_GEN = 4


def load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    step_folders = sorted([f for f in os.listdir(ROOT_DIR) if f.startswith("step_")])

    for step in step_folders:
        step_path = os.path.join(ROOT_DIR, step)

        # Columns: real image + mask + generated images
        fig, axes = plt.subplots(NUM_MASKS, NUM_GEN + 2, figsize=(18, 12))

        for i in range(NUM_MASKS):
            candidates = [f for f in os.listdir(REAL_IMG_DIR) if f.startswith(f"image{i}")]
            if len(candidates) == 0:
                raise FileNotFoundError(f"No file found for image{i} in {REAL_IMG_DIR}")

            axes[i, 0].imshow(load_rgb(os.path.join(REAL_IMG_DIR, candidates[0])))
            axes[i, 0].set_title(f"image{i} with needle", fontsize=10)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(load_rgb(os.path.join(step_path, f"{i}_conditioning.png")))
            axes[i, 1].set_title(f"mask {i}", fontsize=10)
            axes[i, 1].axis("off")

            for j in range(NUM_GEN):
                axes[i, j + 2].imshow(load_rgb(os.path.join(step_path, f"{i}_gen_{j}.png")))
                axes[i, j + 2].set_title(f"gen {j}", fontsize=10)
                axes[i, j + 2].axis("off")

        plt.suptitle(step, fontsize=16)
        plt.tight_layout()

        save_path = os.path.join(SAVE_DIR, f"{step}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
