# Figure 1.a: 2x5 grid of randomly sampled (seeded) preprocessed EUS images.

import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

IMAGE_FOLDER = "data/processed/images"
SAVE_PATH = "outputs/figures/fig1a_dataset_samples.png"


def plot_2x5_images(image_folder, save_path, seed=42):
    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    if len(image_files) < 10:
        raise ValueError("Not enough images in the folder (need at least 10).")

    random.seed(seed)
    selected_files = random.sample(image_files, 10)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axes.flatten()):
        img_np = np.array(Image.open(os.path.join(image_folder, selected_files[i])))

        # Force grayscale if single channel
        if len(img_np.shape) == 2:
            ax.imshow(img_np, cmap='gray')
        else:
            ax.imshow(img_np)

        ax.axis('off')

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    plot_2x5_images(IMAGE_FOLDER, SAVE_PATH)
