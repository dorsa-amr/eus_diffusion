import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# Configuration
# ============================================================

OUTPUT_ROOT = "/home/dorsa/Uni-Concordia/Mitacs/eus_diffusion/algorithmic_approach/output_dataset_v4"
SAVE_ROOT = os.path.join("/home/dorsa/Uni-Concordia/Mitacs/eus_diffusion/algorithmic_approach", "needle_simulation_visualization_v4")

N_FRAMES = 4
RANDOM_SEED = 0

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# Visualization
# ============================================================

def visualize_subject(
    subject_img_path,
    subject_mask_path,
    category,
    subject_name,
    save_dir
):
    img_files = sorted([
        f for f in os.listdir(subject_img_path)
        if f.lower().endswith(".png")
    ])

    if len(img_files) < N_FRAMES:
        return

    start = random.randint(0, len(img_files) - N_FRAMES)
    selected = img_files[start:start + N_FRAMES]

    fig, axes = plt.subplots(
        2, N_FRAMES,
        figsize=(4 * N_FRAMES, 6),
        # constrained_layout=True
    )

    fig.suptitle(f"{category} — {subject_name}", fontsize=16)

    for i, fname in enumerate(selected):
        img = np.array(Image.open(os.path.join(subject_img_path, fname)))

        mask_name = fname.replace(".png", "_core.png")
        mask_path = os.path.join(subject_mask_path, mask_name)

        if not os.path.exists(mask_path):
            mask = np.zeros_like(img)
        else:
            mask = np.array(Image.open(mask_path))

        # Images row
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("images", fontsize=12)

        # Masks row
        axes[1, i].imshow(mask, cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_xlabel(fname, fontsize=12, rotation=25)
        if i == 0:
            axes[1, i].set_ylabel("masks", fontsize=12)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{category}_{subject_name}.png")
    
    plt.subplots_adjust(bottom=0.22, hspace=0.05)
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Main loop
# ============================================================

def main():
    img_root = os.path.join(OUTPUT_ROOT, "images")
    mask_root = os.path.join(OUTPUT_ROOT, "core_masks")

    for category in sorted(os.listdir(img_root)):
        category_img_path = os.path.join(img_root, category)
        category_mask_path = os.path.join(mask_root, category)

        if not os.path.isdir(category_img_path):
            continue

        print(f"[INFO] Processing category: {category}")

        for subject in sorted(os.listdir(category_img_path)):
            subject_img_path = os.path.join(category_img_path, subject)
            subject_mask_path = os.path.join(category_mask_path, subject)

            if not os.path.isdir(subject_img_path):
                continue

            visualize_subject(
                subject_img_path,
                subject_mask_path,
                category,
                subject,
                save_dir=os.path.join(SAVE_ROOT, category)
            )


if __name__ == "__main__":
    main()
    print("Visualization completed.")
