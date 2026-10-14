# Figure 1.b: grid of base Stable Diffusion vs LoRA checkpoints (rows) for fixed seeds (columns).
# Reads the images written by scripts/01_lora/evaluate_lora.py.

import os
from PIL import Image, ImageDraw, ImageFont

def make_multirow_grid(
    base_images,
    lora_images_dict,  # {step: [images]}
    seeds,
    font_size=50,
    pad=40
):
    steps = sorted(lora_images_dict.keys())
    cols = len(seeds)
    rows = 1 + len(steps)  # base + all checkpoints

    w, h = base_images[0].size

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Margins
    top_margin = pad * 2
    left_margin = pad * 8
    bottom_margin = pad

    grid_w = cols * w + left_margin + pad
    grid_h = rows * h + top_margin + bottom_margin

    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    # ---- Column labels (seeds) ----
    for i, seed in enumerate(seeds):
        x = left_margin + i * w + w // 2
        y = top_margin - pad
        draw.text((x, y), f"seed_{seed}", fill="black", font=font, anchor="mm")

    # ---- Row labels ----
    row_labels = ["Base SD"] + [f"LoRA {step//1000}K" for step in steps]

    for r, label in enumerate(row_labels):
        y = top_margin + r * h + h // 2

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]

        text_img = Image.new("RGB", (text_w + 20, h), "white")
        text_draw = ImageDraw.Draw(text_img)

        text_draw.text(
            (100, h // 2),
            label,
            fill="black",
            font=font,
            anchor="mm"
        )

        # Rotate 90 degrees
        rotated = text_img.rotate(90, expand=True)

        # Paste onto grid
        grid.paste(rotated, (0, top_margin + r * h))

    # ---- Paste BASE row ----
    for c, img in enumerate(base_images):
        x = left_margin + c * w
        y = top_margin
        grid.paste(img, (x, y))

    # ---- Paste LoRA rows ----
    for r, step in enumerate(steps, start=1):
        imgs = lora_images_dict[step]
        for c, img in enumerate(imgs):
            x = left_margin + c * w
            y = top_margin + r * h
            grid.paste(img, (x, y))

    return grid

OUTPUT_ROOT = "outputs/lora_eus_v1_eval_outputs"
GRID_PATH = os.path.join(OUTPUT_ROOT, "final_comparison_grid.png")

CHECKPOINTS = [10000, 20000, 30000, 40000, 50000]

# Must match the seeds used in evaluate_lora.py
SEEDS = [64, 123, 256, 514, 1054, 2060, 4096, 8200]


def main():
    base_dir = os.path.join(OUTPUT_ROOT, "base")
    base_images = [
        Image.open(os.path.join(base_dir, f"seed_{seed}.png"))
        for seed in SEEDS
    ]

    lora_images_dict = {}
    for step in CHECKPOINTS:
        lora_dir = os.path.join(OUTPUT_ROOT, f"lora_{step}")
        lora_images_dict[step] = [
            Image.open(os.path.join(lora_dir, f"seed_{seed}.png"))
            for seed in SEEDS
        ]

    grid = make_multirow_grid(base_images, lora_images_dict, SEEDS)
    grid.save(GRID_PATH)
    print("Saved:", GRID_PATH)


if __name__ == "__main__":
    main()
