# Generates images from the base Stable Diffusion v1.5 model and from each LoRA checkpoint
# with the same fixed seeds, so checkpoints can be compared side by side (report Figure 1.b).
#
# Output layout (read by scripts/figures/make_lora_checkpoint_grid.py):
#   <output_root>/base/seed_<seed>.png
#   <output_root>/lora_<step>/seed_<seed>.png

import argparse
import os
import torch
from diffusers import StableDiffusionPipeline

# -------- CONFIG --------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

PROMPT = "endoscopic ultrasound image of the pancreas, fan-shaped scan"
NEG_PROMPT = (
    "photograph, camera, probe, equipment, text, labels, "
    "watermark, UI, monitor, endoscope"
)

# Seeds used for Figure 1.b
SEEDS = [64, 123, 256, 514, 1054, 2060, 4096, 8200]
CHECKPOINTS = [10000, 20000, 30000, 40000, 50000]

NUM_STEPS = 30
CFG = 6.5
HEIGHT = WIDTH = 512


def generate(pipe, seeds):
    images = []
    for seed in seeds:
        g = torch.Generator(device="cuda").manual_seed(seed)
        img = pipe(
            PROMPT,
            negative_prompt=NEG_PROMPT,
            num_inference_steps=NUM_STEPS,
            guidance_scale=CFG,
            height=HEIGHT,
            width=WIDTH,
            generator=g
        ).images[0]
        images.append(img)
    return images


def save_images(images, seeds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for seed, img in zip(seeds, images):
        img.save(os.path.join(out_dir, f"seed_{seed}.png"))


def main():
    parser = argparse.ArgumentParser(description="Generate base SD and LoRA checkpoint images with fixed seeds.")
    parser.add_argument("--lora_root", default="outputs/lora_eus_v1")
    parser.add_argument("--output_root", default="outputs/lora_eus_v1_eval_outputs")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=CHECKPOINTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_attention_slicing()

    print("Generating base SD images...")
    save_images(generate(pipe, args.seeds), args.seeds, os.path.join(args.output_root, "base"))

    for step in args.checkpoints:
        print(f"Evaluating LoRA checkpoint {step}")
        pipe.load_lora_weights(os.path.join(args.lora_root, f"checkpoint-{step}"))
        save_images(generate(pipe, args.seeds), args.seeds, os.path.join(args.output_root, f"lora_{step}"))
        pipe.unload_lora_weights()

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
