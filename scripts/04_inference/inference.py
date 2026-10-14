# Combines the trained ControlNet with the EUS LoRA to generate EUS images with needles from
# needle masks (report §3). The LoRA is applied dynamically (not fused into the weights).
#
# Output layout (read by scripts/figures/make_inference_grids.py):
#   <output_dir>/mask_<idx>/conditioning.png, gen_0.png ... gen_<N-1>.png, grid.png
# where <idx> is the index of the mask in the sorted mask directory.

import argparse
import os
import torch
import random
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
from tqdm import tqdm

from torchvision.utils import make_grid
import torchvision.transforms as T

# -------- CONFIG --------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
PROMPT = "An endoscopic ultrasound image from pancreas with a needle"


def parse_args():
    parser = argparse.ArgumentParser(description="ControlNet + LoRA inference on needle masks.")
    parser.add_argument("--controlnet_path", default="outputs/controlnet_eus/checkpoint-30000/controlnet")
    parser.add_argument("--lora_path", default="outputs/lora_eus_v1/checkpoint-100000")
    parser.add_argument("--mask_dir", default="data/controlnet_dataset/test/conditioning_images")
    parser.add_argument("--output_dir", default="outputs/inference_results")
    parser.add_argument("--num_images_per_mask", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    # Strength of the LoRA (EUS appearance). 1.0 = full LoRA, 0.0 = base Stable Diffusion.
    # Change it here or pass e.g. --lora_scale 0.8 to weaken the ultrasound adaptation.
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--num_masks", type=int, default=None,
                        help="Randomly sample this many masks (seeded) instead of processing all of them.")
    parser.add_argument("--disable_safety_checker", action="store_true",
                        help="Disable the NSFW filter, which wrongly blacks out some ultrasound images.")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------- LOAD MODELS --------
    print("Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_path,
        torch_dtype=torch.float16
    )

    print("Loading Pipeline...")
    extra_kwargs = {"safety_checker": None} if args.disable_safety_checker else {}
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        **extra_kwargs
    ).to("cuda")

    # -------- LOAD LORA --------
    # Not fused: the LoRA strength is set per call via cross_attention_kwargs={"scale": lora_scale}.
    print("Loading LoRA...")
    pipe.load_lora_weights(args.lora_path)

    # -------- MEMORY OPTIMIZATION --------
    pipe.enable_attention_slicing()

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        print("xFormers not available — skipping")

    # -------- GET MASK FILES --------
    mask_files = sorted([
        f for f in os.listdir(args.mask_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ])

    if args.num_masks is not None:
        random.seed(42)
        mask_files = random.sample(mask_files, args.num_masks)

    print(f"Found {len(mask_files)} masks")

    # -------- MAIN LOOP --------
    for idx, mask_name in enumerate(tqdm(mask_files)):

        mask_path = os.path.join(args.mask_dir, mask_name)

        mask = Image.open(mask_path).convert("RGB").resize((512, 512))

        mask_output_dir = os.path.join(args.output_dir, f"mask_{idx:03d}")
        os.makedirs(mask_output_dir, exist_ok=True)

        mask.save(os.path.join(mask_output_dir, "conditioning.png"))

        generated_images = []

        # -------- GENERATE MULTIPLE IMAGES (different seed per image) --------
        for i in range(args.num_images_per_mask):

            seed = 1000 * idx + i
            generator = torch.Generator(device="cuda").manual_seed(seed)

            image = pipe(
                prompt=PROMPT,
                image=mask,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                cross_attention_kwargs={"scale": args.lora_scale},
                generator=generator
            ).images[0]

            image.save(os.path.join(mask_output_dir, f"gen_{i}.png"))

            generated_images.append(image)

        # -------- SAVE GRID --------
        grid = make_grid(
            [T.ToTensor()(img) for img in generated_images],
            nrow=args.num_images_per_mask
        )
        T.ToPILImage()(grid).save(os.path.join(mask_output_dir, "grid.png"))

    print("All masks processed successfully!")


if __name__ == "__main__":
    main()
