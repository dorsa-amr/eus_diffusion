# This code combines the trained ControlNet and LoRA to perform inference on a new prompt and mask. 

import os
import torch
import random
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
from tqdm import tqdm

# Optional (for grid saving)
from torchvision.utils import make_grid
import torchvision.transforms as T

# -------- CONFIG --------
controlnet_path = "outputs/controlnet_eus/checkpoint-30000/controlnet"
lora_path = "outputs/lora_eus_v1/checkpoint-100000"
base_model = "runwayml/stable-diffusion-v1-5"

# mask_dir = "data/controlnet_dataset/test/conditioning_images"
# output_dir = "outputs/inference_results_controlnetpluslora-withoutlora.fuse()"

mask_dir = "data/controlnet_dataset/val/conditioning_images"
output_dir = "outputs/inference_results_controlnetpluslora-withoutlora.fuse()_for_val_set"

num_images_per_mask = 4
num_inference_steps = 30
guidance_scale = 7.5
lora_scale = 0.8

prompt = "An endoscopic ultrasound image from pancreas with a needle"

os.makedirs(output_dir, exist_ok=True)

# -------- LOAD MODELS --------
print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    controlnet_path,
    torch_dtype=torch.float16
)

print("Loading Pipeline...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# -------- LOAD LORA --------
print("Loading LoRA...")
pipe.load_lora_weights(lora_path)
# pipe.fuse_lora(lora_scale=lora_scale) #comment this for Dynamic LoRA, with fusing, LoRA is baked into weights (in memory only)

# Cleaner than fuse, More controllable
# pipe.load_lora_weights(lora_path, weight_name=None, adapter_name="default")
# pipe.set_adapters(["default"], adapter_weights=[lora_scale])

# -------- MEMORY OPTIMIZATION --------
pipe.enable_attention_slicing()

# Optional (use if low VRAM)
# pipe.enable_model_cpu_offload()

# Optional (if xformers works in your env)
try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    print("xFormers not available — skipping")

# -------- GET MASK FILES --------
mask_files = sorted([
    f for f in os.listdir(mask_dir)
    if f.endswith((".png", ".jpg", ".jpeg"))
])

# For limiting the number of masks during testing, comment if you want to process all masks. (IMPORTANT: set random seed for reproducibility)
# random.seed(42)
# mask_files = random.sample(mask_files, 20)

print(f"Found {len(mask_files)} masks")

# -------- MAIN LOOP --------
for idx, mask_name in enumerate(tqdm(mask_files)):

    mask_path = os.path.join(mask_dir, mask_name)

    # Load mask
    mask = Image.open(mask_path).convert("RGB").resize((512, 512))

    # Create output folder per mask
    mask_output_dir = os.path.join(output_dir, f"mask_{idx:03d}")
    os.makedirs(mask_output_dir, exist_ok=True)

    # Save conditioning image
    mask.save(os.path.join(mask_output_dir, "conditioning.png"))

    generated_images = []

    # -------- GENERATE MULTIPLE IMAGES (with different seeds) --------
    for i in range(num_images_per_mask):

        # Different seed per image (IMPORTANT)
        seed = 1000 * idx + i
        generator = torch.Generator(device="cuda").manual_seed(seed)

        image = pipe(
            prompt=prompt,
            image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        # Save image
        image.save(os.path.join(mask_output_dir, f"gen_{i}.png"))

        generated_images.append(image)

    # -------- SAVE GRID --------
    try:
        grid = make_grid(
            [T.ToTensor()(img) for img in generated_images],
            nrow=num_images_per_mask
        )
        T.ToPILImage()(grid).save(
            os.path.join(mask_output_dir, "grid.png")
        )
    except Exception as e:
        print(f"Grid saving failed for {mask_name}: {e}")

print("All masks processed successfully!")