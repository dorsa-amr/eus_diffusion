# Fuses a LoRA checkpoint into Stable Diffusion v1.5 and saves the merged model, used as the
# base model for train_controlnet_merged.sh. Not part of the final pipeline.

from diffusers import StableDiffusionPipeline
import torch

base_model = "runwayml/stable-diffusion-v1-5"
lora_path = "outputs/lora_eus_v1/checkpoint-90000"
save_path = "outputs/sd_eus_merged"

# 1. Load base model (IMPORTANT: float32 for stability)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float32
).to("cuda")

# 2. Load LoRA
pipe.load_lora_weights(lora_path)

print("LoRA loaded")

# 3. Fuse LoRA into model
pipe.fuse_lora()

print("LoRA fused")

# 4. Move to CPU before saving (IMPORTANT)
pipe.to("cpu")

# 5. Save full merged model
pipe.save_pretrained(save_path)

print("Model saved successfully")
