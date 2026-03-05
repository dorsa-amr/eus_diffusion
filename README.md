Train a conditional latent diffusion model (Stable Diffusion + ControlNet) to generate full EUS images containing needles, conditioned on explicit needle geometry (e.g., centerline / mask), rather than inserting needles into existing ultrasound images.


(We first fine-tune Stable Diffusion on real EUS pancreas images to match the domain, then train a ControlNet conditioned on needle masks or centerlines to generate full EUS images with needles (not just overlay needles onto real frames).)


# Build dataset

python scripts/build_dataset.py

# Train LoRA

python training/train_lora.py
