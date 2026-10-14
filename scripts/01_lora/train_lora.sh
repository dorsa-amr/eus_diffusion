#!/usr/bin/env bash
# DreamBooth-LoRA fine-tuning of Stable Diffusion v1.5 on EUS pancreas images (report §1.2).
# Run from the project root. train_dreambooth_lora.py is the diffusers v0.25.0 example script,
# modified to cast the trainable LoRA parameters to FP32 (required for FP16 mixed precision).
#
# The original run wrote to outputs/lora_eus_maxrealism_v2, which was later renamed to outputs/lora_eus_v1.

python scripts/01_lora/train_dreambooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="data/processed/images" \
  --instance_prompt="endoscopic ultrasound image" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --learning_rate=2e-5 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=2000 \
  --max_train_steps=100000 \
  --checkpointing_steps=10000 \
  --rank=8 \
  --mixed_precision="fp16" \
  --report_to="tensorboard" \
  --output_dir="outputs/lora_eus_v1"
