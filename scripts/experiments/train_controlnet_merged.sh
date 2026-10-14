#!/usr/bin/env bash
# EXPERIMENT (not in the report): ControlNet trained on top of the LoRA-merged model
# produced by merge_sd_with_lora.py instead of the base Stable Diffusion v1.5.

# LD_PRELOAD works around a libstdc++ version conflict in the conda env.
LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 diffusers/examples/controlnet/train_controlnet.py \
  --pretrained_model_name_or_path outputs/sd_eus_merged \
  --output_dir outputs/controlnet_eus_merged_models \
  --train_data_dir data/controlnet_dataset/train \
  --resolution 512 \
  --learning_rate 1e-5 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --max_train_steps 30000 \
  --checkpointing_steps 2000 \
  --validation_steps 1000 \
  --validation_prompt "An endoscopic ultrasound image from pancreas with a needle" \
  --validation_image \
  data/controlnet_dataset/validation_masks/mask1_cancer_C11_V3_0074.png \
  data/controlnet_dataset/validation_masks/mask2_cancer_C16_V2_0528.png \
  data/controlnet_dataset/validation_masks/mask3_healthy_H18_V1_1081.png \
  data/controlnet_dataset/validation_masks/mask4_pancreatitis_P02_V4_0863.png \
  --num_validation_images 4 \
  --report_to tensorboard \
  --image_column "image" \
  --conditioning_image_column "conditioning_image" \
  --caption_column "text" \
  --proportion_empty_prompts 0.1 \
  --tracker_project_name controlnet_eus
