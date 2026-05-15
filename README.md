EUS Diffusion is a pipeline for fine-tuning Stable Diffusion v1.5 with LoRA on endoscopic ultrasound images and training ControlNet to generate needle-inserted EUS images.

## Pipeline

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Build the ControlNet dataset
   ```bash
   python scripts/conditioning/build_controlnet_dataset.py
   ```
3. Train ControlNet
   ```bash
   python diffusers/examples/controlnet/train_controlnet.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --output_dir outputs/controlnet_eus ...
   ```
4. Run inference
   ```bash
   python scripts/inference.py
   ```

## Notes

- `data/` and `outputs/` are excluded from version control.
- Visualization and temporary utility scripts are ignored.
- Use `outputs/controlnet_eus` for ControlNet checkpoints and `outputs/lora_eus_v1` for LoRA weights.
