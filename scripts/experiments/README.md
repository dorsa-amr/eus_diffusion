# Experiments

Scripts that were explored during the project but are **not** part of the pipeline described in the report.
Run them from the project root.

| Script | Purpose |
|---|---|
| `build_lora_dataset.py` | Original dataset pipeline: preprocessing + frame-level split + captions |
| `split_dataset.py` | 90/10 frame-level split for LoRA (unused: DreamBooth trains on all images) |
| `make_captions.py` | Per-image caption `.txt` files (unused: DreamBooth uses `--instance_prompt`) |
| `make_fan_mask.py` | Binary fan/FOV masks, explored as a conditioning signal |
| `merge_sd_with_lora.py` | Fuses the EUS LoRA into SD v1.5 and saves it to `outputs/sd_eus_merged` |
| `train_controlnet_merged.sh` | ControlNet trained on the LoRA-merged model instead of base SD v1.5 |
