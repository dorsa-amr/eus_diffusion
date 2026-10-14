# EUS Diffusion

Generation of endoscopic ultrasound (EUS) images with needle structures using Stable Diffusion v1.5,
LoRA and ControlNet.

1. **LoRA**: Stable Diffusion v1.5 is fine-tuned with DreamBooth-LoRA on the Endoscopic Ultrasound
   Database of the Pancreas (Jaramillo et al., SIPAIM 2020) so that it learns how EUS images look.
2. **Needle simulation**: a physics-inspired algorithm inserts synthetic needles into the EUS frames and
   produces matching needle masks.
3. **ControlNet**: ControlNet is trained to generate EUS images with needles from the needle masks.
4. **Inference**: ControlNet and the LoRA are combined to generate several EUS images per needle mask.

## Repository layout

```
scripts/
├── 01_lora/
│   ├── prepare_dataset.py           # resize + zero-pad TIFF frames to 512x512 PNGs
│   ├── train_dreambooth_lora.py     # diffusers v0.25.0 example, LoRA params cast to FP32
│   ├── train_lora.sh                # LoRA training command
│   └── evaluate_lora.py             # base SD vs LoRA checkpoints, fixed seeds
├── 02_needle_simulation/
│   ├── simulate_needles.py          # synthetic needles + core/halo masks
│   └── visualize_simulation.py      # per-patient sanity-check grids
├── 03_controlnet/
│   ├── split_by_patients.py         # 70/15/15 patient-level split
│   ├── build_controlnet_dataset.py  # train/val/test sets + metadata.jsonl
│   ├── train_controlnet.patch       # our changes to diffusers' train_controlnet.py
│   └── train_controlnet.sh          # ControlNet training command
├── 04_inference/
│   └── inference.py                 # ControlNet + LoRA generation from needle masks
├── figures/                         # scripts for the report figures (1.a, 1.b, 2, 3)
└── experiments/                     # explored ideas, not part of the final pipeline
```

`data/` and `outputs/` hold the datasets, checkpoints and generated images. They are not version-controlled.

## Setup

```bash
conda create -n eus_diffusion python=3.10 && conda activate eus_diffusion
pip install -r requirements.txt

# diffusers v0.27.2 from source, with our patch for the ControlNet training script
git clone --branch v0.27.2 https://github.com/huggingface/diffusers.git
git -C diffusers apply ../scripts/03_controlnet/train_controlnet.patch
pip install -e diffusers
```

The patch makes `train_controlnet.py` load conditioning masks from the file paths in `metadata.jsonl`, and
save validation images to `<output_dir>/validation_images/step_<N>/`.

## Pipeline

Run all commands from the project root.

### 1. LoRA fine-tuning

```bash
python scripts/01_lora/prepare_dataset.py --src_root /path/to/Endoscopic_pancreas_data   # -> data/processed/images
bash scripts/01_lora/train_lora.sh                                                       # -> outputs/lora_eus_v1
python scripts/01_lora/evaluate_lora.py                                                  # -> outputs/lora_eus_v1_eval_outputs
```

Training settings: SD v1.5, 512×512, batch size 1, gradient accumulation 2, LR 2e-5 (cosine, 2000 warmup steps),
100k steps, checkpoint every 10k, LoRA rank 8, FP16, instance prompt `"endoscopic ultrasound image"`.

### 2. Needle simulation

```bash
python scripts/02_needle_simulation/simulate_needles.py       # -> outputs/algorithmic_approach/output_dataset_v4
python scripts/02_needle_simulation/visualize_simulation.py   # optional sanity check
```

### 3. ControlNet

```bash
python scripts/03_controlnet/split_by_patients.py         # -> data/controlnet_dataset/splits
python scripts/03_controlnet/build_controlnet_dataset.py  # -> data/controlnet_dataset/{train,val,test,validation_masks}
bash scripts/03_controlnet/train_controlnet.sh            # -> outputs/controlnet_eus
```

Every 3rd frame is kept, and halo masks are used as conditioning images. Training settings: SD v1.5, 512×512,
batch size 1, gradient accumulation 4, LR 1e-5, FP16, gradient checkpointing, 30k steps, checkpoint every 2k,
validation every 1k, 10% empty prompts. Prompt: `"An endoscopic ultrasound image from pancreas with a needle"`.

### 4. Inference

```bash
python scripts/04_inference/inference.py                   # test-set masks -> outputs/inference_results
python scripts/04_inference/inference.py --lora_scale 0.8  # weaker LoRA (default 1.0)
python scripts/04_inference/inference.py --help            # all options
```

Defaults: ControlNet `checkpoint-30000`, LoRA `checkpoint-100000`, 30 steps, guidance 7.5, 4 images per mask,
seed `1000 * mask_index + i`. Some outputs can come out black because Stable Diffusion's NSFW filter is wrongly
triggered by ultrasound images. Use `--disable_safety_checker` to turn the filter off.

### Figures

| Figure | Script |
|---|---|
| 1.a: dataset samples | `scripts/figures/plot_dataset_samples.py` |
| 1.b: base SD vs LoRA checkpoints | `scripts/figures/make_lora_checkpoint_grid.py` |
| 2: ControlNet validation over training | `scripts/figures/make_controlnet_validation_grids.py` |
| 3: inference results | `scripts/figures/make_inference_grids.py` |
