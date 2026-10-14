# Runs the original LoRA dataset pipeline: preprocessing, a 90/10 frame-level split and caption files.
# Only the preprocessing step is used by the final pipeline (DreamBooth trains on the whole folder
# with a single instance prompt), so the split and captions are kept here for reference.

import subprocess

print("Step 1: Collecting and preprocessing images...")
subprocess.run(["python", "scripts/01_lora/prepare_dataset.py"], check=True)

print("Step 2: Splitting dataset...")
subprocess.run(["python", "scripts/experiments/split_dataset.py"], check=True)

print("Step 3: Creating captions...")
subprocess.run(["python", "scripts/experiments/make_captions.py"], check=True)

print("Dataset pipeline finished.")
