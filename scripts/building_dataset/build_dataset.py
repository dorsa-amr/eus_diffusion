import subprocess

print("Step 1: Collecting and preprocessing images...")
subprocess.run(["python", "scripts/prepare_dataset.py"], check=True)

print("Step 2: Splitting dataset...")
subprocess.run(["python", "scripts/split_dataset.py"], check=True)

print("Step 3: Creating captions...")
subprocess.run(["python", "scripts/make_captions.py"], check=True)

print("Dataset pipeline finished.")
