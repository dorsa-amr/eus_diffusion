from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted([p.name for p in IMG_DIR.glob("*.png")])

random.seed(42)
random.shuffle(files)

split = int(0.9 * len(files))

with open(SPLIT_DIR / "train.txt", "w") as f:
    f.write("\n".join(files[:split]))

with open(SPLIT_DIR / "val.txt", "w") as f:
    f.write("\n".join(files[split:]))

print("Dataset split created.")
