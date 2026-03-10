from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMG_DIR = PROJECT_ROOT / "data" / "processed" / "images"

CAPTION = "endoscopic ultrasound image"

for img_path in IMG_DIR.glob("*.png"):

    txt_path = img_path.with_suffix(".txt")

    with open(txt_path, "w") as f:
        f.write(CAPTION)

print("Captions created.")
