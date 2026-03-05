import os
import numpy as np
import cv2
import tifffile as tiff
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib

# ============================================================
# Configuration
# ============================================================

INPUT_ROOT = "/home/dorsa/Uni-Concordia/Mitacs/eus_diffusion/data/processed/images"
OUTPUT_ROOT = "/home/dorsa/Uni-Concordia/Mitacs/eus_diffusion/algorithmic_approach/output_dataset_v4"

# CATEGORY_MAP = {
#     "Cancer_pancreas": "cancer",
#     "Healthy_pancreas": "healthy",
#     "Pancreatitis": "pancreatitis"
# }

FOV_THRESHOLD = 10
MIN_CENTERLINE_RATIO = 0.85
MAX_ATTEMPTS = 60

ENTRY_JITTER_PX = 2
ANGLE_JITTER_DEG = 2
INTENSITY_JITTER = 0.1

MAX_FRAMES_PER_PATIENT = None  #10 # set None to use all frames

N_VISUALIZE = 5   # number of frames to visualize per patient


# ============================================================
# I/O
# ============================================================

def read_gray_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"]:
        arr = tiff.imread(path)
        if arr.ndim > 2:
            arr = arr[0]
        arr = arr.astype(np.float32)
        arr = 255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return arr.astype(np.uint8)
    return np.array(Image.open(path).convert("L"))


def ensure_dirs(base, category, patient):
    for sub in ["images", "core_masks", "halo_masks", "centerlines", "tips"]:
        os.makedirs(os.path.join(base, sub, category, patient), exist_ok=True)


# ============================================================
# FOV utilities
# ============================================================

def compute_fov_mask(image):
    return image > FOV_THRESHOLD


def get_fov_boundary(mask):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    boundary = mask & (~eroded.astype(bool))
    ys, xs = np.where(boundary)
    return list(zip(xs, ys))


def centerline_inside_ratio(points, fov_mask):
    inside = [
        fov_mask[y, x]
        for x, y in points
        if 0 <= x < fov_mask.shape[1] and 0 <= y < fov_mask.shape[0]
    ]
    return np.mean(inside)


# ============================================================
# Geometry
# ============================================================

def generate_centerline_from_state(h, w, fov_mask, state):
    x0, y0 = state["entry"]
    angle = state["angle"]
    length = state["length"]
    bend = state["bend"]

    t = np.linspace(0, 1, length)
    xs = x0 + t * length * np.cos(angle)
    ys = y0 + t * length * np.sin(angle)
    ys += bend * np.sin(np.pi * t)

    points = np.stack([xs, ys], axis=1).astype(int)

    mask = (
        (points[:, 0] >= 0) & (points[:, 0] < w) &
        (points[:, 1] >= 0) & (points[:, 1] < h)
    )
    points = points[mask]

    return points, tuple(points[-1])



def sample_base_needle_state(image, rng):
    h, w = image.shape
    fov_mask = compute_fov_mask(image)
    boundary = get_fov_boundary(fov_mask)

    if len(boundary) < 50:
        raise RuntimeError("FOV too small for needle simulation")


    for attempt in range(MAX_ATTEMPTS):
        # entry = boundary[rng.integers(len(boundary))]

        # Keep only boundary points near top region
        top_boundary = [
            pt for pt in boundary
            if pt[1] < int(h * 0.30)  # top 30% of image
        ]

        if len(top_boundary) < 20:
            continue

        entry = top_boundary[rng.integers(len(top_boundary))]

        cx, cy = w // 2, h // 2
        base_angle = np.arctan2(cy - entry[1], cx - entry[0])
        base_angle += np.deg2rad(rng.uniform(-15, 15)) # We reduced jitter here

        max_possible = max_length_inside_fov(entry, base_angle, fov_mask)

        if max_possible < 40: #0.1 * h
            continue

        min_len = max(40, int(max_possible * 0.55))
        max_len = int(max_possible * 0.95)

        if min_len >= max_len:
            continue

        state = {
            "entry": entry,
            "angle": base_angle,
            "length": rng.integers(min_len, max_len),
            "bend": rng.uniform(-5, 5),
            "thickness": rng.integers(4, 7),
            "params": {
                "core_weight": rng.uniform(18, 40),
                "halo_ratio": rng.uniform(0.08, 0.2),
                "dropout_prob": 0.03,
                "noise_std": 3
            }
        }

        pts, _ = generate_centerline_from_state(h, w, fov_mask, state)

        # progressively relax constraint
        required_ratio = max(0.65, MIN_CENTERLINE_RATIO - attempt * 0.03)

        if centerline_inside_ratio(pts, fov_mask) >= required_ratio:
            return state


    raise RuntimeError("Could not sample valid base needle state")


def jitter_state(state,rng):
    s = state.copy()
    s["entry"] = (
        state["entry"][0] + rng.integers(-ENTRY_JITTER_PX, ENTRY_JITTER_PX + 1),
        state["entry"][1] + rng.integers(-ENTRY_JITTER_PX, ENTRY_JITTER_PX + 1)
    )
    s["angle"] = state["angle"] + np.deg2rad(
        rng.uniform(-ANGLE_JITTER_DEG, ANGLE_JITTER_DEG)
    )

    p = state["params"].copy()
    p["core_weight"] *= rng.uniform(1 - INTENSITY_JITTER, 1 + INTENSITY_JITTER)
    s["params"] = p
    return s


# ============================================================
# Rendering
# ============================================================

def draw_masks(shape, points, thickness):
    h, w = shape
    core = np.zeros((h, w), np.uint8)
    halo = np.zeros((h, w), np.uint8)
    for i in range(len(points) - 1):
        cv2.line(core, tuple(points[i]), tuple(points[i + 1]), 255, thickness)
        halo_thickness = int(thickness * 2)
        cv2.line(halo, tuple(points[i]), tuple(points[i + 1]), 255, halo_thickness)
    return core, halo

def max_length_inside_fov(entry, angle, fov_mask, max_cap=400):
    h, w = fov_mask.shape
    x0, y0 = entry

    step = 1.0
    length = 0.0

    while length < max_cap:
        x = int(x0 + length * np.cos(angle))
        y = int(y0 + length * np.sin(angle))

        if x < 0 or x >= w or y < 0 or y >= h:
            break

        if not fov_mask[y, x]:
            break

        length += step

    # return int(length)
    return int(length * 0.95) ##That prevents needle tip from touching edge exactly.


def apply_appearance(image, core, halo, params, rng):
    img = image.astype(np.float32).copy()
    h, w = img.shape

    cx = w // 2
    cy = 0

    # Precompute Sobel once
    gx = cv2.Sobel(core.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(core.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

    ys, xs = np.where(core > 0)

    for idx, (y, x) in enumerate(zip(ys, xs)):

        beam_vec = np.array([x - cx, y - cy], dtype=np.float32)
        beam_vec /= (np.linalg.norm(beam_vec) + 1e-6)

        needle_vec = np.array([gx[y, x], gy[y, x]], dtype=np.float32)
        needle_vec /= (np.linalg.norm(needle_vec) + 1e-6)

        alignment = abs(np.dot(beam_vec, needle_vec))

        gamma = 0.8
        min_vis = 0.25

        shaft_position = idx / (len(ys) + 1e-6)

        visibility = min_vis + (1 - min_vis) * ((1 - alignment) ** gamma)
        
        longitudinal_factor = 0.6 + 0.4 * np.sin(np.pi * shaft_position)
        visibility *= longitudinal_factor

        depth_factor = 1 - (y / h) * 0.3
        
        visibility *= depth_factor

        core_intensity = params["core_weight"] * visibility
        img[y, x] += core_intensity

    # ---- Halo: local-only blur ----
    halo_region = halo > 0

    # Extract local patch for halo only
    halo_pixels = img.copy()
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    img[halo_region] = (
        img[halo_region] * 0.6 +
        blurred[halo_region] * 0.4
    )

    # ---- Add small noise only in halo ----
    noise = rng.normal(0, params["noise_std"], img.shape)
    img[halo_region] += noise[halo_region]

    return np.clip(img, 0, 255).astype(np.uint8)

# ============================================================
# Visualization (simple)
# ============================================================

def visualize_image_and_mask(image, mask, title_prefix=""):
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray")
    plt.title(f"{title_prefix} - Image")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(4, 4))
    plt.imshow(mask, cmap="gray")
    plt.title(f"{title_prefix} - Mask")
    plt.axis("off")
    plt.show()


# ============================================================
# Main pipeline
# ============================================================

def group_images_by_patient(input_dir):
    groups = {}

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".png"):
            continue

        name = os.path.splitext(fname)[0]
        parts = name.split("_")

        # Example: cancer_C01_V1_0164
        category = parts[0]            # cancer
        patient_id = parts[1]          # C01

        key = (category, patient_id)

        if key not in groups:
            groups[key] = []

        groups[key].append(fname)

    # Sort frames per patient
    for key in groups:
        groups[key] = sorted(groups[key])

    return groups


def process_all_patients():
    groups = group_images_by_patient(INPUT_ROOT)

    for (category, patient_id), frames in tqdm(groups.items(), desc="Processing patients"):

        ensure_dirs(OUTPUT_ROOT, category, patient_id)

        seed = int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)

        first_frame = read_gray_image(os.path.join(INPUT_ROOT, frames[0]))

        try:
            base_state = sample_base_needle_state(first_frame, rng)
        except RuntimeError as e:
            print(f"[SKIP PATIENT] {patient_id}: {e}")
            continue

        frames_to_use = frames
        if MAX_FRAMES_PER_PATIENT is not None:
            frames_to_use = frames[:MAX_FRAMES_PER_PATIENT]

        for fname in frames_to_use:
            image = read_gray_image(os.path.join(INPUT_ROOT, fname))

            state = jitter_state(base_state, rng)

            pts, tip = generate_centerline_from_state(
                *image.shape, compute_fov_mask(image), state
            )

            core, halo = draw_masks(image.shape, pts, state["thickness"])
            out = apply_appearance(image, core, halo, state["params"], rng)

            base = os.path.splitext(fname)[0]

            Image.fromarray(out).save(
                os.path.join(OUTPUT_ROOT, "images", category, patient_id, base + ".png")
            )
            Image.fromarray(core).save(
                os.path.join(OUTPUT_ROOT, "core_masks", category, patient_id, base + "_core.png")
            )
            Image.fromarray(halo).save(
                os.path.join(OUTPUT_ROOT, "halo_masks", category, patient_id, base + "_halo.png")
            )
            np.save(
                os.path.join(OUTPUT_ROOT, "centerlines", category, patient_id, base + "_centerline.npy"),
                pts
            )
            with open(
                os.path.join(OUTPUT_ROOT, "tips", category, patient_id, base + "_tip.txt"), "w"
            ) as f:
                f.write(f"{tip[0]} {tip[1]}")

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    process_all_patients()
    print("Synthetic needle dataset generated successfully.")
