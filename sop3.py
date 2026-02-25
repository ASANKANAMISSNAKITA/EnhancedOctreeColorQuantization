import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from math import sqrt
import glob
import os

# ==========================================
# CONFIGURATION: TYPE YOUR IMAGE NAME HERE
# ==========================================
TARGET_IMAGE_NAME = "kodim02.png"   # <-- change if needed
K = 10                               # palette size
N_SEEDS = 30                        # tries to "hunt" a redundant pair (random restarts)
PIXEL_SAMPLE = 20000                # set None for all pixels (recommended >= 5000)
REDUNDANT_DE = 3.5                 # find palettes with min ΔE76 below this (JND threshold)
MAX_ITER = 10                       # k-means iterations

# ==========================================
# 1) SETUP & HELPER FUNCTIONS
# ==========================================

def load_image_flexible(target_name):
    image_path = None

    # Priority 1: exact file name
    if target_name and os.path.exists(target_name):
        image_path = target_name
    else:
        if target_name:
            print(f"Warning: '{target_name}' not found. Searching for other images...")
        patterns = ["*.tiff", "*.tif", "*.jpg", "*.jpeg", "*.png", "*.bmp"]
        found = []
        for p in patterns:
            found.extend(glob.glob(p))
        if found:
            image_path = found[0]

    if not image_path:
        raise FileNotFoundError("No image files found in directory.")

    print(f"Loading image: {image_path}")
    img = io.imread(image_path)

    # RGBA -> RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Grayscale -> RGB
    if img.ndim == 2:
        img = color.gray2rgb(img)

    # Ensure uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).round().astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    return img

def rgb_palette_to_lab(palette_rgb_uint8):
    # palette: (K,3) uint8 -> lab: (K,3)
    rgb01 = palette_rgb_uint8[np.newaxis, :, :].astype(np.float32) / 255.0
    return color.rgb2lab(rgb01)[0].astype(np.float32)

def calc_delta_e_lab(lab1, lab2):
    return sqrt((lab1[0]-lab2[0])**2 + (lab1[1]-lab2[1])**2 + (lab1[2]-lab2[2])**2)

def rgb_u8_to_lab_pixels(pixels_u8):
    rgb01 = pixels_u8.astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb01.reshape(-1, 1, 3)).reshape(-1, 3)
    return lab.astype(np.float32)

def lab_centers_to_rgb_u8(centers_lab):
    rgb = color.lab2rgb(centers_lab.reshape(-1, 1, 3)).reshape(-1, 3)
    return np.clip(rgb * 255.0, 0, 255).round().astype(np.uint8)

# ---------- K-MEANS++ INIT (LAB) ----------
def kmeans_plus_plus_init(X, k):
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float32)

    # first center
    idx = np.random.randint(n)
    centers[0] = X[idx]

    d2 = np.sum((X - centers[0])**2, axis=1)

    for c in range(1, k):
        probs = d2 / (np.sum(d2) + 1e-12)
        idx = np.random.choice(n, p=probs)
        centers[c] = X[idx]
        new_d2 = np.sum((X - centers[c])**2, axis=1)
        d2 = np.minimum(d2, new_d2)

    return centers

# ---------- LLOYD K-MEANS (LAB) ----------
def kmeans_lloyd_lab(X, k, max_iter=20):
    centers = kmeans_plus_plus_init(X, k)

    for _ in range(max_iter):
        d2 = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
        labels = np.argmin(d2, axis=1)

        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centers[j] = X[mask].mean(axis=0)

        if np.max(np.abs(new_centers - centers)) < 1e-4:
            centers = new_centers
            break

        centers = new_centers

    counts = np.bincount(labels, minlength=k).astype(np.int32)
    return centers.astype(np.float32), counts

# ==========================================
# 2) LOAD IMAGE + PREPARE PIXELS
# ==========================================
try:
    img = load_image_flexible(TARGET_IMAGE_NAME)
except Exception as e:
    print(f"Image load error: {e}. Using dummy image.")
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    img[10:30, 10:30] = (200, 200, 50)

pixels_u8 = img.reshape(-1, 3)

# Optional subsample for speed
if PIXEL_SAMPLE is not None and len(pixels_u8) > PIXEL_SAMPLE:
    idx = np.random.choice(len(pixels_u8), PIXEL_SAMPLE, replace=False)
    pixels_u8 = pixels_u8[idx]

# Work in LAB for clustering (more relevant to perceptual duplicates)
X_lab = rgb_u8_to_lab_pixels(pixels_u8)

# ==========================================
# 3) THE "HUNTER" LOOP (worst redundancy)
# ==========================================
print("Searching for a redundant palette (min ΔE76) ...")

worst_palette = None            # uint8 (K,3)
worst_conflict = None           # (i, j, deltaE, seed)
min_delta_e_found = 1e9

for seed in range(N_SEEDS):
    np.random.seed(seed)  # reproducible

    centers_lab, counts = kmeans_lloyd_lab(X_lab, K, max_iter=MAX_ITER)
    palette_u8 = lab_centers_to_rgb_u8(centers_lab)

    lab_palette = rgb_palette_to_lab(palette_u8)

    local_min = 1e9
    local_pair = None

    for i in range(K):
        for j in range(i + 1, K):
            de = calc_delta_e_lab(lab_palette[i], lab_palette[j])
            if de < local_min:
                local_min = de
                local_pair = (i, j, de, seed)

    if local_min < min_delta_e_found:
        min_delta_e_found = local_min
        worst_palette = palette_u8
        worst_conflict = local_pair

    if min_delta_e_found < REDUNDANT_DE:
        break

print(f"Found worst-case conflict: ΔE76 = {min_delta_e_found:.2f}")

if worst_conflict is None or worst_palette is None:
    raise RuntimeError("No palette/conflict found. Try increasing N_SEEDS, K, or PIXEL_SAMPLE.")

# ==========================================
# 4) CREATE THE VISUAL PROOF (1-based display labels)
# ==========================================
idx1, idx2, dist_val, seed = worst_conflict
color_a = worst_palette[idx1]
color_b = worst_palette[idx2]

fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 2])

# PLOT 1: The Generated Palette
ax1 = fig.add_subplot(gs[0])
ax1.imshow([worst_palette])
ax1.set_title(
    f"Generated Palette (K={K}) — K-means (seed={seed})\n"
    f"Standard Output (No perceptual separation constraint)",
    fontsize=12
)
ax1.set_yticks([])
ax1.set_xticks(np.arange(K))
ax1.set_xticklabels([f"Color {i+1}" for i in range(K)])  # ✅ 1-based labels

# Arrows pointing to redundant colors (positions remain 0-based)
ax1.annotate(
    "NEAR-DUPLICATE\n(The Problem)",
    xy=(idx1, 0.5),
    xytext=((idx1 + idx2) / 2, 2.4),
    arrowprops=dict(facecolor="red", arrowstyle="->", connectionstyle="arc3,rad=-0.25"),
    ha="center", color="red", fontsize=10, weight="bold"
)
ax1.annotate(
    "",
    xy=(idx2, 0.5),
    xytext=((idx1 + idx2) / 2, 2.4),
    arrowprops=dict(facecolor="red", arrowstyle="->", connectionstyle="arc3,rad=0.25"),
)

# PLOT 2: Explanation (✅ 1-based color numbers)
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
explanation = (
    "SOP 3 PROOF: Lack of perceptual separation metric\n\n"
    f"The algorithm selected Color {idx1+1} and Color {idx2+1} as separate clusters,\n"
    f"but their perceptual distance is only ΔE76 = {dist_val:.2f}.\n"
    "This means they can look visually indistinguishable, wasting palette slots\n"
    "in a limited K-color palette."
)
ax2.text(
    0.5, 0.5, explanation, ha="center", va="center", fontsize=11,
    bbox=dict(facecolor="#ffdddd", edgecolor="red", pad=15)
)

# PLOT 3: Side-by-side “eye test” (✅ 1-based color numbers)
ax3 = fig.add_subplot(gs[2])
comparison_img = np.zeros((110, 220, 3), dtype=np.uint8)
comparison_img[:, :110] = color_a
comparison_img[:, 110:] = color_b

ax3.imshow(comparison_img)
ax3.set_title("EYE TEST: Can you distinguish these two?", fontweight="bold", fontsize=12)
rgb_a = tuple(int(x) for x in color_a)
rgb_b = tuple(int(x) for x in color_b)
ax3.set_xticks([55, 165])
ax3.set_xticklabels([f"Color {idx1+1}\nRGB: {rgb_a}", f"Color {idx2+1}\nRGB: {rgb_b}"], fontsize=10)
ax3.set_yticks([])
ax3.axvline(x=109.5, color="white", linewidth=5)

plt.tight_layout()
plt.show()