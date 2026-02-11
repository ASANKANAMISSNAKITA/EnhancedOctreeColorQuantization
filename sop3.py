import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from skimage import io, color
from math import sqrt
import glob
import os

# ==========================================
# CONFIGURATION: TYPE YOUR IMAGE NAME HERE
# ==========================================
TARGET_IMAGE_NAME = "starry.jpg"   # <-- change if needed
K = 8                              # palette size
N_SEEDS = 30                       # tries to "hunt" a redundant pair
PIXEL_SAMPLE = 20000               # set None for all pixels (recommended >= 5000)
REDUNDANT_DE = 5.0                 # "near-duplicate" threshold (ΔE76)

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
    rgb01 = palette_rgb_uint8[np.newaxis, :, :] / 255.0
    return color.rgb2lab(rgb01)[0]

def calc_delta_e_lab(lab1, lab2):
    return sqrt((lab1[0]-lab2[0])**2 + (lab1[1]-lab2[1])**2 + (lab1[2]-lab2[2])**2)

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

# MiniBatchKMeans works better on float [0,1]
pixels = pixels_u8.astype(np.float32) / 255.0

# ==========================================
# 3) THE "HUNTER" LOOP (worst redundancy)
# ==========================================
print("Searching for a redundant palette to demonstrate the SOP (perceptual duplicates)...")

worst_palette = None            # uint8 (K,3)
worst_conflict = None           # (i, j, deltaE, seed)
min_delta_e_found = 1e9

batch_size = min(2048, len(pixels))

for seed in range(N_SEEDS):
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        random_state=seed,
        init="k-means++",
        n_init=10,
        batch_size=batch_size,
        max_iter=200,
        max_no_improvement=20,
        reassignment_ratio=0.01
    )
    kmeans.fit(pixels)

    centers01 = np.clip(kmeans.cluster_centers_, 0.0, 1.0)
    palette_u8 = (centers01 * 255.0).round().astype(np.uint8)

    # compute pairwise min ΔE in LAB (fast: convert once)
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

print(f"Found worst-case conflict: Delta E76 = {min_delta_e_found:.2f}")

# Safety fallback
if worst_conflict is None or worst_palette is None:
    raise RuntimeError("No palette/conflict found. Try increasing N_SEEDS or K, or PIXEL_SAMPLE.")

# ==========================================
# 4) CREATE THE VISUAL PROOF
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
    f"Generated Palette (K={K}) — MiniBatchKMeans (seed={seed})\n"
    f"Standard Output (No perceptual separation constraint)",
    fontsize=12
)
ax1.set_yticks([])
ax1.set_xticks(np.arange(K))
ax1.set_xticklabels([f"Color {i}" for i in range(K)])

# Arrows pointing to redundant colors
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

# PLOT 2: Explanation
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
explanation = (
    "SOP 3 PROOF: Lack of perceptual separation metric\n\n"
    f"The algorithm selected Color {idx1} and Color {idx2} as separate clusters,\n"
    f"but their perceptual distance is only ΔE76 = {dist_val:.2f}.\n"
    "This means they can look visually indistinguishable, wasting palette slots\n"
    "in a limited K-color palette."
)
ax2.text(
    0.5, 0.5, explanation, ha="center", va="center", fontsize=11,
    bbox=dict(facecolor="#ffdddd", edgecolor="red", pad=15)
)

# PLOT 3: Side-by-side “eye test”
ax3 = fig.add_subplot(gs[2])
comparison_img = np.zeros((110, 220, 3), dtype=np.uint8)
comparison_img[:, :110] = color_a
comparison_img[:, 110:] = color_b

ax3.imshow(comparison_img)
ax3.set_title("EYE TEST: Can you distinguish these two?", fontweight="bold", fontsize=12)
ax3.set_xticks([55, 165])
ax3.set_xticklabels([f"Color {idx1}\nRGB: {tuple(color_a)}", f"Color {idx2}\nRGB: {tuple(color_b)}"])
ax3.set_yticks([])
ax3.axvline(x=109.5, color="white", linewidth=5)

plt.tight_layout()
plt.show()
