# ============================================================
# SOP 2 ENHANCED Visualization (Patch-based, robust)
# Dominant vs Rare Color Bias in Palette Generation
#
# 2x2:
# [ Original (with rare patch) ] [ Generated Palette ]
# [ RGB Cube (pixels)          ] [ Evidence + Explanation Text  ]
# ============================================================

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------- PARAMETERS ----------------
IMAGE_PATH = "starry.jpg"      # <-- put your image here
K = 8
MAX_ITER = 10
SEED = 1
SAMPLE_POINTS = 1500           # keep light so it won't lag

# Rare accent patch (small but visually important)
PATCH_SIZE = 0                 # if 0, auto-set to ~2% of min(image_w, image_h)
PATCH_COLOR = (0, 80, 255)     # strong blue (visible)
PATCH_POS = "bottom_right"     # "top_left", "top_right", "bottom_left", "bottom_right"

# Evidence settings
NEAR_THRESH = 25               # "near" distance in RGB (15–40 works)

# ---------------- K-MEANS (RGB) ----------------
def squared_euclidean(c1, c2):
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return dr*dr + dg*dg + db*db

def kmeans_rgb(pixels, K, max_iter=10, seed=None):
    if seed is not None:
        random.seed(seed)

    pixels = [tuple(map(int, p)) for p in pixels]
    N = len(pixels)
    K = min(K, N)

    centroids = random.sample(pixels, K)

    for _ in range(max_iter):
        clusters = [[] for _ in range(K)]

        for p in pixels:
            idx = min(range(K), key=lambda i: squared_euclidean(p, centroids[i]))
            clusters[idx].append(p)

        for i in range(K):
            if clusters[i]:
                arr = np.array(clusters[i], dtype=np.float32)
                centroids[i] = tuple(np.mean(arr, axis=0).astype(np.int32))

    return centroids

# ---------------- PALETTE SWATCH ----------------
def make_swatch(palette, h=60, w=50):
    img = Image.new("RGB", (w * len(palette), h))
    for i, c in enumerate(palette):
        c = tuple(map(int, c))
        for x in range(i * w, (i + 1) * w):
            for y in range(h):
                img.putpixel((x, y), c)
    return img

# ---------------- RGB CUBE ----------------
def plot_rgb_cube(ax, pixels, title):
    pts = np.array(pixels, dtype=np.uint8)
    if len(pts) > SAMPLE_POINTS:
        idx = np.random.choice(len(pts), SAMPLE_POINTS, replace=False)
        pts = pts[idx]

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts/255.0, s=3, alpha=0.6)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_title(title)

# ---------------- ADD RARE PATCH ----------------
def add_rare_patch(img, patch_size, patch_color, pos="bottom_right"):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size

    # auto patch size if user left it as 0
    if patch_size is None or patch_size <= 0:
        patch_size = max(6, int(0.02 * min(w, h)))  # ~2% of min dimension, at least 6px

    if pos == "top_left":
        x0, y0 = 10, 10
    elif pos == "top_right":
        x0, y0 = w - patch_size - 10, 10
    elif pos == "bottom_left":
        x0, y0 = 10, h - patch_size - 10
    else:  # bottom_right
        x0, y0 = w - patch_size - 10, h - patch_size - 10

    # filled patch
    draw.rectangle([x0, y0, x0 + patch_size, y0 + patch_size], fill=patch_color)
    # border (high-contrast)
    draw.rectangle([x0, y0, x0 + patch_size, y0 + patch_size], outline=(255, 255, 255), width=2)

    return out, patch_size, (x0, y0, x0 + patch_size, y0 + patch_size)

# ---------------- EVIDENCE CHECKS ----------------
def percent_pixels_near_color(pixels_uint8, target_rgb, near_thresh):
    px = pixels_uint8.astype(np.int32)
    t = np.array(target_rgb, dtype=np.int32)
    d2 = np.sum((px - t) ** 2, axis=1)
    near = int(np.sum(d2 <= near_thresh * near_thresh))
    total = int(px.shape[0])
    return near, total, (near / total) * 100.0

def palette_has_near_color(palette, target_rgb, near_thresh):
    t = np.array(target_rgb, dtype=np.int32)
    best = None
    for c in palette:
        c = np.array(c, dtype=np.int32)
        d = np.sqrt(np.sum((c - t) ** 2))
        if best is None or d < best:
            best = d
    return (best is not None and best <= near_thresh), best

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Load image safely
    try:
        orig = Image.open(IMAGE_PATH).convert("RGB")
    except FileNotFoundError:
        raise SystemExit(f"ERROR: Cannot find file '{IMAGE_PATH}'. Put the image in the same folder or fix IMAGE_PATH.")

    # Add rare patch
    test_img, final_patch_size, patch_box = add_rare_patch(orig, PATCH_SIZE, PATCH_COLOR, PATCH_POS)

    img_arr = np.array(test_img, dtype=np.uint8)
    pixels = img_arr.reshape(-1, 3)
    pixels_list = [tuple(p) for p in pixels]

    # Generate palette
    palette = kmeans_rgb(pixels_list, K, MAX_ITER, SEED)
    swatch = make_swatch(palette)

    # Evidence: how rare is the patch color?
    near_count, total, near_pct = percent_pixels_near_color(pixels, PATCH_COLOR, NEAR_THRESH)

    # Evidence: did palette capture it (or something near it)?
    has_patch, best_d = palette_has_near_color(palette, PATCH_COLOR, NEAR_THRESH)

    # -------- 2x2 FIGURE --------
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # (1) Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_arr)
    ax1.set_title(f"Input Image (rare patch: {PATCH_COLOR}, size={final_patch_size}px)")
    ax1.axis("off")

    # (2) Palette
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.array(swatch))
    ax2.set_title(f"Generated Palette (RGB K-Means, K={K})")
    ax2.axis("off")

    # (3) RGB cube
    ax3 = fig.add_subplot(gs[1, 0], projection="3d")
    plot_rgb_cube(ax3, pixels_list, "RGB Cube (Sampled Pixels)")

    # (4) Explanation + evidence
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.text(
        0.05, 0.75,
        "SOP 2 Observation (Dominant vs Rare Color Bias)\n\n"
        f"• Patch color: {PATCH_COLOR}\n"
        f"• Pixels near patch color: {near_count} / {total} (~{near_pct:.3f}%)\n"
        f"• Palette captured patch color (±{NEAR_THRESH})? {has_patch}\n"
        f"• Closest palette distance to patch: {best_d:.2f}\n\n"
        "Why it happens:\n"
        "• K-Means minimizes overall error.\n"
        "• Dominant colors (many pixels) pull centroids.\n"
        "• Small regions can be ignored if they don’t reduce the global error much.",
        fontsize=10
    )

    plt.suptitle("SOP 2 Visual: Rare-but-visible colors can be missed in palette generation", fontsize=14)
    plt.show()
