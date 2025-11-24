# ==============================================
# Visualization 2 – Real image + RGB palette swatch
# Original  |  Quantized with RGB K-Means  |  Palette
# ==============================================

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# ------------- PARAMETERS ----------------

IMAGE_PATH = "flower.jpg"   # <-- change to your test image
K = 8                       # number of palette colors (7 or 8)
MAX_ITER = 10               # K-Means iterations
SEED = 1                    # for reproducibility

# ------------- HELPER FUNCTIONS ----------------

def squared_euclidean(c1, c2):
    """Squared Euclidean distance in RGB."""
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return dr*dr + dg*dg + db*db

def kmeans_rgb(pixels, K, max_iter=10, seed=None):
    """
    Plain K-Means in RGB using squared Euclidean distance.
    Returns: palette (list of (R,G,B)), mse_history (per iteration).
    """
    if seed is not None:
        random.seed(seed)

    N = len(pixels)
    if K > N:
        K = N

    # random initialization from existing pixels
    centroids = [list(c) for c in random.sample(pixels, K)]
    mse_history = []

    for it in range(max_iter):
        clusters = [[] for _ in range(K)]
        mse_accum = 0.0

        # --- assignment step ---
        for (r, g, b) in pixels:
            dists = [squared_euclidean((r, g, b), (c[0], c[1], c[2]))
                     for c in centroids]
            k_idx = min(range(K), key=lambda i: dists[i])
            clusters[k_idx].append((r, g, b))
            mse_accum += dists[k_idx]

        mse = mse_accum / N
        mse_history.append(mse)
        print(f"Iter {it}: MSE = {mse:.2f}")

        # --- update step ---
        for k in range(K):
            if clusters[k]:
                sr = sum(p[0] for p in clusters[k]) / len(clusters[k])
                sg = sum(p[1] for p in clusters[k]) / len(clusters[k])
                sb = sum(p[2] for p in clusters[k]) / len(clusters[k])
                centroids[k] = [int(sr), int(sg), int(sb)]
            # if empty cluster: keep old centroid

    palette = [tuple(c) for c in centroids]
    return palette, mse_history

def quantize_image(img, palette):
    """
    Quantize an image: each pixel -> nearest palette color (RGB Euclidean).
    """
    width, height = img.size
    out = Image.new("RGB", (width, height))
    pal = list(palette)

    for y in range(height):
        for x in range(width):
            c = img.getpixel((x, y))
            dists = [squared_euclidean(c, p) for p in pal]
            k_idx = min(range(len(pal)), key=lambda i: dists[i])
            out.putpixel((x, y), pal[k_idx])

    return out

def make_swatch_image(palette, swatch_height=80, swatch_width_per_color=40):
    """
    Build a horizontal palette swatch image from a list of (R,G,B).
    """
    if not palette:
        return None

    w = swatch_width_per_color * len(palette)
    h = swatch_height
    swatch = Image.new("RGB", (w, h))

    for i, color in enumerate(palette):
        for x in range(i * swatch_width_per_color,
                       (i + 1) * swatch_width_per_color):
            for y in range(h):
                swatch.putpixel((x, y), color)

    return swatch

# ------------- MAIN EXPERIMENT ----------------

if __name__ == "__main__":
    # 1) Load image and convert to RGB
    orig_img = Image.open(IMAGE_PATH).convert("RGB")
    width, height = orig_img.size

    # 2) Flatten pixels for K-Means
    pixels = [orig_img.getpixel((x, y))
              for y in range(height)
              for x in range(width)]
    print(f"Total pixels: {len(pixels)}")

    # 3) Run K-Means in RGB
    print(f"\nRunning K-Means in RGB with K={K} ...")
    palette, mse_history = kmeans_rgb(pixels, K=K,
                                      max_iter=MAX_ITER,
                                      seed=SEED)
    final_mse = mse_history[-1]
    print(f"\nFinal MSE (RGB): {final_mse:.2f}")

    print("\nFinal palette (RGB):")
    for i, c in enumerate(palette):
        print(f"  c{i}: {c}")

    # 4) Quantize the image using this palette
    quant_img = quantize_image(orig_img, palette)

    # 5) Build palette swatch image
    swatch_img = make_swatch_image(palette)

    # 6) Plot: Original | Quantized | Palette
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original
    axes[0].imshow(np.array(orig_img))
    axes[0].set_title("Original image")
    axes[0].axis("off")

    # Quantized
    axes[1].imshow(np.array(quant_img))
    axes[1].set_title(f"Quantized (RGB K-Means, K={K})")
    axes[1].axis("off")

    # Palette swatch
    if swatch_img is not None:
        axes[2].imshow(np.array(swatch_img))
    axes[2].set_title("RGB palette swatch")
    axes[2].axis("off")

    plt.suptitle(f"Real-image visualization of RGB Fast K-Means (Final MSE={final_mse:.2f})",
                 fontsize=11)
    plt.tight_layout()
    plt.show()
