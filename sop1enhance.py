# ============================================================
# SOP1 Enhanced (2x2) – CIELAB Fast K-Means Visualization
# [ Original Image ] [ Quantized (Lab K-Means) ]
# [ CIELAB Sphere  ] [ Lab Palette Swatch     ]
# ============================================================

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from skimage.color import rgb2lab, lab2rgb
import random

# ---------------- PARAMETERS ----------------
IMAGE_PATH = "flower.jpg"
K = 8
MAX_ITER = 10
SEED = 1
SAMPLE_POINTS = 5000

# ---------------- HELPERS ----------------

def make_swatch_image(palette, swatch_height=70, swatch_width_per_color=50):
    w = swatch_width_per_color * len(palette)
    h = swatch_height
    img = Image.new("RGB", (w, h))
    for i, c in enumerate(palette):
        for x in range(i * swatch_width_per_color, (i + 1) * swatch_width_per_color):
            for y in range(h):
                img.putpixel((x, y), c)
    return img

def rgb_list_to_lab_array(pixels):
    arr = np.array(pixels, dtype=np.float32) / 255.0
    return rgb2lab(arr.reshape(-1, 1, 3)).reshape(-1, 3)

# ---------------- LAB K-MEANS ----------------

def kmeans_lab(pixels_rgb, K, max_iter=10, seed=None):
    rng = np.random.RandomState(seed)

    pixels_lab = rgb_list_to_lab_array(pixels_rgb)
    N = len(pixels_lab)
    K = min(K, N)

    centroids = pixels_lab[rng.choice(N, K, replace=False)].copy()
    mse_history = []

    for it in range(max_iter):
        clusters = [[] for _ in range(K)]
        mse = 0.0

        for p in pixels_lab:
            dists = np.sum((centroids - p) ** 2, axis=1)
            idx = int(np.argmin(dists))
            clusters[idx].append(p)
            mse += dists[idx]

        mse /= N
        mse_history.append(mse)
        print(f"[Lab] Iter {it}: MSE_Lab = {mse:.2f}")

        for k in range(K):
            if clusters[k]:
                centroids[k] = np.mean(clusters[k], axis=0)

    rgb_centroids = lab2rgb(centroids.reshape(-1, 1, 3)).reshape(-1, 3)
    rgb_centroids = np.clip(rgb_centroids * 255, 0, 255).astype(np.uint8)
    palette_rgb = [tuple(c) for c in rgb_centroids]

    return palette_rgb, centroids, mse_history

# ---------------- LAB QUANTIZATION ----------------

def quantize_image_lab(img, centroids_lab, palette_rgb):
    arr_rgb = np.array(img, dtype=np.float32) / 255.0
    lab_img = rgb2lab(arr_rgb)

    out = np.zeros_like(arr_rgb)
    for y in range(arr_rgb.shape[0]):
        for x in range(arr_rgb.shape[1]):
            dists = np.sum((centroids_lab - lab_img[y, x]) ** 2, axis=1)
            out[y, x] = np.array(palette_rgb[np.argmin(dists)]) / 255.0

    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))

# ---------------- LAB SPHERE PLOT ----------------

def plot_lab_sphere(ax, pixels_rgb, title):
    pts = np.array(pixels_rgb)
    if len(pts) > SAMPLE_POINTS:
        pts = pts[np.random.choice(len(pts), SAMPLE_POINTS, replace=False)]

    lab = rgb2lab((pts / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    ax.set_xlim(-128, 128)
    ax.set_ylim(-128, 128)
    ax.set_zlim(0, 100)

    ax.set_xlabel("a* (green ↔ red)")
    ax.set_ylabel("b* (blue ↔ yellow)")
    ax.set_zlabel("L* (lightness)")
    ax.set_title(title)

    # Chroma circle at L* = 50
    theta = np.linspace(0, 2 * np.pi, 200)
    r = 80
    ax.plot(r*np.cos(theta), r*np.sin(theta), np.full_like(theta, 50),
            linestyle="--", linewidth=1)

    # Vertical L* axis
    ax.plot([0,0],[0,0],[0,100], color="black", linewidth=2)

    # Direction arrows
    ax.quiver(0,0,50,  90,0,0, arrow_length_ratio=0.08)
    ax.quiver(0,0,50, -90,0,0, arrow_length_ratio=0.08)
    ax.quiver(0,0,50,  0,90,0, arrow_length_ratio=0.08)
    ax.quiver(0,0,50,  0,-90,0, arrow_length_ratio=0.08)

    ax.scatter(a, b, L, c=pts/255.0, s=2, alpha=0.6)

# ---------------- MAIN ----------------

if __name__ == "__main__":
    orig = Image.open(IMAGE_PATH).convert("RGB")
    pixels = list(orig.getdata())

    palette_rgb, centroids_lab, mse_hist = kmeans_lab(
        pixels, K, MAX_ITER, SEED
    )
    final_mse = mse_hist[-1]

    quant = quantize_image_lab(orig, centroids_lab, palette_rgb)
    swatch = make_swatch_image(palette_rgb)

    # -------- 2x2 FIGURE --------
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.35, wspace=0.25)

    fig.suptitle(
        f"CIELAB Fast K-Means Visualization (Final MSE_Lab = {final_mse:.2f})",
        fontsize=13
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig)
    ax1.set_title("Original image")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(quant)
    ax2.set_title(f"Quantized (Lab K-Means, K={K})")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0], projection="3d")
    plot_lab_sphere(ax3, pixels, 'CIELAB Perceptual Space')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(np.array(swatch))
    ax4.set_title("Lab-based palette swatch")
    ax4.axis("off")

    plt.show()
