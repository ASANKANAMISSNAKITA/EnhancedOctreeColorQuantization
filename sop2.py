# ============================================================
# SOP 2 Visualization (Using your own image)
# Dominant vs Rare Color Bias in Palette Generation
#
# 2x2:
# [ Original (with rare patch) ] [ Generated Palette ]
# [ RGB Cube (pixels)          ] [ Explanation Text  ]
# ============================================================

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------- PARAMETERS ----------------
IMAGE_PATH = "starry.jpg"   # <-- put your image here
K = 8
MAX_ITER = 10
SEED = 1
SAMPLE_POINTS = 5000

# Rare accent patch (small but visually important)
PATCH_SIZE = 0                         # <-- make smaller to be "rare"
PATCH_COLOR = (0, 0, 0)             # <-- strong color
PATCH_POS = "bottom_right"              # "top_left", "top_right", "bottom_left", "bottom_right"

# ---------------- K-MEANS (RGB) ----------------
def squared_euclidean(c1, c2):
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return dr*dr + dg*dg + db*db

def kmeans_rgb(pixels, K, max_iter=10, seed=None):
    if seed is not None:
        random.seed(seed)

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
                centroids[i] = tuple(
                    int(sum(c[j] for c in clusters[i]) / len(clusters[i]))
                    for j in range(3)
                )

    return centroids

# ---------------- PALETTE SWATCH ----------------
def make_swatch(palette, h=60, w=50):
    img = Image.new("RGB", (w * len(palette), h))
    for i, c in enumerate(palette):
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

    if pos == "top_left":
        x0, y0 = 10, 10
    elif pos == "top_right":
        x0, y0 = w - patch_size - 10, 10
    elif pos == "bottom_left":
        x0, y0 = 10, h - patch_size - 10
    else:  # bottom_right
        x0, y0 = w - patch_size - 10, h - patch_size - 10

    draw.rectangle([x0, y0, x0 + patch_size, y0 + patch_size], fill=patch_color)

    # optional border to make it obvious
    draw.rectangle([x0, y0, x0 + patch_size, y0 + patch_size], outline=(0, 0, 0), width=2)

    return out

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Load your image
    orig = Image.open(IMAGE_PATH).convert("RGB")

    # Add a small rare-color patch so we can test if palette captures it
    test_img = add_rare_patch(orig, PATCH_SIZE, PATCH_COLOR, PATCH_POS)

    pixels = list(test_img.getdata())

    # Generate palette from the test image
    palette = kmeans_rgb(pixels, K, MAX_ITER, SEED)
    swatch = make_swatch(palette)

    # -------- 2x2 FIGURE --------
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.array(test_img))
    ax1.set_title("Input Image (with rare accent patch)")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.array(swatch))
    ax2.set_title("Generated Palette (RGB K-Means)")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0], projection="3d")
    plot_rgb_cube(ax3, pixels, "RGB Cube (Sampled Pixels)")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.text(
        0.05, 0.6,
        "SOP 2 Observation:\n\n"
        "• Dominant colors occupy most pixels\n"
        "• The accent patch is visually important but rare\n"
        "• K-Means centroids are pulled toward dominant regions\n"
        "• Rare colors may be missing from the final palette",
        fontsize=11
    )

    plt.suptitle("SOP 2 Visual: Dominant vs Rare Color Bias in Palette Generation", fontsize=14)
    plt.show()
