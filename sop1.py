# ==============================================
# SOP1 (Not Enhanced) – RGB Fast K-Means (2x2)
# [ Original ] [ Quantized ]
# [ RGB Cube  ] [ Swatch   ]
# + prints iteration MSE + final palette
# ==============================================

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

IMAGE_PATH = "starry.jpg"
K = 8
MAX_ITER = 10
SEED = 1
SAMPLE_POINTS = 5000

def squared_euclidean(c1, c2):
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return dr*dr + dg*dg + db*db

def kmeans_rgb(pixels, K, max_iter=10, seed=None, early_stop=True):
    if seed is not None:
        random.seed(seed)

    N = len(pixels)
    K = min(K, N)

    centroids = [tuple(c) for c in random.sample(pixels, K)]
    mse_history = []
    prev_mse = None

    for it in range(max_iter):
        clusters = [[] for _ in range(K)]
        mse_accum = 0.0

        # --- assignment step ---
        for p in pixels:
            dists = [squared_euclidean(p, c) for c in centroids]
            idx = int(np.argmin(dists))
            clusters[idx].append(p)
            mse_accum += dists[idx]

        mse = mse_accum / N
        mse_history.append(mse)

        # ✅ PRINT ITERATION COMPUTATION
        print(f"[RGB] Iter {it}: MSE_RGB = {mse:.2f}")

        # optional early stop (remove if you don't want it)
        if early_stop and prev_mse is not None and mse >= prev_mse:
            print("[RGB] Early stop: MSE did not improve.")
            break
        prev_mse = mse

        # --- update step ---
        new_centroids = []
        for i in range(K):
            if clusters[i]:
                sr = sum(px[0] for px in clusters[i]) / len(clusters[i])
                sg = sum(px[1] for px in clusters[i]) / len(clusters[i])
                sb = sum(px[2] for px in clusters[i]) / len(clusters[i])
                new_centroids.append((int(sr), int(sg), int(sb)))
            else:
                new_centroids.append(centroids[i])  # keep if empty cluster
        centroids = new_centroids

    return centroids, mse_history

def quantize_image(img, palette):
    w, h = img.size
    out = Image.new("RGB", (w, h))
    src = img.load()
    dst = out.load()

    for y in range(h):
        for x in range(w):
            p = src[x, y]
            dists = [squared_euclidean(p, c) for c in palette]
            dst[x, y] = palette[int(np.argmin(dists))]

    return out

def make_swatch_image(palette, swatch_h=70, w_per=50):
    img = Image.new("RGB", (w_per * len(palette), swatch_h))
    for i, c in enumerate(palette):
        for x in range(i * w_per, (i + 1) * w_per):
            for y in range(swatch_h):
                img.putpixel((x, y), c)
    return img

def plot_rgb_cube(ax, pixels, title):
    pts = np.array(pixels, dtype=np.uint8)
    if len(pts) > SAMPLE_POINTS:
        idx = np.random.choice(len(pts), SAMPLE_POINTS, replace=False)
        pts = pts[idx]

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts/255.0, s=2, alpha=0.6)
    ax.set_xlim(0, 255); ax.set_ylim(0, 255); ax.set_zlim(0, 255)
    ax.set_xlabel("Red"); ax.set_ylabel("Green"); ax.set_zlabel("Blue")
    ax.set_title(title)

if __name__ == "__main__":
    orig = Image.open(IMAGE_PATH).convert("RGB")
    pixels = list(orig.getdata())
    print(f"Total pixels: {len(pixels)}")

    palette, mse_hist = kmeans_rgb(pixels, K, MAX_ITER, SEED, early_stop=True)
    final_mse = mse_hist[-1]

    print("\nFinal palette (RGB centroids):")
    for i, c in enumerate(palette):
        print(f"  c{i}: {c}")

    quant = quantize_image(orig, palette)
    swatch = make_swatch_image(palette)

    # -------- 2x2 Figure --------
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], wspace=0.25, hspace=0.35)

    fig.suptitle(
        f"RGB Fast K-Means Visualization (Final MSE={final_mse:.2f})",
        fontsize=13
    )

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(orig)
    ax_orig.set_title("Original image")
    ax_orig.axis("off")

    ax_quant = fig.add_subplot(gs[0, 1])
    ax_quant.imshow(quant)
    ax_quant.set_title(f"Quantized (RGB K-Means, K={K})")
    ax_quant.axis("off")

    ax_cube = fig.add_subplot(gs[1, 0], projection="3d")
    plot_rgb_cube(ax_cube, pixels, "RGB Cube (Sampled pixels)")

    ax_swatch = fig.add_subplot(gs[1, 1])
    ax_swatch.imshow(np.array(swatch))
    ax_swatch.set_title("RGB palette swatch")
    ax_swatch.axis("off")

    plt.show()
