# ==========================================================
# SOP2 Baseline (No graph) – Original + Palette + BIG Runtime (+ Pixel Count)
# ==========================================================

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

IMAGE_PATH = "birdhouse.jpg"
K = 8
MAX_ITER = 10


def squared_euclidean_batch(pixels, centroids):
    diff = pixels[:, None, :] - centroids[None, :, :]
    return np.sum(diff * diff, axis=2)

def init_centroids_deterministic(pixels, K):
    N = len(pixels)
    K = min(K, N)
    idx = np.linspace(0, N - 1, K, dtype=int)
    return pixels[idx].copy()

def kmeans_rgb_numpy(pixels_u8, K, max_iter=10, early_stop=True):
    pixels = pixels_u8.astype(np.float32)
    N = pixels.shape[0]

    centroids = init_centroids_deterministic(pixels, K)
    mse_hist = []
    prev_mse = None

    for it in range(max_iter):
        d2 = squared_euclidean_batch(pixels, centroids)
        labels = np.argmin(d2, axis=1)
        min_d2 = d2[np.arange(N), labels]
        mse = float(np.mean(min_d2))
        mse_hist.append(mse)

        print(f"[BASE] Iter {it}: MSE_RGB = {mse:.2f}")

        if early_stop and prev_mse is not None and mse >= prev_mse:
            print("[BASE] Early stop: MSE did not improve.")
            break
        prev_mse = mse

        counts = np.bincount(labels, minlength=K).astype(np.float32)
        new_centroids = centroids.copy()
        for ch in range(3):
            sums = np.bincount(labels, weights=pixels[:, ch], minlength=K).astype(np.float32)
            mask = counts > 0
            new_centroids[mask, ch] = sums[mask] / counts[mask]
        centroids = new_centroids

    return np.clip(centroids, 0, 255).astype(np.uint8), mse_hist

def make_swatch_image(palette, swatch_h=90, w_per=70):
    img = Image.new("RGB", (w_per * len(palette), swatch_h))
    for i, c in enumerate([tuple(map(int, x)) for x in palette]):
        for x in range(i * w_per, (i + 1) * w_per):
            for y in range(swatch_h):
                img.putpixel((x, y), c)
    return img


if __name__ == "__main__":
    orig = Image.open(IMAGE_PATH).convert("RGB")
    ow, oh = orig.size
    orig_np = np.asarray(orig, dtype=np.uint8)
    total_pixels = ow * oh

    print(f"Upload size: {ow}x{oh}  (pixels={total_pixels:,})")

    t0 = time.perf_counter()
    palette, mse_hist = kmeans_rgb_numpy(orig_np.reshape(-1, 3), K, MAX_ITER, early_stop=True)
    t1 = time.perf_counter()

    runtime = t1 - t0
    final_mse = mse_hist[-1]

    print("\nFinal palette (RGB centroids):")
    for i, c in enumerate(palette):
        print(f"  c{i}: ({int(c[0])}, {int(c[1])}, {int(c[2])})")

    print(f"\nRuntime (baseline on full upload): {runtime:.2f} seconds")
    print(f"Final MSE_RGB: {final_mse:.2f}")

    swatch = make_swatch_image(palette)

    # -----------------------------
    # 1x2 Figure (NO graph)
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("SOP2 Baseline (Full-res)", fontsize=16)

    # ✅ BIG runtime showcase (now includes pixel count)
    fig.text(
        0.5, 0.91,
        f"Runtime: {runtime:.2f}s   |   Final MSE: {final_mse:.2f}   |   Pixels: {total_pixels:,}   |   K={K}",
        ha="center", va="center",
        fontsize=18, fontweight="bold"
    )

    # Original image (now includes pixel count in title too)
    axes[0].imshow(orig)
    axes[0].set_title(f"Original Upload ({ow}×{oh}px | {total_pixels:,} px)", fontsize=13)
    axes[0].axis("off")

    # Palette swatch
    axes[1].imshow(np.asarray(swatch, dtype=np.uint8))
    axes[1].set_title("Final palette swatch", fontsize=13)
    axes[1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()
