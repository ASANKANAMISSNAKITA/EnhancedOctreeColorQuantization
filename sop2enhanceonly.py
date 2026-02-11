# ==========================================================
# SOP2 (ENH ONLY) – Resize to <=512 then run memory-safe K-Means
# (NO QUANTIZED VISUALS) – Show runtime, MSE, K + pixel count
# 1x2 Figure: [ Resized for Processing ] [ ENH Palette Swatch ]
# ==========================================================

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

IMAGE_PATH = "birdhouse.jpg"
K = 8
MAX_ITER = 10
PROCESS_MAX = 512

# ✅ enhancement: chunked assignment (prevents NxK memory blow-up)
CHUNK_SIZE = 120000  # adjust (50k–300k) depending on RAM


def resize_to_max(img, max_side=512):
    w, h = img.size
    if max(w, h) <= max_side:
        return img, False
    scale = max_side / float(max(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    return img.resize((new_w, new_h), resample), True


def init_centroids_deterministic(pixels_f32, K):
    N = len(pixels_f32)
    K = min(K, N)
    idx = np.linspace(0, N - 1, K, dtype=int)
    return pixels_f32[idx].copy()


def kmeans_rgb_numpy_chunked(pixels_u8, K, max_iter=10, early_stop=True, tag="ENH", chunk_size=120000):
    pixels = pixels_u8.astype(np.float32)  # (N,3)
    N = pixels.shape[0]
    K = min(K, N)

    centroids = init_centroids_deterministic(pixels, K)
    mse_hist = []
    prev_mse = None

    for it in range(max_iter):
        counts = np.zeros(K, dtype=np.float32)
        sums = np.zeros((K, 3), dtype=np.float32)
        mse_sum = 0.0

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = pixels[start:end]

            diff = chunk[:, None, :] - centroids[None, :, :]
            d2 = np.sum(diff * diff, axis=2)

            labels = np.argmin(d2, axis=1)
            min_d2 = d2[np.arange(len(chunk)), labels]
            mse_sum += float(np.sum(min_d2))

            counts += np.bincount(labels, minlength=K).astype(np.float32)
            for ch in range(3):
                sums[:, ch] += np.bincount(labels, weights=chunk[:, ch], minlength=K).astype(np.float32)

        mse = mse_sum / float(N)
        mse_hist.append(mse)
        print(f"[{tag}] Iter {it}: MSE_RGB = {mse:.2f}")

        if early_stop and prev_mse is not None and mse >= prev_mse:
            print(f"[{tag}] Early stop: MSE did not improve.")
            break
        prev_mse = mse

        mask = counts > 0
        centroids[mask] = sums[mask] / counts[mask, None]

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
    orig_pixels = ow * oh

    # ENH: resize to <=512 then run KMeans
    proc_img, resized = resize_to_max(orig, PROCESS_MAX)
    pw, ph = proc_img.size
    proc_pixels = pw * ph

    proc_np = np.asarray(proc_img, dtype=np.uint8)

    print(f"\nUPLOAD: {ow}x{oh}  (pixels={orig_pixels:,})")
    print(f"ENH processing cap: <= {PROCESS_MAX}px")
    print(f"ENH resized to: {pw}x{ph}  (pixels={proc_pixels:,})  | reduction ~{orig_pixels/proc_pixels:.1f}×")

    t0 = time.perf_counter()
    pal_enh, mse_enh = kmeans_rgb_numpy_chunked(
        proc_np.reshape(-1, 3), K, MAX_ITER, early_stop=True, tag="ENH", chunk_size=CHUNK_SIZE
    )
    t1 = time.perf_counter()
    runtime_enh = t1 - t0
    final_mse_enh = mse_enh[-1]

    sw_enh = make_swatch_image(pal_enh)

    # 1x2 (no quantized output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SOP2 Enhancement (Resize ≤512)", fontsize=15)

    fig.text(
        0.5, 0.90,
        f"ENH: {runtime_enh:.2f}s | MSE: {final_mse_enh:.2f} | Pixels: {proc_pixels:,} | K={K}",
        ha="center", va="center",
        fontsize=13, fontweight="bold"
    )

    axes[0].imshow(proc_img)
    axes[0].set_title(f"Resized for Processing ({pw}×{ph}px | {proc_pixels:,} px)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(np.asarray(sw_enh, dtype=np.uint8))
    axes[1].set_title("ENH Final palette swatch", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()
