# ==========================================================
# SOP2 (Proof + Enhancement) – High-res is slower, resizing fixes it
# (NO QUANTIZED VISUALS) – Focus on Runtime, MSE, K + PIXEL COUNT
#
# 1x3 Figure:
# [ Original Upload ] [ Resized for Processing ] [ ENH Palette Swatch ]
# ==========================================================

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

IMAGE_PATH = "birdhouse.jpg"
K = 8
MAX_ITER = 10
PROCESS_MAX = 512
PLOT_MAX_SIDE = 900


def squared_euclidean_batch(pixels, centroids):
    diff = pixels[:, None, :] - centroids[None, :, :]
    return np.sum(diff * diff, axis=2)

def resize_to_max(img, max_side=512):
    w, h = img.size
    if max(w, h) <= max_side:
        return img, False
    scale = max_side / float(max(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.BILINEAR), True

def init_centroids_deterministic(pixels, K):
    N = len(pixels)
    K = min(K, N)
    idx = np.linspace(0, N - 1, K, dtype=int)
    return pixels[idx].copy()

def kmeans_rgb_numpy(pixels_u8, K, max_iter=10, early_stop=True, tag="RGB"):
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

        print(f"[{tag}] Iter {it}: MSE_RGB = {mse:.2f}")

        if early_stop and prev_mse is not None and mse >= prev_mse:
            print(f"[{tag}] Early stop: MSE did not improve.")
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

def plot_preview(img: Image.Image, max_side=PLOT_MAX_SIDE) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    return img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)


if __name__ == "__main__":
    orig = Image.open(IMAGE_PATH).convert("RGB")
    ow, oh = orig.size
    orig_np = np.asarray(orig, dtype=np.uint8)
    orig_pixels = ow * oh

    print(f"\nUPLOAD: {ow}x{oh}  (pixels={orig_pixels:,})")

    # BASE runtime + MSE (proof)
    print("\n=== BASELINE: K-Means on ORIGINAL upload (no resizing) ===")
    t0 = time.perf_counter()
    pal_base, mse_base = kmeans_rgb_numpy(orig_np.reshape(-1, 3), K, MAX_ITER, early_stop=True, tag="BASE")
    t1 = time.perf_counter()
    runtime_base = t1 - t0
    final_mse_base = mse_base[-1]

    # ENH runtime + MSE (enhancement)
    print("\n=== ENHANCED: Resize to <=512 then K-Means ===")
    proc_img, resized = resize_to_max(orig, PROCESS_MAX)
    pw, ph = proc_img.size
    proc_np = np.asarray(proc_img, dtype=np.uint8)
    proc_pixels = pw * ph

    print(f"Processing cap: <= {PROCESS_MAX}px")
    print(f"Resized to: {pw}x{ph}  (pixels={proc_pixels:,})  | reduction ~{orig_pixels/proc_pixels:.1f}×")

    t2 = time.perf_counter()
    pal_enh, mse_enh = kmeans_rgb_numpy(proc_np.reshape(-1, 3), K, MAX_ITER, early_stop=True, tag="ENH")
    t3 = time.perf_counter()
    runtime_enh = t3 - t2
    final_mse_enh = mse_enh[-1]

    speedup = runtime_base / max(runtime_enh, 1e-9)
    print(f"\nSPEEDUP (BASE/ENH): ~{speedup:.1f}× faster with resizing")

    sw_enh = make_swatch_image(pal_enh)

    # -----------------------------
    # VISUAL: 1x3 (NO quantized)
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("SOP2 Enhancement (Resize <=512)", fontsize=16)

    # ✅ BIG summary line (now includes pixel counts)
    fig.text(
        0.5, 0.91,
        f"BASE: {runtime_base:.2f}s | MSE: {final_mse_base:.2f} | Pixels: {orig_pixels:,}   ||   "
        f"ENH: {runtime_enh:.2f}s | MSE: {final_mse_enh:.2f} | Pixels: {proc_pixels:,}   ||   "
        f"Speedup: ~{speedup:.1f}× | K={K}",
        ha="center", va="center",
        fontsize=17, fontweight="bold"
    )

    axes[0].imshow(plot_preview(orig))
    axes[0].set_title(f"Original Upload ({ow}×{oh}px | {orig_pixels:,} px)", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(proc_img)
    axes[1].set_title(f"Resized for Processing ({pw}×{ph}px | {proc_pixels:,} px)", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(np.asarray(sw_enh, dtype=np.uint8))
    axes[2].set_title("ENH Final palette swatch", fontsize=13)
    axes[2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()
