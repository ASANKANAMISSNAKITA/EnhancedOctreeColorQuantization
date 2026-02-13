# ==========================================================
# SOP1 + SOP2 + SOP3 (ENHANCED) — Single Script (Matches Your Architecture)
#
# Stage 1 (Existing / Huang 2021):
#   Input Image -> RGB Cube Binning -> Pick Most Frequent -> Pick Most Distinct -> Initialized Palette
#
# Stage 2 (Enhanced for SOP1 + SOP2):
#   Initialized Palette + Image -> Resize ≤512 -> Sampling -> RGB→CIELAB -> Assign Nearest (Wu–Lin)
#   -> Update Means -> MSE Stop Checker -> Stage2 Palette
#
# Stage 3 (SOP3 Enhancement — Perceptual Separation using ΔE, threshold=4):
#   Stage2 Palette (CIELAB) -> Check pairwise ΔE76 -> If any pair < 4:
#   keep stronger color (bigger cluster support), replace weaker slot by selecting a new color
#   that maximizes minimum ΔE distance from the current palette (from sampled pixels).
#
# SPEED FIXES (keeps same logic but faster):
#   ✅ Convert sampled pixels RGB→LAB ONCE (not per-iteration)
#   ✅ Convert palette RGB→LAB once per iteration (palette updates each iter)
#   ✅ Quantized preview uses DOWNSCALED image (fast) + vectorized distance
#
# Requires: pillow, numpy, matplotlib, tqdm, scikit-image
# ==========================================================

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm
import random
import math
import time
from skimage.color import rgb2lab, lab2rgb

# ----------------- PARAMETERS -----------------
IMAGE_PATH = "4.1.02.tiff"

K = 8
CUBE_BINS = 16
COUNT_THRESHOLD = 1

# Huang discrete sampling rates:
# 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125
SAMPLE_RATE = 0.25

MAX_ITER = 10
RANDOM_SEED = 1

# SOP2: Stage 2 processing cap
PROCESS_MAX = 512

# SOP3: perceptual separation threshold (ΔE76 in Lab)
DELTA_E_THRESH = 4.0
SOP3_PASSES = 2                 # how many times to re-check after reseeding
SOP3_CANDIDATES = 2500          # how many sampled pixels to consider for reseeding (speed knob)

# Visual controls
SPHERE_POINTS = 5000
QUANT_PREVIEW_MAX = 256  # quantize preview downscale cap (speed)

# ============================================================
# STAGE 1 — RGB CUBE (Existing)
# ============================================================
def rgb_to_cube_index(r, g, b, bins):
    return ((r * bins) // 256, (g * bins) // 256, (b * bins) // 256)

def squared_euclidean_rgb(c1, c2):
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return dr*dr + dg*dg + db*db

def build_rgb_cubes(img, bins, count_threshold):
    """Stage 1 uses ALL pixels (no resize, no sampling)."""
    W, H = img.size
    cube_stats = {}  # (rb,gb,bb) -> [count, sumR, sumG, sumB]

    print("Stage 1: RGB Cube Binning...")
    for y in tqdm(range(H), desc="Stage 1 rows"):
        for x in range(W):
            r, g, b = img.getpixel((x, y))
            idx = rgb_to_cube_index(r, g, b, bins)
            if idx not in cube_stats:
                cube_stats[idx] = [0, 0, 0, 0]
            cube_stats[idx][0] += 1
            cube_stats[idx][1] += r
            cube_stats[idx][2] += g
            cube_stats[idx][3] += b

    initc, initn = [], []
    for count, sr, sg, sb in cube_stats.values():
        if count >= count_threshold:
            initc.append((sr // count, sg // count, sb // count))
            initn.append(count)

    print(f"Stage 1: Candidate colors initc = {len(initc)}")
    return initc, initn

def initial_palette_generation(initc, initn, K):
    """Pick Most Frequent then Pick Most Distinct (same structure as your original)."""
    N = len(initc)
    if N == 0:
        return []
    K = min(K, N)

    selected = [False] * N
    palette = []

    # Pick Most Frequent
    j = max(range(N), key=lambda i: initn[i])
    selected[j] = True
    palette.append(initc[j])

    # Pick Most Distinct (DistN)
    while len(palette) < K:
        best_i, best_score = None, -1.0
        for i in range(N):
            if selected[i]:
                continue
            dist_i = min(squared_euclidean_rgb(initc[i], p) for p in palette)
            score = dist_i * math.sqrt(initn[i])
            if score > best_score:
                best_score, best_i = score, i
        if best_i is None:
            break
        selected[best_i] = True
        palette.append(initc[best_i])

    print(f"Stage 1: Initialized palette size = {len(palette)}")
    return palette

# ============================================================
# STAGE 2 PRE-STEP — SOP2 Resize ≤512
# ============================================================
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

# ============================================================
# STAGE 2 INPUT — Sampling (Huang discrete rates)
# ============================================================
def block_sample_pixels(img, sampling_rate):
    W, H = img.size
    sampled = []

    if abs(sampling_rate - 1.0) < 1e-9:
        sampled = list(img.getdata())
        print(f"Stage 2: Sampling rate 1.0 -> {len(sampled)} pixels")
        return sampled

    if abs(sampling_rate - 0.5) < 1e-9:
        bs, spb = 2, 2
    elif abs(sampling_rate - 0.25) < 1e-9:
        bs, spb = 2, 1
    elif abs(sampling_rate - 0.125) < 1e-9:
        bs, spb = 4, 2
    elif abs(sampling_rate - 0.0625) < 1e-9:
        bs, spb = 4, 1
    elif abs(sampling_rate - 0.03125) < 1e-9:
        bs, spb = 8, 2
    else:
        print(f"Stage 2: Sampling rate {sampling_rate} not in Huang set -> random sampling")
        for y in range(H):
            for x in range(W):
                if random.random() < sampling_rate:
                    sampled.append(img.getpixel((x, y)))
        print(f"Stage 2: Random sampled -> {len(sampled)} pixels")
        return sampled

    for by in range(0, H, bs):
        for bx in range(0, W, bs):
            coords = [(x, y)
                      for y in range(by, min(by + bs, H))
                      for x in range(bx, min(bx + bs, W))]
            if not coords:
                continue
            chosen = random.sample(coords, min(spb, len(coords)))
            for (x, y) in chosen:
                sampled.append(img.getpixel((x, y)))

    eff = len(sampled) / float(W * H)
    print(f"Stage 2: Block sampling rate={sampling_rate} -> {len(sampled)} pixels (eff~{eff:.5f})")
    return sampled

# ============================================================
# SOP1 — RGB→CIELAB conversion helpers (VECTORIZED)
# ============================================================
def rgb_list_to_lab_array(rgb_list):
    """rgb_list: list[(R,G,B)] uint8 -> lab (N,3) float32"""
    arr = np.asarray(rgb_list, dtype=np.float32) / 255.0
    lab = rgb2lab(arr.reshape(-1, 1, 3)).reshape(-1, 3)
    return lab.astype(np.float32)

def lab_array_to_rgb_list(lab_arr):
    """lab_arr: (K,3) -> list[(R,G,B)] uint8"""
    rgb = lab2rgb(lab_arr.reshape(-1, 1, 3)).reshape(-1, 3)
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return [tuple(map(int, c)) for c in rgb_u8]

# ============================================================
# WU–LIN acceleration on 3D vectors (Lab)
# ============================================================
def _prep_wulin_3d(palette_3d):
    pal = np.asarray(palette_3d, dtype=np.float32)  # (K,3)
    len2s = np.sum(pal * pal, axis=1)
    norms = np.sqrt(len2s)
    order = np.argsort(norms)  # ascending
    return pal[order], norms[order], len2s[order], order

def wu_lin_nearest_color_3d(x0, x1, x2, pal_sorted, norms, len2s):
    x_sq = x0*x0 + x1*x1 + x2*x2
    x_norm = math.sqrt(x_sq)
    Kp = pal_sorted.shape[0]

    lo, hi = 0, Kp - 1
    best_k = 0
    best_diff = float("inf")

    while lo <= hi:
        mid = (lo + hi) // 2
        diff = norms[mid] - x_norm
        ad = abs(diff)
        if ad < best_diff:
            best_diff = ad
            best_k = mid
        if diff < 0:
            lo = mid + 1
        elif diff > 0:
            hi = mid - 1
        else:
            break

    def consider(idx, sed1_min, nearest_idx):
        y = pal_sorted[idx]
        len2 = float(len2s[idx])
        dot_xy = x0*y[0] + x1*y[1] + x2*y[2]
        sed1 = len2 - 2.0 * dot_xy
        if sed1 < sed1_min:
            return sed1, idx
        return sed1_min, nearest_idx

    sed1_min = float("inf")
    nearest_idx = best_k
    sed1_min, nearest_idx = consider(best_k, sed1_min, nearest_idx)

    i = best_k - 1
    while i >= 0:
        y_norm = float(norms[i])
        if y_norm * (y_norm - 2.0 * x_norm) >= sed1_min:
            break
        sed1_min, nearest_idx = consider(i, sed1_min, nearest_idx)
        i -= 1

    i = best_k + 1
    while i < Kp:
        y_norm = float(norms[i])
        if y_norm * (y_norm - 2.0 * x_norm) >= sed1_min:
            break
        sed1_min, nearest_idx = consider(i, sed1_min, nearest_idx)
        i += 1

    return nearest_idx, (x_sq + sed1_min)

# ============================================================
# STAGE 2 — Fast K-Means in CIELAB (Assign -> Update Means -> MSE Stop)
# (Returns palette_lab + cluster support counts for SOP3)
# ============================================================
def fast_kmeans_refine_lab(sampled_rgb, initial_palette_rgb, max_iter=10):
    if not sampled_rgb:
        return initial_palette_rgb, [], None, None

    sampled_lab = rgb_list_to_lab_array(sampled_rgb)              # (N,3)
    palette_lab = rgb_list_to_lab_array(initial_palette_rgb)      # (K,3)

    mse_hist = []
    prev_mse = None

    last_counts = None

    for it in range(max_iter):
        pal_sorted, norms, len2s, order = _prep_wulin_3d(palette_lab)

        counts = np.zeros(K, dtype=np.int32)
        sums = np.zeros((K, 3), dtype=np.float64)
        mse_accum = 0.0

        for p in sampled_lab:
            k_sorted, sed = wu_lin_nearest_color_3d(float(p[0]), float(p[1]), float(p[2]),
                                                   pal_sorted, norms, len2s)
            k = int(order[k_sorted])
            counts[k] += 1
            sums[k] += p
            mse_accum += sed

        mse = mse_accum / float(len(sampled_lab))
        mse_hist.append(mse)
        print(f"[Stage2-Lab] Iter {it}: MSE_Lab = {mse:.2f}")

        if prev_mse is not None and mse >= prev_mse:
            break
        prev_mse = mse

        for k in range(K):
            if counts[k] > 0:
                palette_lab[k] = (sums[k] / counts[k]).astype(np.float32)

        lens = np.sum(palette_lab * palette_lab, axis=1)
        palette_lab = palette_lab[np.argsort(lens)]

        last_counts = counts.copy()

    final_palette_rgb = lab_array_to_rgb_list(palette_lab)
    return final_palette_rgb, mse_hist, palette_lab, last_counts

# ============================================================
# SOP3 — Perceptual Separation (ΔE76 in Lab), threshold = 4
# ============================================================
def delta_e76_matrix(lab_arr):
    """Pairwise ΔE76 (K,K) for palette lab (K,3)."""
    diff = lab_arr[:, None, :] - lab_arr[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return np.sqrt(np.maximum(d2, 0.0))

def pick_reseed_candidate(sampled_lab, palette_lab, max_candidates=2000):
    """
    Choose a new color from sampled_lab that maximizes its minimum ΔE to current palette.
    Uses a subset of sampled points for speed.
    """
    N = sampled_lab.shape[0]
    if N == 0:
        return palette_lab[0].copy()

    if N > max_candidates:
        idx = np.random.choice(N, max_candidates, replace=False)
        cand = sampled_lab[idx]
    else:
        cand = sampled_lab

    # compute min ΔE to palette for each candidate
    diff = cand[:, None, :] - palette_lab[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    de = np.sqrt(np.maximum(d2, 0.0))
    min_de = np.min(de, axis=1)

    best = int(np.argmax(min_de))
    return cand[best].astype(np.float32)

def enforce_perceptual_separation(sampled_lab, palette_lab, counts, thresh=4.0, passes=2, max_candidates=2000):
    """
    If any two palette colors are too close (ΔE < thresh),
    keep the one with higher cluster support (counts),
    reseed the weaker slot with a candidate that is far from the palette.
    """
    pal = palette_lab.copy()
    cnt = counts.copy() if counts is not None else np.ones((pal.shape[0],), dtype=np.int32)

    for _ in range(passes):
        de = delta_e76_matrix(pal)
        np.fill_diagonal(de, np.inf)

        # find closest pair
        i, j = np.unravel_index(np.argmin(de), de.shape)
        min_de = float(de[i, j])

        if min_de >= thresh:
            break

        # choose keep/drop by cluster support
        if cnt[i] >= cnt[j]:
            keep, drop = i, j
        else:
            keep, drop = j, i

        # reseed dropped slot
        new_color = pick_reseed_candidate(sampled_lab, pal, max_candidates=max_candidates)
        pal[drop] = new_color
        cnt[drop] = 0  # unknown support after reseed (ok for repeated passes)

    return pal

# ============================================================
# VISUALS — CIELAB sphere + swatch + fast quantized preview
# ============================================================
def make_swatch_image(palette_rgb, swatch_h=70, w_per=50):
    img = Image.new("RGB", (w_per * len(palette_rgb), swatch_h))
    for i, c in enumerate(palette_rgb):
        c = tuple(map(int, c))
        for x in range(i * w_per, (i + 1) * w_per):
            for y in range(swatch_h):
                img.putpixel((x, y), c)
    return img

def downscale_for_preview(img, max_side=256):
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    return img.resize((nw, nh), resample)

def quantize_preview_lab_fast(img, palette_rgb):
    img_small = downscale_for_preview(img, QUANT_PREVIEW_MAX)
    arr_rgb = np.asarray(img_small, dtype=np.float32) / 255.0
    lab_img = rgb2lab(arr_rgb)  # (H,W,3)

    pal_lab = rgb_list_to_lab_array(palette_rgb).astype(np.float32)  # (K,3)
    H, W = lab_img.shape[:2]
    lab_flat = lab_img.reshape(-1, 3).astype(np.float32)

    d2 = np.sum((lab_flat[:, None, :] - pal_lab[None, :, :]) ** 2, axis=2)
    idx = np.argmin(d2, axis=1)

    pal_rgb = np.asarray(palette_rgb, dtype=np.uint8)
    out = pal_rgb[idx].reshape(H, W, 3)
    return Image.fromarray(out)

def plot_lab_sphere(ax, pixels_rgb, title, max_points=5000):
    pts = np.asarray(pixels_rgb, dtype=np.uint8)
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    lab = rgb2lab((pts.astype(np.float32) / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    ax.set_xlim(-128, 128)
    ax.set_ylim(-128, 128)
    ax.set_zlim(0, 100)
    ax.set_xlabel("a* (green ↔ red)")
    ax.set_ylabel("b* (blue ↔ yellow)")
    ax.set_zlabel("L* (lightness)")
    ax.set_title(title)

    theta = np.linspace(0, 2 * np.pi, 200)
    r = 80
    ax.plot(r*np.cos(theta), r*np.sin(theta), np.full_like(theta, 50),
            linestyle="--", linewidth=1)
    ax.plot([0, 0], [0, 0], [0, 100], color="black", linewidth=2)

    ax.scatter(a, b, L, c=pts.astype(np.float32) / 255.0, s=2, alpha=0.6)

# ============================================================
# MAIN — follows your proposed architecture order
# ============================================================
if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Input Image (Stage 1 uses original)
    orig = Image.open(IMAGE_PATH).convert("RGB")
    ow, oh = orig.size
    pixels_all = list(orig.getdata())

    # ---------- STAGE 1 ----------
    initc, initn = build_rgb_cubes(orig, CUBE_BINS, COUNT_THRESHOLD)
    initialized_palette = initial_palette_generation(initc, initn, K=K)

    # ---------- STAGE 2 ----------
    proc_img, resized = resize_to_max(orig, PROCESS_MAX)
    pw, ph = proc_img.size

    print("\nStage 2: Resize ≤512 (SOP2)")
    print(f"Original: {ow}x{oh} -> Processed: {pw}x{ph} (cap={PROCESS_MAX})")

    sampled_pixels = block_sample_pixels(proc_img, SAMPLE_RATE)

    print("Stage 2: RGB→CIELAB, Assign Nearest (Wu–Lin), Update Means, MSE Stop...")
    t0 = time.perf_counter()
    stage2_palette_rgb, mse_hist, stage2_palette_lab, stage2_counts = fast_kmeans_refine_lab(
        sampled_pixels, initialized_palette, max_iter=MAX_ITER
    )
    t1 = time.perf_counter()

    runtime = t1 - t0
    final_mse = mse_hist[-1] if mse_hist else 0.0
    print(f"\nSTAGE2 DONE: Runtime={runtime:.2f}s | Final MSE_Lab={final_mse:.2f} | K={K}")

    # ---------- STAGE 3 (SOP3) ----------
    # Apply perceptual separation on the Stage2 palette (Lab), threshold ΔE=4
    if stage2_palette_lab is not None:
        sampled_lab = rgb_list_to_lab_array(sampled_pixels)
        fixed_palette_lab = enforce_perceptual_separation(
            sampled_lab=sampled_lab,
            palette_lab=stage2_palette_lab,
            counts=stage2_counts,
            thresh=DELTA_E_THRESH,
            passes=SOP3_PASSES,
            max_candidates=SOP3_CANDIDATES
        )
        final_palette_rgb = lab_array_to_rgb_list(fixed_palette_lab)
        print(f"[SOP3] Applied perceptual separation: ΔE76 threshold = {DELTA_E_THRESH}")
    else:
        final_palette_rgb = stage2_palette_rgb

    # ---------- VISUALS ----------
    quant = quantize_preview_lab_fast(proc_img, final_palette_rgb)
    swatch = make_swatch_image(final_palette_rgb)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.35, wspace=0.25)

    fig.suptitle(
        f"SOP1+SOP2+SOP3 Enhanced | Runtime={runtime:.2f}s | MSE_Lab={final_mse:.2f} | "
        f"K={K} | Proc={pw}×{ph} | SampleRate={SAMPLE_RATE} | ΔE≥{DELTA_E_THRESH}",
        fontsize=12
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig)
    ax1.set_title("Original image (input)")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(quant)
    ax2.set_title("Quantized Preview (Final palette)")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0], projection="3d")
    plot_lab_sphere(ax3, pixels_all, "CIELAB Sphere (a*, b*, L*)", max_points=SPHERE_POINTS)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(np.asarray(swatch, dtype=np.uint8))
    ax4.set_title("Final palette swatch (RGB view)")
    ax4.axis("off")

    plt.tight_layout()
    plt.show()
