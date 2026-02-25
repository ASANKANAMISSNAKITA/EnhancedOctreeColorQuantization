# ==========================================================
# SOP1 + SOP2 + SOP3 (ENHANCED) — FAST VERSION (same logic, fixed speed)
#
# Stage 1: RGB Cube binning -> most frequent -> most distinct -> init palette
# Stage 2: Resize <=512 -> Sampling -> RGB->CIELAB -> Assign nearest -> Update means -> MSE stop
# Stage 3: ΔE76 de-duplicate (threshold=4) with reseeding from sampled pixels
#
# VIS FIX:
#   ✅ Quantized output is generated at proc_img resolution (NOT 256 preview)
#   ✅ imshow interpolation disabled (crisper display)
#   ✅ Figure size stays EXACTLY (12, 8)
# ==========================================================

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import random
import math
import time
from skimage.color import rgb2lab, lab2rgb

# ----------------- PARAMETERS -----------------
IMAGE_PATH = "kodim02.png"

K = 10
CUBE_BINS = 16
COUNT_THRESHOLD = 1

SAMPLE_RATE = 0.25
MAX_ITER = 10
RANDOM_SEED = 1

PROCESS_MAX = 512

DELTA_E_THRESH = 4
SOP3_PASSES = 2
SOP3_CANDIDATES = 2500  # speed knob

SPHERE_POINTS = 5000

# Stage2 speed knob (prevents NxK memory blow-up on huge images)
CHUNK_SIZE = 120000  # 50k–300k depending on RAM


# ============================================================
# Helpers
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


def rgb_list_to_lab_array(rgb_list_or_arr):
    """Accept list[(R,G,B)] or ndarray(N,3) uint8 -> lab (N,3) float32"""
    arr = np.asarray(rgb_list_or_arr, dtype=np.float32) / 255.0
    lab = rgb2lab(arr.reshape(-1, 1, 3)).reshape(-1, 3)
    return lab.astype(np.float32)


def lab_array_to_rgb_list(lab_arr):
    rgb = lab2rgb(lab_arr.reshape(-1, 1, 3)).reshape(-1, 3)
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return [tuple(map(int, c)) for c in rgb_u8]


# ============================================================
# STAGE 1 — RGB CUBE (FAST vectorized)
# ============================================================
def build_rgb_cubes_fast(img, bins, count_threshold):
    arr = np.asarray(img, dtype=np.uint8)  # (H,W,3)
    pix = arr.reshape(-1, 3).astype(np.int32)  # (N,3)
    N = pix.shape[0]

    rb = (pix[:, 0] * bins) // 256
    gb = (pix[:, 1] * bins) // 256
    bb = (pix[:, 2] * bins) // 256

    cube_id = (rb * bins + gb) * bins + bb
    B3 = bins * bins * bins

    counts = np.bincount(cube_id, minlength=B3).astype(np.int32)
    sum_r = np.bincount(cube_id, weights=pix[:, 0], minlength=B3)
    sum_g = np.bincount(cube_id, weights=pix[:, 1], minlength=B3)
    sum_b = np.bincount(cube_id, weights=pix[:, 2], minlength=B3)

    mask = counts >= count_threshold
    idxs = np.nonzero(mask)[0]

    initc = []
    initn = []
    for cid in idxs:
        c = counts[cid]
        mr = int(sum_r[cid] // c)
        mg = int(sum_g[cid] // c)
        mb = int(sum_b[cid] // c)
        initc.append((mr, mg, mb))
        initn.append(int(c))

    print(f"Stage 1: Candidate colors initc = {len(initc)} (from {N:,} pixels)")
    return initc, initn


def squared_euclidean_rgb(c1, c2):
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return dr*dr + dg*dg + db*db


def initial_palette_generation(initc, initn, K):
    N = len(initc)
    if N == 0:
        return []
    K = min(K, N)

    selected = [False] * N
    palette = []

    j = max(range(N), key=lambda i: initn[i])
    selected[j] = True
    palette.append(initc[j])

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
# STAGE 2 INPUT — Sampling (Huang discrete rates)
# ============================================================
def block_sample_pixels(img, sampling_rate):
    W, H = img.size

    if abs(sampling_rate - 1.0) < 1e-9:
        sampled = np.asarray(img, dtype=np.uint8).reshape(-1, 3)
        print(f"Stage 2: Sampling rate 1.0 -> {len(sampled):,} pixels")
        return sampled

    sampled = []
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
        sampled = np.asarray(sampled, dtype=np.uint8).reshape(-1, 3)
        print(f"Stage 2: Random sampled -> {len(sampled):,} pixels")
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

    sampled = np.asarray(sampled, dtype=np.uint8).reshape(-1, 3)
    eff = len(sampled) / float(W * H)
    print(f"Stage 2: Block sampling rate={sampling_rate} -> {len(sampled):,} pixels (eff~{eff:.5f})")
    return sampled


# ============================================================
# STAGE 2 — Fast K-Means in CIELAB (FAST chunked assignment)
# ============================================================
def init_centroids_from_palette_lab(palette_rgb):
    return rgb_list_to_lab_array(palette_rgb).astype(np.float32)


def fast_kmeans_refine_lab_fast(sampled_rgb_u8, initial_palette_rgb, max_iter=10, chunk_size=120000):
    if sampled_rgb_u8 is None or len(sampled_rgb_u8) == 0:
        return initial_palette_rgb, [], None, None, None

    sampled_lab = rgb_list_to_lab_array(sampled_rgb_u8)
    N = sampled_lab.shape[0]

    centroids = init_centroids_from_palette_lab(initial_palette_rgb)
    Kc = centroids.shape[0]

    mse_hist = []
    prev_mse = None
    last_counts = None

    for it in range(max_iter):
        counts = np.zeros(Kc, dtype=np.int32)
        sums = np.zeros((Kc, 3), dtype=np.float64)
        mse_sum = 0.0

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = sampled_lab[start:end]

            diff = chunk[:, None, :] - centroids[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            labels = np.argmin(d2, axis=1)
            min_d2 = d2[np.arange(len(chunk)), labels]

            mse_sum += float(np.sum(min_d2))

            counts += np.bincount(labels, minlength=Kc).astype(np.int32)
            for ch in range(3):
                sums[:, ch] += np.bincount(labels, weights=chunk[:, ch], minlength=Kc)

        mse = mse_sum / float(N)
        mse_hist.append(mse)
        print(f"[Stage2-Lab] Iter {it}: MSE_Lab = {mse:.2f}")

        if prev_mse is not None and mse >= prev_mse:
            break
        prev_mse = mse

        mask = counts > 0
        centroids[mask] = (sums[mask] / counts[mask, None]).astype(np.float32)

        lens = np.sum(centroids * centroids, axis=1)
        order = np.argsort(lens)
        centroids = centroids[order]
        counts = counts[order]
        last_counts = counts.copy()

    final_palette_rgb = lab_array_to_rgb_list(centroids)
    return final_palette_rgb, mse_hist, centroids, last_counts, sampled_lab


# ============================================================
# SOP3 — ΔE de-dup + reseed
# ============================================================
def delta_e76_matrix(lab_arr):
    diff = lab_arr[:, None, :] - lab_arr[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return np.sqrt(np.maximum(d2, 0.0))


def pick_reseed_candidate(sampled_lab, palette_lab, max_candidates=2000):
    N = sampled_lab.shape[0]
    if N == 0:
        return palette_lab[0].copy()

    if N > max_candidates:
        idx = np.random.choice(N, max_candidates, replace=False)
        cand = sampled_lab[idx]
    else:
        cand = sampled_lab

    diff = cand[:, None, :] - palette_lab[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    de = np.sqrt(np.maximum(d2, 0.0))
    min_de = np.min(de, axis=1)

    best = int(np.argmax(min_de))
    return cand[best].astype(np.float32)


def enforce_perceptual_separation(sampled_lab, palette_lab, counts, thresh=4.0, passes=2, max_candidates=2000):
    pal = palette_lab.copy()
    cnt = counts.copy() if counts is not None else np.ones((pal.shape[0],), dtype=np.int32)

    for _ in range(passes):
        de = delta_e76_matrix(pal)
        np.fill_diagonal(de, np.inf)

        i, j = np.unravel_index(np.argmin(de), de.shape)
        min_de = float(de[i, j])
        if min_de >= thresh:
            break

        keep, drop = (i, j) if cnt[i] >= cnt[j] else (j, i)
        pal[drop] = pick_reseed_candidate(sampled_lab, pal, max_candidates=max_candidates)
        cnt[drop] = 0

    return pal


# ============================================================
# SOP3 DIAGNOSTICS — detect near-duplicate pairs + show replaced slots
# (ADDED: does not change the algorithm, only prints what happens)
# ============================================================
def find_near_duplicate_pairs(palette_lab, thresh):
    """
    Returns list of (i, j, deltaE) for all pairs with ΔE < thresh.
    """
    if palette_lab is None or len(palette_lab) < 2:
        return []

    de = delta_e76_matrix(palette_lab.astype(np.float32))
    pairs = []
    Kc = de.shape[0]
    for i in range(Kc):
        for j in range(i + 1, Kc):
            if de[i, j] < thresh:
                pairs.append((i, j, float(de[i, j])))

    pairs.sort(key=lambda x: x[2])  # smallest ΔE first
    return pairs


def report_palette_duplicates(label, palette_rgb, palette_lab, thresh):
    """
    Prints near-duplicate pairs (ΔE < thresh) and min ΔE.
    """
    print(f"\n[{label}] Palette duplicate check (ΔE76 < {thresh}):")

    if palette_lab is None or len(palette_lab) < 2:
        print("  (Palette not available / too small)")
        return

    de = delta_e76_matrix(palette_lab.astype(np.float32))
    np.fill_diagonal(de, np.inf)
    min_de = float(np.min(de))
    print(f"  Min pairwise ΔE76: {min_de:.2f}")

    pairs = find_near_duplicate_pairs(palette_lab, thresh)
    if not pairs:
        print("  ✅ No near-duplicate pairs found.")
        return

    print(f"  ⚠ Found {len(pairs)} near-duplicate pair(s):")
    for (i, j, d) in pairs:
        rgb_i = palette_rgb[i] if palette_rgb is not None else None
        rgb_j = palette_rgb[j] if palette_rgb is not None else None
        print(f"   - Pair ({i}, {j}) ΔE={d:.2f} | RGB{i}={rgb_i} vs RGB{j}={rgb_j}")


def report_sop3_replacements(stage2_lab, fixed_lab, stage2_rgb, final_rgb, eps=1e-3):
    """
    Shows which palette indices were replaced by SOP3 by comparing LAB before/after
    at the same index (SOP3 modifies pal[drop] in place, keeping index order).
    """
    print("\n[SOP3] Replacement summary:")

    if stage2_lab is None or fixed_lab is None:
        print("  (Palette LAB not available; cannot compare.)")
        return

    stage2_lab = np.asarray(stage2_lab, dtype=np.float32)
    fixed_lab = np.asarray(fixed_lab, dtype=np.float32)

    if stage2_lab.shape != fixed_lab.shape:
        print("  (Palette shapes differ; cannot compare.)")
        return

    # ΔE between same index before/after (using Euclidean in LAB, consistent with ΔE76 core)
    diffs = np.sqrt(np.sum((stage2_lab - fixed_lab) ** 2, axis=1))
    changed = np.where(diffs > eps)[0].tolist()

    if not changed:
        print("  No palette slots were replaced (SOP3 made no changes).")
        return

    print(f"  Replaced slot index/indices: {changed}")
    for idx in changed:
        before_rgb = stage2_rgb[idx] if stage2_rgb is not None else None
        after_rgb = final_rgb[idx] if final_rgb is not None else None
        print(f"   - Slot {idx}: {before_rgb}  →  {after_rgb}  (ΔLab≈{diffs[idx]:.4f})")


# ============================================================
# VISUALS (CRISP + ACCURATE)
# ============================================================
def make_swatch_image(palette_rgb, swatch_h=70, w_per=50):
    img = Image.new("RGB", (w_per * len(palette_rgb), swatch_h))
    for i, c in enumerate(palette_rgb):
        c = tuple(map(int, c))
        for x in range(i * w_per, (i + 1) * w_per):
            for y in range(swatch_h):
                img.putpixel((x, y), c)
    return img


def quantize_lab_image_fullres(img, palette_rgb, chunk_size=120000):
    """
    Quantize at img resolution using Lab nearest centroid (chunked).
    Returns PIL image same size as img.
    """
    arr_rgb = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
    lab_img = rgb2lab(arr_rgb).astype(np.float32)

    H, W = lab_img.shape[:2]
    lab_flat = lab_img.reshape(-1, 3)

    pal_lab = rgb_list_to_lab_array(palette_rgb).astype(np.float32)
    pal_rgb = np.asarray(palette_rgb, dtype=np.uint8)

    out_flat = np.empty((lab_flat.shape[0], 3), dtype=np.uint8)

    for start in range(0, lab_flat.shape[0], chunk_size):
        end = min(start + chunk_size, lab_flat.shape[0])
        chunk = lab_flat[start:end]
        d2 = np.sum((chunk[:, None, :] - pal_lab[None, :, :]) ** 2, axis=2)
        idx = np.argmin(d2, axis=1)
        out_flat[start:end] = pal_rgb[idx]

    out = out_flat.reshape(H, W, 3)
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
# MAIN
# ============================================================
if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    t_total0 = time.perf_counter()

    orig = Image.open(IMAGE_PATH).convert("RGB")
    pixels_all = np.asarray(orig, dtype=np.uint8).reshape(-1, 3)

    # ---------- STAGE 1 ----------
    t_s10 = time.perf_counter()
    initc, initn = build_rgb_cubes_fast(orig, CUBE_BINS, COUNT_THRESHOLD)
    initialized_palette = initial_palette_generation(initc, initn, K=K)
    t_s11 = time.perf_counter()
    stage1_time = t_s11 - t_s10

    # ---------- STAGE 2 ----------
    proc_img, resized = resize_to_max(orig, PROCESS_MAX)
    pw, ph = proc_img.size
    sampled_pixels = block_sample_pixels(proc_img, SAMPLE_RATE)

    print("\nStage 2: RGB→CIELAB, Assign Nearest, Update Means, MSE Stop... (FAST)")
    t_s20 = time.perf_counter()
    stage2_palette_rgb, mse_hist, stage2_palette_lab, stage2_counts, sampled_lab = fast_kmeans_refine_lab_fast(
        sampled_pixels, initialized_palette, max_iter=MAX_ITER, chunk_size=CHUNK_SIZE
    )
    t_s21 = time.perf_counter()
    stage2_time = t_s21 - t_s20

    final_mse = mse_hist[-1] if mse_hist else 0.0

    # ---------- DIAGNOSTICS: BEFORE SOP3 ----------
    report_palette_duplicates(
        label="BEFORE SOP3 (Stage 2 Palette)",
        palette_rgb=stage2_palette_rgb,
        palette_lab=stage2_palette_lab,
        thresh=DELTA_E_THRESH
    )

    # ---------- STAGE 3 ----------
    if stage2_palette_lab is not None:
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

        # ---------- DIAGNOSTICS: AFTER SOP3 + WHAT CHANGED ----------
        report_palette_duplicates(
            label="AFTER SOP3 (Final Palette)",
            palette_rgb=final_palette_rgb,
            palette_lab=fixed_palette_lab,
            thresh=DELTA_E_THRESH
        )

        report_sop3_replacements(
            stage2_lab=stage2_palette_lab,
            fixed_lab=fixed_palette_lab,
            stage2_rgb=stage2_palette_rgb,
            final_rgb=final_palette_rgb,
            eps=1e-3
        )
    else:
        final_palette_rgb = stage2_palette_rgb

    t_total1 = time.perf_counter()
    total_time = t_total1 - t_total0

    print("\n====================")
    print(f"ENH Total Runtime: {total_time:.4f}s")
    print(f"Stage1 Runtime:    {stage1_time:.4f}s")
    print(f"Stage2 Runtime:    {stage2_time:.4f}s")
    print(f"Final MSE_Lab:     {final_mse:.2f}")
    print("====================\n")

    # ---------- VISUALS (ACCURATE + CRISP) ----------
    quant = quantize_lab_image_fullres(proc_img, final_palette_rgb, chunk_size=CHUNK_SIZE)
    swatch = make_swatch_image(final_palette_rgb)

    # ✅ KEEP FIGURE SIZE EXACTLY LIKE YOURS
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.35, wspace=0.25)

    fig.suptitle(
        f"ENHANCED SOP1+SOP2+SOP3 | Total={total_time:.4f}s | Stage1={stage1_time:.4f}s | Stage2={stage2_time:.4f}s\n"
        f"MSE_Lab={final_mse:.2f} | K={K} | Proc={pw}×{ph} | SampleRate={SAMPLE_RATE} | ΔE≥{DELTA_E_THRESH}",
        fontsize=12, fontweight="bold"
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig, interpolation="nearest")
    ax1.set_title("Original image (input)")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(quant, interpolation="nearest")
    ax2.set_title("Quantized Output")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0], projection="3d")
    plot_lab_sphere(ax3, pixels_all, "CIELAB Sphere (a*, b*, L*)", max_points=SPHERE_POINTS)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(np.asarray(swatch, dtype=np.uint8), interpolation="nearest")
    ax4.set_title("Final palette swatch (RGB view)")
    ax4.axis("off")

    plt.tight_layout()
    plt.show()