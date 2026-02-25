# ======================================================
# RGB Cube + Fast K-Means Palette Generation System
# Existing Algorithm (Huang 2021) with:
#  - Stage 1: RGB cube initial palette
#  - Stage 2: Fast K-Means + Wu–Lin
#  - Block-based sampling (rates: 1, 0.5, 0.25, 0.125, 0.0625, 0.03125)
# + RUNTIME ADDED (Stage 1 time, Stage 2 time, Total time)
# + OUTPUT CHANGED to 2x2 PANEL (Original | Quantized Output | RGB Cube | Swatch)
# + MSE DISPLAYED in the title (MSE_RGB from Stage 2)
# ✅ FIX: Quantized panel is now ACCURATE to the algorithm output
# ✅ FIX: Original/Quant/Swatch display CRISP (no matplotlib smoothing blur)
# ======================================================

from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm
import random
import math
import time

# ----------------- VISUAL GLOBAL FIX (NO BLUR) -----------------
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.resample"] = False

# ----------------- PARAMETERS -----------------
IMAGE_PATH ="kodim02.png"
K = 10
CUBE_BINS = 16
COUNT_THRESHOLD = 1

SAMPLE_RATE = 0.25
MAX_ITER = 10


# ----------------- HELPER FUNCTIONS -----------------
def rgb_to_cube_index(r, g, b, bins):
    rb = (r * bins) // 256
    gb = (g * bins) // 256
    bb = (b * bins) // 256
    return (rb, gb, bb)

def squared_euclidean(c1, c2):
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return dr*dr + dg*dg + db*db


# ------------- WU–LIN NEAREST-PALETTE ACCELERATION -------------
def wu_lin_nearest_color(r, g, b, palette, norms, len2s):
    x2 = r*r + g*g + b*b
    x_norm = math.sqrt(x2)

    Kp = len(palette)
    lo, hi = 0, Kp - 1
    best_k = 0
    best_diff = float('inf')

    while lo <= hi:
        mid = (lo + hi) // 2
        diff = norms[mid] - x_norm
        abs_diff = abs(diff)
        if abs_diff < best_diff:
            best_diff = abs_diff
            best_k = mid

        if diff < 0:
            lo = mid + 1
        elif diff > 0:
            hi = mid - 1
        else:
            break

    def consider(idx, sed1_min, nearest_idx):
        y = palette[idx]
        len2 = len2s[idx]
        dot_xy = r*y[0] + g*y[1] + b*y[2]
        sed1 = len2 - 2.0 * dot_xy
        if sed1 < sed1_min:
            return sed1, idx
        return sed1_min, nearest_idx

    sed1_min = float('inf')
    nearest_idx = best_k
    sed1_min, nearest_idx = consider(best_k, sed1_min, nearest_idx)

    i = best_k - 1
    while i >= 0:
        y_norm = norms[i]
        lower_bound = y_norm * (y_norm - 2.0 * x_norm)
        if lower_bound >= sed1_min:
            break
        sed1_min, nearest_idx = consider(i, sed1_min, nearest_idx)
        i -= 1

    i = best_k + 1
    while i < Kp:
        y_norm = norms[i]
        lower_bound = y_norm * (y_norm - 2.0 * x_norm)
        if lower_bound >= sed1_min:
            break
        sed1_min, nearest_idx = consider(i, sed1_min, nearest_idx)
        i += 1

    sed = x2 + sed1_min
    return nearest_idx, sed


# ----------------- STAGE 1: INITIAL PALETTE (RGB CUBE) -----------------
def build_rgb_cubes(img, bins, count_threshold):
    width, height = img.size
    cube_stats = {}

    print("Scanning image and building RGB cubes (Stage 1)...")
    for y in tqdm(range(height), desc="Rows processed"):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            cube_idx = rgb_to_cube_index(r, g, b, bins)
            if cube_idx not in cube_stats:
                cube_stats[cube_idx] = {"count": 0, "sum_r": 0, "sum_g": 0, "sum_b": 0}

            cube_stats[cube_idx]["count"] += 1
            cube_stats[cube_idx]["sum_r"] += r
            cube_stats[cube_idx]["sum_g"] += g
            cube_stats[cube_idx]["sum_b"] += b

    initc = []
    initn = []
    for stats in cube_stats.values():
        if stats["count"] >= count_threshold:
            c_count = stats["count"]
            mean_r = stats["sum_r"] // c_count
            mean_g = stats["sum_g"] // c_count
            mean_b = stats["sum_b"] // c_count
            initc.append((mean_r, mean_g, mean_b))
            initn.append(c_count)

    print(f"Number of candidate colors (initc): {len(initc)}")
    return initc, initn


# ----------------- BLOCK-BASED SAMPLING (STAGE 2 INPUT) -----------------
def block_sample_pixels(img, sampling_rate):
    width, height = img.size
    sampled = []

    if abs(sampling_rate - 1.0) < 1e-9:
        for y in range(height):
            for x in range(width):
                sampled.append(img.getpixel((x, y)))
        print(f"Sampling rate 1.0: sampled all {len(sampled)} pixels.")
        return sampled

    if abs(sampling_rate - 0.5) < 1e-9:
        block_size, samples_per_block = 2, 2
    elif abs(sampling_rate - 0.25) < 1e-9:
        block_size, samples_per_block = 2, 1
    elif abs(sampling_rate - 0.125) < 1e-9:
        block_size, samples_per_block = 4, 2
    elif abs(sampling_rate - 0.0625) < 1e-9:
        block_size, samples_per_block = 4, 1
    elif abs(sampling_rate - 0.03125) < 1e-9:
        block_size, samples_per_block = 8, 2
    else:
        print(f"Sampling rate {sampling_rate} not in Huang's set; using random per-pixel sampling.")
        for y in range(height):
            for x in range(width):
                if random.random() < sampling_rate:
                    sampled.append(img.getpixel((x, y)))
        print(
            f"Random sampling: sampled {len(sampled)} pixels "
            f"(effective rate ~ {len(sampled)/(width*height):.5f})"
        )
        return sampled

    for by in range(0, height, block_size):
        for bx in range(0, width, block_size):
            coords = []
            for y in range(by, min(by + block_size, height)):
                for x in range(bx, min(bx + block_size, width)):
                    coords.append((x, y))
            if not coords:
                continue

            k = min(samples_per_block, len(coords))
            chosen = random.sample(coords, k)
            for (x, y) in chosen:
                sampled.append(img.getpixel((x, y)))

    effective_rate = len(sampled) / (width * height)
    print(
        f"Block-based sampling: rate={sampling_rate}, "
        f"sampled {len(sampled)} pixels "
        f"(effective rate ~ {effective_rate:.5f})"
    )
    return sampled


# ----------------- INITIAL PALETTE (STAGE 1) -----------------
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
        best_idx = None
        best_score = -1.0
        for i in range(N):
            if selected[i]:
                continue
            dist_i = min(squared_euclidean(initc[i], p) for p in palette)
            score = dist_i * math.sqrt(initn[i])
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break

        selected[best_idx] = True
        palette.append(initc[best_idx])

    print(f"Initial palette generated (Stage 1) with {len(palette)} colors.")
    return palette


# ----------------- STAGE 2: FAST K-MEANS + WU–LIN -----------------
def fast_kmeans_palette_refinement(sampled_pixels, initial_palette, max_iter=10):
    if not sampled_pixels:
        print("No sampled pixels, skipping Stage 2 refinement.")
        return initial_palette, None

    palette = [list(c) for c in initial_palette]
    Kp = len(palette)
    SPN = len(sampled_pixels)

    Iter = 0
    StopF = 0
    prev_mse = None
    last_mse = None

    while Iter < max_iter and not StopF:
        len2s = []
        norms = []
        for c in palette:
            l2 = c[0]*c[0] + c[1]*c[1] + c[2]*c[2]
            len2s.append(l2)
            norms.append(math.sqrt(l2))

        combined = list(zip(palette, norms, len2s))
        combined.sort(key=lambda t: t[1])
        palette, norms, len2s = zip(*combined)
        palette = [list(c) for c in palette]
        norms = list(norms)
        len2s = list(len2s)

        clusters = [[] for _ in range(Kp)]
        mse_accum = 0.0

        for (r, g, b) in sampled_pixels:
            k_idx, sed = wu_lin_nearest_color(r, g, b, palette, norms, len2s)
            clusters[k_idx].append((r, g, b))
            mse_accum += sed

        MSE1_iter = mse_accum / SPN
        last_mse = MSE1_iter
        print(f"Iteration {Iter}: MSE1({Iter}) = {MSE1_iter:.2f}")

        if Iter > 0 and MSE1_iter >= prev_mse:
            StopF = 1
        prev_mse = MSE1_iter

        for k in range(Kp):
            if clusters[k]:
                sr = sum(p[0] for p in clusters[k]) / len(clusters[k])
                sg = sum(p[1] for p in clusters[k]) / len(clusters[k])
                sb = sum(p[2] for p in clusters[k]) / len(clusters[k])
                palette[k] = [int(sr), int(sg), int(sb)]

        palette.sort(key=lambda c: c[0]**2 + c[1]**2 + c[2]**2)
        Iter += 1

    refined_palette = [tuple(c) for c in palette]
    print(f"Final palette generated (Stage 2) after {Iter} iterations.")
    return refined_palette, last_mse


# ======================================================
# OUTPUT ONLY (2x2 PANEL)
# ======================================================
def make_palette_swatch_image(palette, swatch_h=70, w_per=50):
    if not palette:
        return Image.new("RGB", (w_per, swatch_h), (0, 0, 0))

    sw = Image.new("RGB", (w_per * len(palette), swatch_h))
    for i, color in enumerate(palette):
        for x in range(i * w_per, (i + 1) * w_per):
            for y in range(swatch_h):
                sw.putpixel((x, y), color)
    return sw

def downscale_for_preview(img, max_side=512):
    w, h = img.size
    if max(w, h) <= max_side:
        return img.copy()
    scale = max_side / float(max(w, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    try:
        resample = Image.Resampling.NEAREST  # ✅ display-only (crisp)
    except AttributeError:
        resample = Image.NEAREST
    return img.resize((nw, nh), resample)

def quantize_image_wulin(img, palette):
    w, h = img.size
    src = img.load()
    out = Image.new("RGB", (w, h))
    dst = out.load()

    pal = [list(c) for c in palette]
    len2s = []
    norms = []
    for c in pal:
        l2 = c[0]*c[0] + c[1]*c[1] + c[2]*c[2]
        len2s.append(l2)
        norms.append(math.sqrt(l2))

    combined = list(zip(pal, norms, len2s))
    combined.sort(key=lambda t: t[1])
    pal, norms, len2s = zip(*combined)
    pal = [list(c) for c in pal]
    norms = list(norms)
    len2s = list(len2s)

    for y in range(h):
        for x in range(w):
            r, g, b = src[x, y]
            idx, _ = wu_lin_nearest_color(r, g, b, pal, norms, len2s)
            dst[x, y] = tuple(pal[idx])

    return out

def plot_rgb_cube_on_ax(ax, sampled_pixels, palette, max_points=5000):
    if len(sampled_pixels) > max_points:
        plot_points = random.sample(sampled_pixels, max_points)
    else:
        plot_points = sampled_pixels

    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    ax.set_title("RGB Cube (Sampled)")

    for (r, g, b) in plot_points:
        ax.scatter(r, g, b, color=(r/255, g/255, b/255), s=5)

    for (r, g, b) in palette:
        ax.scatter(r, g, b, color=(r/255, g/255, b/255),
                   s=90, marker="X", edgecolors="black", linewidths=1.3)

def show_2x2_panel(img, sampled_pixels, final_palette,
                   stage1_time, stage2_time, total_time,
                   final_mse_rgb,
                   K, SAMPLE_RATE, MAX_ITER):
    swatch = make_palette_swatch_image(final_palette, swatch_h=70, w_per=50)

    quant_full = quantize_image_wulin(img, final_palette)
    quant = downscale_for_preview(quant_full, max_side=512)

    mse_txt = f"{final_mse_rgb:.2f}" if final_mse_rgb is not None else "N/A"

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.35, wspace=0.25)

    fig.suptitle(
        f"ORIGINAL (Huang 2021) | Total={total_time:.4f}s | Stage1={stage1_time:.4f}s | Stage2={stage2_time:.4f}s\n"
        f"MSE_RGB={mse_txt} | K={K} | SampleRate={SAMPLE_RATE} | MaxIter={MAX_ITER}",
        fontsize=12, fontweight="bold"
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img, interpolation="nearest")  # ✅ crisp
    ax1.set_title("Original Image (Input)")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(quant, interpolation="nearest")  # ✅ crisp
    ax2.set_title("Quantized Output")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0], projection="3d")
    plot_rgb_cube_on_ax(ax3, sampled_pixels, final_palette, max_points=5000)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(swatch, interpolation="nearest")  # ✅ crisp
    ax4.set_title("Final Palette Swatch")
    ax4.axis("off")

    plt.tight_layout()
    plt.show()


# ----------------- MAIN -----------------
if __name__ == "__main__":
    t_total0 = time.perf_counter()

    img = Image.open(IMAGE_PATH).convert("RGB")
    w, h = img.size
    total_pixels = w * h
    print(f"\nUPLOAD: {w}x{h}  (pixels={total_pixels:,})")
    print(f"PARAMS: K={K} | bins={CUBE_BINS} | thr={COUNT_THRESHOLD} | sample_rate={SAMPLE_RATE} | max_iter={MAX_ITER}")

    # Stage 1
    t_s10 = time.perf_counter()
    initc, initn = build_rgb_cubes(img, bins=CUBE_BINS, count_threshold=COUNT_THRESHOLD)
    initial_palette = initial_palette_generation(initc, initn, K=K)
    t_s11 = time.perf_counter()
    stage1_time = t_s11 - t_s10
    print(f"\nRUNTIME Stage 1: {stage1_time:.4f} s")

    # Stage 2 input
    sampled_pixels = block_sample_pixels(img, SAMPLE_RATE)
    print(f"Stage 2 input sampled pixels: {len(sampled_pixels):,}")

    # Stage 2
    t_s20 = time.perf_counter()
    final_palette, final_mse_rgb = fast_kmeans_palette_refinement(sampled_pixels, initial_palette, max_iter=MAX_ITER)
    t_s21 = time.perf_counter()
    stage2_time = t_s21 - t_s20
    print(f"RUNTIME Stage 2: {stage2_time:.4f} s")

    # Total
    t_total1 = time.perf_counter()
    total_time = t_total1 - t_total0
    print(f"RUNTIME Total (Stage1+Stage2+sampling): {total_time:.4f} s\n")

    show_2x2_panel(
        img, sampled_pixels, final_palette,
        stage1_time, stage2_time, total_time,
        final_mse_rgb,
        K, SAMPLE_RATE, MAX_ITER
    )
