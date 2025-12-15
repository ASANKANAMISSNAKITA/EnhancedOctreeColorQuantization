# ======================================================
# RGB Cube + Fast K-Means Palette Generation System
# Existing Algorithm (Huang 2021) with:
#  - Stage 1: RGB cube initial palette
#  - Stage 2: Fast K-Means + Wu–Lin
#  - Block-based sampling (rates: 1, 0.5, 0.25, 0.125, 0.0625, 0.03125)
# ======================================================

from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm
import random
import math

# ----------------- PARAMETERS -----------------

IMAGE_PATH = "flower.jpg"   # <-- your image
K = 7                       # desired number of palette colors
CUBE_BINS = 16              # RGB cube division per axis (16 => 4096 cubes)
COUNT_THRESHOLD = 1         # Thr: minimum pixel count for a cube to be considered

# IMPORTANT: use one of the discrete rates from the paper:
# 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125
SAMPLE_RATE = 0.125         # data sampling rate for Stage 2
MAX_ITER = 10               # Max_cycle in the paper

# ----------------- HELPER FUNCTIONS -----------------

def rgb_to_cube_index(r, g, b, bins):
    """
    Map an RGB color to a cube index in a bins x bins x bins grid.
    """
    rb = (r * bins) // 256
    gb = (g * bins) // 256
    bb = (b * bins) // 256
    return (rb, gb, bb)

def squared_euclidean(c1, c2):
    """
    Squared Euclidean distance SED(c1, c2).
    """
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return dr*dr + dg*dg + db*db

# ------------- WU–LIN NEAREST-PALETTE ACCELERATION -------------

def wu_lin_nearest_color(r, g, b, palette, norms, len2s):
    """
    Wu–Lin accelerated nearest-palette search for one pixel x = (r,g,b).

    palette : list of [R,G,B] colors, sorted by length ||y||
    norms   : precomputed ||y|| for each palette entry
    len2s   : precomputed ||y||^2 for each palette entry

    Returns: (nearest_index, SED(x, y_nearest))
    """
    # ||x||^2 and ||x||
    x2 = r*r + g*g + b*b
    x_norm = math.sqrt(x2)

    K = len(palette)

    # ---- 1) find index whose ||y|| is closest to ||x|| ----
    lo, hi = 0, K - 1
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
            break  # exact match

    # helper to evaluate a candidate palette index
    def consider(idx, sed1_min, nearest_idx):
        y = palette[idx]
        len2 = len2s[idx]
        # dot(x,y)
        dot_xy = r*y[0] + g*y[1] + b*y[2]
        # SED1(x,y) = ||y||^2 - 2 * dot(x,y)
        sed1 = len2 - 2.0 * dot_xy
        if sed1 < sed1_min:
            return sed1, idx
        return sed1_min, nearest_idx

    # ---- 2) evaluate the closest-length color first ----
    sed1_min = float('inf')
    nearest_idx = best_k
    sed1_min, nearest_idx = consider(best_k, sed1_min, nearest_idx)

    # ---- 3) scan left with Wu–Lin kick-out condition ----
    i = best_k - 1
    while i >= 0:
        y_norm = norms[i]
        # Wu–Lin lower bound: ||y|| (||y|| − 2||x||)
        lower_bound = y_norm * (y_norm - 2.0 * x_norm)
        if lower_bound >= sed1_min:
            break  # no closer color further in this direction
        sed1_min, nearest_idx = consider(i, sed1_min, nearest_idx)
        i -= 1

    # ---- 4) scan right with Wu–Lin kick-out condition ----
    i = best_k + 1
    while i < K:
        y_norm = norms[i]
        lower_bound = y_norm * (y_norm - 2.0 * x_norm)
        if lower_bound >= sed1_min:
            break
        sed1_min, nearest_idx = consider(i, sed1_min, nearest_idx)
        i += 1

    # Convert SED1 back to full SED:
    # SED(x,y) = ||x||^2 + SED1(x,y)
    sed = x2 + sed1_min
    return nearest_idx, sed

# ----------------- STAGE 1: INITIAL PALETTE (RGB CUBE) -----------------

def build_rgb_cubes(img, bins, count_threshold):
    """
    Build initc(i) and initn(i) using an RGB cube.

    Stage 1 uses *all* pixels (no sampling).
    """
    width, height = img.size
    cube_stats = {}   # key: (rb,gb,bb), value: dict(count, sum_r, sum_g, sum_b)

    print("Scanning image and building RGB cubes (Stage 1)...")

    for y in tqdm(range(height), desc="Rows processed"):
        for x in range(width):
            r, g, b = img.getpixel((x, y))

            cube_idx = rgb_to_cube_index(r, g, b, bins)
            if cube_idx not in cube_stats:
                cube_stats[cube_idx] = {
                    "count": 0,
                    "sum_r": 0,
                    "sum_g": 0,
                    "sum_b": 0
                }
            cube_stats[cube_idx]["count"] += 1
            cube_stats[cube_idx]["sum_r"] += r
            cube_stats[cube_idx]["sum_g"] += g
            cube_stats[cube_idx]["sum_b"] += b

    # build initc(i), initn(i) from cubes that pass threshold
    initc = []  # candidate colors
    initn = []  # their frequencies

    for stats in cube_stats.values():
        if stats["count"] >= count_threshold:
            c_count = stats["count"]
            # (you can switch to round(...) if you want exact Huang behavior)
            mean_r = stats["sum_r"] // c_count
            mean_g = stats["sum_g"] // c_count
            mean_b = stats["sum_b"] // c_count
            initc.append((mean_r, mean_g, mean_b))
            initn.append(c_count)

    print(f"Number of candidate colors (initc): {len(initc)}")
    return initc, initn

# ----------------- BLOCK-BASED SAMPLING (STAGE 2 INPUT) -----------------

def block_sample_pixels(img, sampling_rate):
    """
    Block-based sampling to match Huang's discrete sampling rates:
      1.0    -> all pixels
      0.5    -> 2 pixels per 2x2 block
      0.25   -> 1 pixel per 2x2 block
      0.125  -> 2 pixels per 4x4 block
      0.0625 -> 1 pixel per 4x4 block
      0.03125 -> 2 pixels per 8x8 block

    For other rates, falls back to per-pixel random sampling.
    """
    width, height = img.size
    sampled = []

    # sampling_rate = 1.0 -> every pixel
    if abs(sampling_rate - 1.0) < 1e-9:
        for y in range(height):
            for x in range(width):
                sampled.append(img.getpixel((x, y)))
        print(f"Sampling rate 1.0: sampled all {len(sampled)} pixels.")
        return sampled

    # Decide block size and samples per block
    if abs(sampling_rate - 0.5) < 1e-9:
        block_size = 2
        samples_per_block = 2
    elif abs(sampling_rate - 0.25) < 1e-9:
        block_size = 2
        samples_per_block = 1
    elif abs(sampling_rate - 0.125) < 1e-9:
        block_size = 4
        samples_per_block = 2
    elif abs(sampling_rate - 0.0625) < 1e-9:
        block_size = 4
        samples_per_block = 1
    elif abs(sampling_rate - 0.03125) < 1e-9:
        block_size = 8
        samples_per_block = 2
    else:
        # fallback: simple per-pixel random sampling like your original code
        print(f"Sampling rate {sampling_rate} not in Huang's set; using random per-pixel sampling.")
        for y in range(height):
            for x in range(width):
                if random.random() < sampling_rate:
                    sampled.append(img.getpixel((x, y)))
        print(f"Random sampling: sampled {len(sampled)} pixels "
              f"(effective rate ~ {len(sampled)/(width*height):.5f})")
        return sampled

    # Block-based sampling
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
    print(f"Block-based sampling: rate={sampling_rate}, "
          f"sampled {len(sampled)} pixels "
          f"(effective rate ~ {effective_rate:.5f})")

    return sampled

# ----------------- INITIAL PALETTE (STAGE 1) -----------------

def initial_palette_generation(initc, initn, K):
    """
    Stage 1: Initial Palette Generation (existing algorithm)

    Step 1: Selected(i) = 0, Cno = 0.
    Step 2: choose initc(j) with max initn(j).
    Step 3: DistN(i) = Dist(i) * sqrt(initn(i)) for unselected colors.
    Step 4: repeat until Cno == K.
    """
    N = len(initc)
    if N == 0:
        return []

    K = min(K, N)

    selected = [False] * N
    palette = []

    # Step 1: initial palette empty, Cno = 0
    Cno = 0

    # Step 2: choose the candidate with max initn(i)
    j = max(range(N), key=lambda i: initn[i])
    selected[j] = True
    palette.append(initc[j])
    Cno += 1

    # Step 3 & 4: iteratively add colors until we have K colors
    while Cno < K:
        best_idx = None
        best_score = -1.0

        for i in range(N):
            if selected[i]:
                continue

            # Dist(i): min squared distance to any color in current palette
            dist_i = min(squared_euclidean(initc[i], p) for p in palette)
            # DistN(i) = Dist(i) * (initn(i))^0.5
            score = dist_i * math.sqrt(initn[i])

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break  # no more candidates

        selected[best_idx] = True
        palette.append(initc[best_idx])
        Cno += 1

    print(f"Initial palette generated (Stage 1) with {len(palette)} colors.")
    return palette

# ----------------- STAGE 2: FAST K-MEANS + WU–LIN -----------------

def fast_kmeans_palette_refinement(sampled_pixels, initial_palette, max_iter=10):
    """
    Stage 2: Fast K-Means Algorithm (existing algorithm + Wu–Lin acceleration)
    """
    if not sampled_pixels:
        print("No sampled pixels, skipping Stage 2 refinement.")
        return initial_palette

    palette = [list(c) for c in initial_palette]
    K = len(palette)
    SPN = len(sampled_pixels)

    Iter = 0
    StopF = 0
    prev_mse = None

    while Iter < max_iter and not StopF:
        # --- precompute norms and squared norms for current palette ---
        len2s = []
        norms = []
        for c in palette:
            l2 = c[0]*c[0] + c[1]*c[1] + c[2]*c[2]
            len2s.append(l2)
            norms.append(math.sqrt(l2))

        # ensure palette is sorted by length (ascending)
        combined = list(zip(palette, norms, len2s))
        combined.sort(key=lambda t: t[1])  # sort by ||y||
        palette, norms, len2s = zip(*combined)
        palette = [list(c) for c in palette]
        norms = list(norms)
        len2s = list(len2s)

        # Step 2: assignment step (Wu–Lin nearest search)
        clusters = [[] for _ in range(K)]
        mse_accum = 0.0

        for (r, g, b) in sampled_pixels:
            k_idx, sed = wu_lin_nearest_color(r, g, b, palette, norms, len2s)
            clusters[k_idx].append((r, g, b))
            mse_accum += sed  # SED(SCP_i, CCP_i)

        # Step 4: compute MSE1(Iter)
        MSE1_iter = mse_accum / SPN
        print(f"Iteration {Iter}: MSE1({Iter}) = {MSE1_iter:.2f}")

        if Iter > 0 and MSE1_iter >= prev_mse:
            StopF = 1
        prev_mse = MSE1_iter

        # Step 3: update palette from group means
        for k in range(K):
            if clusters[k]:
                sr = sum(p[0] for p in clusters[k]) / len(clusters[k])
                sg = sum(p[1] for p in clusters[k]) / len(clusters[k])
                sb = sum(p[2] for p in clusters[k]) / len(clusters[k])
                palette[k] = [int(sr), int(sg), int(sb)]
            # if cluster is empty, keep old color

        # sort K palette colors in ascending order of length for next iteration
        palette.sort(key=lambda c: c[0]**2 + c[1]**2 + c[2]**2)

        Iter += 1

    refined_palette = [tuple(c) for c in palette]
    print(f"Final palette generated (Stage 2) after {Iter} iterations.")
    return refined_palette

# ----------------- VISUALIZATION HELPERS -----------------

def show_palette_swatch(palette, swatch_size=50):
    """
    Display a simple horizontal palette swatch image.
    """
    if not palette:
        print("No palette to display.")
        return

    palette_img = Image.new("RGB", (swatch_size * len(palette), swatch_size))
    for i, color in enumerate(palette):
        for x in range(i * swatch_size, (i + 1) * swatch_size):
            for y in range(swatch_size):
                palette_img.putpixel((x, y), color)

    print("Showing palette swatch...")
    palette_img.show()

def plot_rgb_cube(sampled_pixels, palette, max_points=5000):
    """
    Plot sampled pixels in RGB cube and highlight palette colors as 'X'.
    """
    print("Plotting RGB cube with palette colors...")

    if len(sampled_pixels) > max_points:
        plot_points = random.sample(sampled_pixels, max_points)
    else:
        plot_points = sampled_pixels

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    ax.set_title('RGB Cube - Sampled Colors and Palette')

    # plot sampled pixels
    for (r, g, b) in plot_points:
        ax.scatter(r, g, b, color=(r/255, g/255, b/255), s=5)

    # highlight palette colors
    for color in palette:
        r, g, b = color
        ax.scatter(r, g, b, color=(r/255, g/255, b/255),
                   s=80, marker='X', edgecolors='black', linewidths=1.5)

    plt.show()
    print("Plotting complete. Palette colors are marked as 'X'.")

# ----------------- MAIN -----------------

if __name__ == "__main__":
    # load image
    img = Image.open(IMAGE_PATH).convert("RGB")

    # Stage 1: build RGB cubes and generate initial palette (no sampling here)
    initc, initn = build_rgb_cubes(
        img,
        bins=CUBE_BINS,
        count_threshold=COUNT_THRESHOLD
    )

    initial_palette = initial_palette_generation(initc, initn, K=K)

    # Stage 2: block-based sampling + fast K-Means
    sampled_pixels = block_sample_pixels(img, SAMPLE_RATE)

    final_palette = fast_kmeans_palette_refinement(
        sampled_pixels,
        initial_palette,
        max_iter=MAX_ITER
    )

    # show palette as swatch
    show_palette_swatch(final_palette)

    # show RGB cube with palette points
    plot_rgb_cube(sampled_pixels, final_palette)
