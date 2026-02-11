# ======================================================
# ENHANCED SOP 3: Fix Duplicate / near-duplicate palette colors
# Based on your RGB Cube + Fast K-Means (Huang 2021) pipeline
#
# Enhancements:
#  1) Reseed empty/weak clusters (prevents dead centroids collapsing)
#  2) De-duplicate palette colors (merge + reseed freed slot)
# ======================================================

from PIL import Image
import random, math
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------- PARAMETERS -----------------
IMAGE_PATH = "starry.jpg"
K = 8
CUBE_BINS = 16
COUNT_THRESHOLD = 1
SAMPLE_RATE = 0.25
MAX_ITER = 10

RANDOM_SEED = 7

# If two palette colors are closer than this (Euclidean RGB), mark as near-duplicate
DUP_THRESH = 50  # your current choice

# Enhancement knobs
MIN_CLUSTER_SIZE = 20        # treat clusters smaller than this as "weak"
DEDUP_PASSES = 3             # run dedup multiple times (cheap, helps stabilize)
SHOW_BEFORE_AFTER = True     # show palette before and after fix
SHOW_QUANTIZED = False       # set True if you want before/after quantized images
QUANTIZE_DOWNSCALE = 2       # for speed if SHOW_QUANTIZED True

# ----------------- HELPERS -----------------

def rgb_to_cube_index(r, g, b, bins):
    return ((r * bins) // 256, (g * bins) // 256, (b * bins) // 256)

def sed(c1, c2):
    dr = c1[0]-c2[0]
    dg = c1[1]-c2[1]
    db = c1[2]-c2[2]
    return dr*dr + dg*dg + db*db

def euclid(c1, c2):
    return math.sqrt(sed(c1, c2))

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
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_k = mid
        if diff < 0: lo = mid + 1
        elif diff > 0: hi = mid - 1
        else: break

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
        if y_norm * (y_norm - 2.0 * x_norm) >= sed1_min:
            break
        sed1_min, nearest_idx = consider(i, sed1_min, nearest_idx)
        i -= 1

    i = best_k + 1
    while i < Kp:
        y_norm = norms[i]
        if y_norm * (y_norm - 2.0 * x_norm) >= sed1_min:
            break
        sed1_min, nearest_idx = consider(i, sed1_min, nearest_idx)
        i += 1

    return nearest_idx, x2 + sed1_min

def _prep_wulin(palette):
    pal = [list(c) for c in palette]
    len2s, norms = [], []
    for c in pal:
        l2 = c[0]*c[0] + c[1]*c[1] + c[2]*c[2]
        len2s.append(l2)
        norms.append(math.sqrt(l2))
    combo = list(zip(pal, norms, len2s))
    combo.sort(key=lambda t: t[1])
    pal, norms, len2s = zip(*combo)
    return [list(c) for c in pal], list(norms), list(len2s)

# ----------------- STAGE 1 -----------------

def build_rgb_cubes(img, bins, count_threshold):
    W, H = img.size
    cube = {}
    for y in tqdm(range(H), desc="Stage 1 rows"):
        for x in range(W):
            r, g, b = img.getpixel((x, y))
            idx = rgb_to_cube_index(r, g, b, bins)
            if idx not in cube:
                cube[idx] = [0, 0, 0, 0]  # count, sumR, sumG, sumB
            cube[idx][0] += 1
            cube[idx][1] += r
            cube[idx][2] += g
            cube[idx][3] += b

    initc, initn = [], []
    for (count, sr, sg, sb) in cube.values():
        if count >= count_threshold:
            initc.append((sr // count, sg // count, sb // count))
            initn.append(count)
    return initc, initn

def initial_palette_generation(initc, initn, K):
    N = len(initc)
    if N == 0: return []
    K = min(K, N)
    selected = [False]*N

    j = max(range(N), key=lambda i: initn[i])
    selected[j] = True
    palette = [initc[j]]

    while len(palette) < K:
        best_i, best_score = None, -1
        for i in range(N):
            if selected[i]: continue
            dist_i = min(sed(initc[i], p) for p in palette)
            score = dist_i * math.sqrt(initn[i])
            if score > best_score:
                best_score, best_i = score, i
        if best_i is None: break
        selected[best_i] = True
        palette.append(initc[best_i])
    return palette

# ----------------- STAGE 2 INPUT (sampling) -----------------

def block_sample_pixels(img, sampling_rate):
    W, H = img.size
    sampled = []

    if abs(sampling_rate - 1.0) < 1e-9:
        for y in range(H):
            for x in range(W):
                sampled.append(img.getpixel((x, y)))
        return sampled

    if abs(sampling_rate - 0.5) < 1e-9:   bs, spb = 2, 2
    elif abs(sampling_rate - 0.25) < 1e-9: bs, spb = 2, 1
    elif abs(sampling_rate - 0.125) < 1e-9: bs, spb = 4, 2
    elif abs(sampling_rate - 0.0625) < 1e-9: bs, spb = 4, 1
    elif abs(sampling_rate - 0.03125) < 1e-9: bs, spb = 8, 2
    else:
        for y in range(H):
            for x in range(W):
                if random.random() < sampling_rate:
                    sampled.append(img.getpixel((x, y)))
        return sampled

    for by in range(0, H, bs):
        for bx in range(0, W, bs):
            coords = [(x, y)
                      for y in range(by, min(by+bs, H))
                      for x in range(bx, min(bx+bs, W))]
            if not coords: continue
            for (x, y) in random.sample(coords, min(spb, len(coords))):
                sampled.append(img.getpixel((x, y)))
    return sampled

# ----------------- DUPLICATE CHECK -----------------

def find_near_duplicates(palette, thresh):
    pairs = []
    for i in range(len(palette)):
        for j in range(i+1, len(palette)):
            d = euclid(palette[i], palette[j])
            pairs.append((d, i, j))
    pairs.sort(key=lambda x: x[0])
    near = [p for p in pairs if p[0] < thresh]
    return pairs, near

# ----------------- ENHANCEMENT: RESEEDING + DEDUP -----------------

def farthest_pixel_reseed(sampled, palette):
    """Pick the sampled pixel farthest from its nearest palette color."""
    pal, norms, len2s = _prep_wulin(palette)
    best_p = sampled[0]
    best_d = -1.0
    for (r, g, b) in sampled:
        _, d = wu_lin_nearest_color(r, g, b, pal, norms, len2s)
        if d > best_d:
            best_d = d
            best_p = (r, g, b)
    return best_p

def enforce_dedup(sampled, palette, cluster_sizes, dup_thresh, passes=2):
    """If near-duplicates exist, keep the centroid with bigger cluster, reseed the other."""
    pal = list(palette)
    for _ in range(passes):
        _, near = find_near_duplicates(pal, dup_thresh)
        if not near:
            break

        # Process closest pairs first
        for d, i, j in near:
            if d >= dup_thresh:
                continue

            # Keep the one with larger cluster support
            keep = i if cluster_sizes[i] >= cluster_sizes[j] else j
            drop = j if keep == i else i

            # Reseed dropped centroid to something far from current palette
            new_color = farthest_pixel_reseed(sampled, pal)
            pal[drop] = new_color

            # After reseeding, that cluster size is unknown; set small so it can be reseeded again if needed
            cluster_sizes[drop] = 0

        # re-check in next pass
    return pal

# ----------------- STAGE 2 (ENHANCED refinement) -----------------

def fast_kmeans_refine_enhanced(sampled, init_palette, max_iter=10):
    if not sampled:
        return init_palette

    palette = [tuple(c) for c in init_palette]
    Kp = len(palette)
    prev_mse = None

    for it in range(max_iter):
        pal, norms, len2s = _prep_wulin(palette)

        clusters = [[] for _ in range(Kp)]
        mse = 0.0

        # Assignment
        for (r, g, b) in sampled:
            k_idx, dist = wu_lin_nearest_color(r, g, b, pal, norms, len2s)
            clusters[k_idx].append((r, g, b))
            mse += dist
        mse /= len(sampled)

        # Stop condition (same style as your base)
        if prev_mse is not None and mse >= prev_mse:
            break
        prev_mse = mse

        # Update (mean)
        new_palette = list(palette)
        cluster_sizes = [len(c) for c in clusters]

        for k in range(Kp):
            if clusters[k]:
                sr = sum(p[0] for p in clusters[k]) / len(clusters[k])
                sg = sum(p[1] for p in clusters[k]) / len(clusters[k])
                sb = sum(p[2] for p in clusters[k]) / len(clusters[k])
                new_palette[k] = (int(sr), int(sg), int(sb))

        # Enhancement 1: reseed empty/weak clusters (prevents collapse)
        for k in range(Kp):
            if cluster_sizes[k] < MIN_CLUSTER_SIZE:
                new_palette[k] = farthest_pixel_reseed(sampled, new_palette)
                cluster_sizes[k] = 0

        # Enhancement 2: enforce de-duplication (merge + reseed freed slot)
        new_palette = enforce_dedup(sampled, new_palette, cluster_sizes, DUP_THRESH, passes=DEDUP_PASSES)

        palette = new_palette

    return palette

# ----------------- VIS -----------------

def show_palette(palette, title):
    sw = Image.new("RGB", (50*len(palette), 50))
    for i, c in enumerate(palette):
        for x in range(i*50, (i+1)*50):
            for y in range(50):
                sw.putpixel((x, y), c)
    plt.figure(figsize=(10, 2))
    plt.title(title)
    plt.imshow(sw)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def quantize_preview(img, palette, downscale=2):
    if downscale > 1:
        W, H = img.size
        src = img.resize((max(1, W//downscale), max(1, H//downscale)))
    else:
        src = img.copy()

    pal, norms, len2s = _prep_wulin(palette)
    W, H = src.size
    out = Image.new("RGB", (W, H))
    for y in range(H):
        for x in range(W):
            r, g, b = src.getpixel((x, y))
            k_idx, _ = wu_lin_nearest_color(r, g, b, pal, norms, len2s)
            out.putpixel((x, y), tuple(pal[k_idx]))
    return out

# ----------------- MAIN -----------------

if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    img = Image.open(IMAGE_PATH).convert("RGB")

    initc, initn = build_rgb_cubes(img, CUBE_BINS, COUNT_THRESHOLD)
    stage1 = initial_palette_generation(initc, initn, K)

    sampled = block_sample_pixels(img, SAMPLE_RATE)

    # Baseline (your original Stage 2)
    def fast_kmeans_refine_baseline(sampled, init_palette, max_iter=10):
        if not sampled: return init_palette
        palette = [tuple(c) for c in init_palette]
        prev_mse = None
        for _ in range(max_iter):
            pal, norms, len2s = _prep_wulin(palette)
            clusters = [[] for _ in range(len(palette))]
            mse = 0.0
            for (r, g, b) in sampled:
                k_idx, dist = wu_lin_nearest_color(r, g, b, pal, norms, len2s)
                clusters[k_idx].append((r, g, b))
                mse += dist
            mse /= len(sampled)
            if prev_mse is not None and mse >= prev_mse:
                break
            prev_mse = mse

            new_palette = list(palette)
            for k in range(len(palette)):
                if clusters[k]:
                    sr = sum(p[0] for p in clusters[k]) / len(clusters[k])
                    sg = sum(p[1] for p in clusters[k]) / len(clusters[k])
                    sb = sum(p[2] for p in clusters[k]) / len(clusters[k])
                    new_palette[k] = (int(sr), int(sg), int(sb))
            palette = new_palette
        return palette

    final_baseline = fast_kmeans_refine_baseline(sampled, stage1, MAX_ITER)
    final_enhanced = fast_kmeans_refine_enhanced(sampled, stage1, MAX_ITER)

    # Reports
    print("\n--- BASELINE ---")
    pairs, near = find_near_duplicates(final_baseline, DUP_THRESH)
    for d, i, j in pairs[:10]:
        print(f"  ({i},{j}) dist={d:.2f}  {final_baseline[i]} vs {final_baseline[j]}")
    print(f"Near-duplicates (<{DUP_THRESH}): {len(near)}")

    print("\n--- ENHANCED ---")
    pairs2, near2 = find_near_duplicates(final_enhanced, DUP_THRESH)
    for d, i, j in pairs2[:10]:
        print(f"  ({i},{j}) dist={d:.2f}  {final_enhanced[i]} vs {final_enhanced[j]}")
    print(f"Near-duplicates (<{DUP_THRESH}): {len(near2)}")

    # Visuals
    if SHOW_BEFORE_AFTER:
        show_palette(final_baseline, f"Baseline Final Palette (K={K})")
        show_palette(final_enhanced, f"Enhanced Final Palette (Dedup+Reseed) (K={K})")

    if SHOW_QUANTIZED:
        qb = quantize_preview(img, final_baseline, downscale=QUANTIZE_DOWNSCALE)
        qe = quantize_preview(img, final_enhanced, downscale=QUANTIZE_DOWNSCALE)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1); plt.title("Baseline Quantized"); plt.imshow(qb); plt.axis("off")
        plt.subplot(1, 2, 2); plt.title("Enhanced Quantized"); plt.imshow(qe); plt.axis("off")
        plt.tight_layout()
        plt.show()
