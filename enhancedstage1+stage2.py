# ======================================================
# RGB vs CIELAB Fast K-Means Palette Generation
#  - Stage 1: Huang-style RGB cube initial palette
#  - Stage 2a: Fast K-Means in RGB (baseline)
#  - Stage 2b: Fast K-Means in CIELAB (enhanced)
#  - Extra: 3D RGB cube + 3D CIELAB sphere-style plots
# ======================================================

from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import random
import math
import numpy as np
from skimage.color import rgb2lab, lab2rgb  # pip install scikit-image

# ----------------- PARAMETERS -----------------

IMAGE_PATH = "flower.jpg"   # <-- your image
K = 7                       # desired number of palette colors
CUBE_BINS = 16              # RGB cube division per axis
COUNT_THRESHOLD = 1         # Thr: minimum pixel count for a cube to be considered
SAMPLE_RATE = 0.1           # fraction of pixels used for K-Means refinement
MAX_ITER = 10               # Max_cycle in the paper

# ----------------- HELPER FUNCTIONS -----------------

def rgb_to_cube_index(r, g, b, bins):
    """Map an RGB color to a cube index in a bins x bins x bins grid."""
    rb = (r * bins) // 256
    gb = (g * bins) // 256
    bb = (b * bins) // 256
    return (rb, gb, bb)

def squared_euclidean(c1, c2):
    """Squared Euclidean distance SED(c1, c2) in 3D."""
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return dr*dr + dg*dg + db*db

# ----------------- STAGE 1: INITIAL PALETTE (RGB CUBE) -----------------

def build_rgb_cubes(img, bins, count_threshold, sample_rate):
    """
    Build initc(i) and initn(i) using an RGB cube.
    Also collect sampled pixels SCP_i for Stage 2.
    """
    width, height = img.size
    cube_stats = {}   # key: (rb,gb,bb), value: dict(count, sum_r, sum_g, sum_b)
    sampled_pixels = []

    print("Scanning image and building RGB cubes...")

    for y in tqdm(range(height), desc="Rows processed"):
        for x in range(width):
            r, g, b = img.getpixel((x, y))

            # --- update cube stats ---
            cube_idx = rgb_to_cube_index(r, g, b, bins)
            if cube_stats.get(cube_idx) is None:
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

            # --- sample pixels for Stage 2 (SCP_i) ---
            if random.random() < sample_rate:
                sampled_pixels.append((r, g, b))

    # build initc(i), initn(i) from cubes that pass threshold
    initc = []  # candidate colors
    initn = []  # their frequencies

    for stats in cube_stats.values():
        if stats["count"] >= count_threshold:
            c_count = stats["count"]
            mean_r = stats["sum_r"] // c_count
            mean_g = stats["sum_g"] // c_count
            mean_b = stats["sum_b"] // c_count
            initc.append((mean_r, mean_g, mean_b))
            initn.append(c_count)

    print(f"Number of candidate colors (initc): {len(initc)}")
    print(f"Number of sampled pixels for Stage 2 (SPN): {len(sampled_pixels)}")

    return initc, initn, sampled_pixels

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

# ----------------- STAGE 2A: FAST K-MEANS IN RGB (BASELINE) -----------------

def fast_kmeans_palette_refinement_rgb(sampled_pixels, initial_palette, max_iter=10):
    """
    Stage 2: Fast K-Means Algorithm in RGB (baseline).
    Returns final palette and MSE history (RGB SED).
    """
    if not sampled_pixels:
        print("No sampled pixels, skipping Stage 2 refinement.")
        return initial_palette, []

    palette = [list(c) for c in initial_palette]
    K = len(palette)
    SPN = len(sampled_pixels)

    Iter = 0
    StopF = 0
    prev_mse = None
    mse_history = []

    while Iter < max_iter and not StopF:
        clusters = [[] for _ in range(K)]
        mse_accum = 0.0

        for (r, g, b) in sampled_pixels:
            dists = [squared_euclidean((r, g, b), (p[0], p[1], p[2])) for p in palette]
            k_idx = min(range(K), key=lambda idx: dists[idx])
            clusters[k_idx].append((r, g, b))
            mse_accum += dists[k_idx]

        MSE1_iter = mse_accum / SPN
        mse_history.append(MSE1_iter)
        print(f"[RGB] Iteration {Iter}: MSE1({Iter}) = {MSE1_iter:.2f}")

        if Iter > 0 and MSE1_iter >= prev_mse:
            StopF = 1
        prev_mse = MSE1_iter

        # recompute means
        for k in range(K):
            if clusters[k]:
                sr = sum(p[0] for p in clusters[k]) / len(clusters[k])
                sg = sum(p[1] for p in clusters[k]) / len(clusters[k])
                sb = sum(p[2] for p in clusters[k]) / len(clusters[k])
                palette[k] = [int(sr), int(sg), int(sb)]

        palette.sort(key=lambda c: c[0]**2 + c[1]**2 + c[2]**2)
        Iter += 1

    refined_palette = [tuple(c) for c in palette]
    print(f"[RGB] Final palette generated after {Iter} iterations.")
    return refined_palette, mse_history

# ----------------- STAGE 2B: FAST K-MEANS IN CIELAB (ENHANCED) -----------------

def rgb_list_to_lab_array(pixels):
    """
    Convert list of (R,G,B) 0-255 to Nx3 Lab array (float).
    """
    arr = np.array(pixels, dtype=np.float32) / 255.0
    lab = rgb2lab(arr.reshape(-1, 1, 3)).reshape(-1, 3)
    return lab

def lab_array_to_rgb_palette(centroids_lab):
    """
    Convert Nx3 Lab centroids to list of (R,G,B) 0-255 ints.
    """
    rgb = lab2rgb(centroids_lab.reshape(-1, 1, 3)).reshape(-1, 3)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return [tuple(int(v) for v in row) for row in rgb]

def fast_kmeans_palette_refinement_lab(sampled_pixels, initial_palette, max_iter=10):
    """
    Stage 2: Fast K-Means Algorithm in CIELAB (enhanced).
    Uses Lab space for distance and MSE.
    """
    if not sampled_pixels:
        print("No sampled pixels, skipping Stage 2 refinement (Lab).")
        return initial_palette, []

    # convert sampled pixels and initial palette to Lab
    pixels_lab = rgb_list_to_lab_array(sampled_pixels)
    init_lab = rgb_list_to_lab_array(initial_palette)

    K = init_lab.shape[0]
    SPN = pixels_lab.shape[0]

    centroids = init_lab.copy()
    Iter = 0
    StopF = 0
    prev_mse = None
    mse_history = []

    while Iter < max_iter and not StopF:
        clusters = [[] for _ in range(K)]
        mse_accum = 0.0

        for lab_col in pixels_lab:
            dists = np.sum((centroids - lab_col) ** 2, axis=1)
            k_idx = int(np.argmin(dists))
            clusters[k_idx].append(lab_col)
            mse_accum += float(dists[k_idx])

        mse_iter = mse_accum / SPN
        mse_history.append(mse_iter)
        print(f"[Lab] Iteration {Iter}: MSE_Lab({Iter}) = {mse_iter:.2f}")

        if Iter > 0 and mse_iter >= prev_mse:
            StopF = 1
        prev_mse = mse_iter

        # recompute centroids in Lab
        for k in range(K):
            if clusters[k]:
                centroids[k] = np.mean(clusters[k], axis=0)

        Iter += 1

    refined_palette_rgb = lab_array_to_rgb_palette(centroids)
    print(f"[Lab] Final palette generated after {Iter} iterations.")
    return refined_palette_rgb, mse_history

# ----------------- VISUALIZATION HELPERS -----------------

def make_swatch_image(palette, swatch_size=50, height=50):
    """Return a NumPy image showing the palette as horizontal blocks."""
    if not palette:
        return None
    img = Image.new("RGB", (swatch_size * len(palette), height))
    for i, color in enumerate(palette):
        for x in range(i * swatch_size, (i + 1) * swatch_size):
            for y in range(height):
                img.putpixel((x, y), color)
    return np.array(img)

def plot_rgb_cube(sampled_pixels, palette, max_points=500):
    """
    Plot sampled pixels in RGB cube and highlight palette colors as 'X'.
    (No plt.show() here – called once at the end.)
    """
    print("Plotting RGB cube with palette colors...")

    if len(sampled_pixels) > max_points:
        plot_points = random.sample(sampled_pixels, max_points)
    else:
        plot_points = sampled_pixels

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    ax.set_title('RGB Cube - Sampled Colors and Palette')

    for (r, g, b) in plot_points:
        ax.scatter(r, g, b, color=(r/255, g/255, b/255), s=3)

    for color in palette:
        r, g, b = color
        ax.scatter(r, g, b, color=(r/255, g/255, b/255),
                   s=80, marker='X', edgecolors='black', linewidths=1.5)

    print("RGB cube ready (will show on global plt.show()).")

def plot_lab_sphere(sampled_pixels, palette, max_points=500):
    """
    Lab 'sphere' visualization similar to the classic CIELAB diagram:
      - z-axis: L* (0 = black, 100 = white)
      - x-axis: a* (green- to red+)
      - y-axis: b* (blue- to yellow+)
    Shows:
      - Sampled pixels as small points
      - Palette colors as large 'X' markers
      - Mid-plane chroma circle and L* axis like the reference diagram
    """
    print("Plotting CIELAB sphere-style diagram with palette colors...")

    # subsample pixels for plotting
    if len(sampled_pixels) > max_points:
        plot_points = random.sample(sampled_pixels, max_points)
    else:
        plot_points = sampled_pixels

    # convert to Lab
    lab_points = rgb_list_to_lab_array(plot_points)   # Nx3: [L, a, b]
    lab_palette = rgb_list_to_lab_array(palette)      # Kx3

    L_vals = lab_points[:, 0]
    a_vals = lab_points[:, 1]
    b_vals = lab_points[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # axes labels like the diagram
    ax.set_xlabel("a*  (green  −  red)")
    ax.set_ylabel("b*  (blue   −  yellow)")
    ax.set_zlabel("L*  (lightness)")

    ax.set_xlim(-128, 128)
    ax.set_ylim(-128, 128)
    ax.set_zlim(0, 100)

    ax.set_title("CIELAB Space – Sampled Colors and Palette")

    # mid-plane chroma circle at L* = 50
    radius = 80
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_a = radius * np.cos(theta)
    circle_b = radius * np.sin(theta)
    circle_L = np.full_like(theta, 50.0)
    ax.plot(circle_a, circle_b, circle_L, linestyle="--", linewidth=1)

    # vertical L* axis at a*=0, b*=0
    ax.plot([0, 0], [0, 0], [0, 100], color="black", linewidth=2)
    ax.text(0, 0, 102, "L* = 100 (White)", ha="center")
    ax.text(0, 0, -4, "L* = 0 (Black)", ha="center")

    # a* and b* arrows in mid-plane
    arrow_L = 50
    arrow_len = 90

    # +a (red)
    ax.quiver(0, 0, arrow_L,
              arrow_len, 0, 0,
              length=1, normalize=False, arrow_length_ratio=0.08)
    ax.text(arrow_len + 5, 0, arrow_L, "+a (Red)", color="red")

    # -a (green)
    ax.quiver(0, 0, arrow_L,
              -arrow_len, 0, 0,
              length=1, normalize=False, arrow_length_ratio=0.08)
    ax.text(-arrow_len - 20, 0, arrow_L, "-a (Green)", color="green")

    # +b (yellow)
    ax.quiver(0, 0, arrow_L,
              0, arrow_len, 0,
              length=1, normalize=False, arrow_length_ratio=0.08)
    ax.text(0, arrow_len + 5, arrow_L, "+b (Yellow)", color="goldenrod")

    # -b (blue)
    ax.quiver(0, 0, arrow_L,
              0, -arrow_len, 0,
              length=1, normalize=False, arrow_length_ratio=0.08)
    ax.text(0, -arrow_len - 15, arrow_L, "-b (Blue)", color="royalblue")

    # scatter sampled pixels (colored using original RGB)
    for (rgb, L, a, b) in zip(plot_points, L_vals, a_vals, b_vals):
        r, g, b_rgb = rgb
        ax.scatter(a, b, L,
                   color=(r / 255, g / 255, b_rgb / 255),
                   s=5, alpha=0.7)

    # highlight palette colors as big X
    for lab_col, rgb in zip(lab_palette, palette):
        L_p, a_p, b_p = lab_col   # [L, a, b]
        r, g, b_rgb = rgb
        ax.scatter(a_p, b_p, L_p,
                   color=(r / 255, g / 255, b_rgb / 255),
                   s=120, marker="X", edgecolors="black", linewidths=1.5)

    print("CIELAB sphere ready (will show on global plt.show()).")

# ----------------- MAIN -----------------

if __name__ == "__main__":
    # load image
    img = Image.open(IMAGE_PATH).convert("RGB")

    # Stage 1: build RGB cubes and generate initial palette
    initc, initn, sampled_pixels = build_rgb_cubes(
        img,
        bins=CUBE_BINS,
        count_threshold=COUNT_THRESHOLD,
        sample_rate=SAMPLE_RATE
    )

    initial_palette = initial_palette_generation(initc, initn, K=K)

    # Stage 2a: Fast K-Means in RGB
    final_palette_rgb, mse_rgb = fast_kmeans_palette_refinement_rgb(
        sampled_pixels,
        initial_palette,
        max_iter=MAX_ITER
    )

    # Stage 2b: Fast K-Means in CIELAB
    final_palette_lab, mse_lab = fast_kmeans_palette_refinement_lab(
        sampled_pixels,
        initial_palette,
        max_iter=MAX_ITER
    )

    # --- Build swatch images ---
    swatch_rgb = make_swatch_image(final_palette_rgb)
    swatch_lab = make_swatch_image(final_palette_lab)

    # --- Plot palettes + MSE curves ---
    fig, axes = plt.subplots(3, 1, figsize=(7, 7))

    if swatch_rgb is not None:
        axes[0].imshow(swatch_rgb)
    axes[0].axis("off")
    axes[0].set_title("Final Palette - Fast K-Means in RGB")

    if swatch_lab is not None:
        axes[1].imshow(swatch_lab)
    axes[1].axis("off")
    axes[1].set_title("Final Palette - Fast K-Means in CIELAB")

    axes[2].plot(mse_rgb, label="RGB space")
    axes[2].plot(mse_lab, label="CIELAB space")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("MSE (space-specific)")
    axes[2].set_title("MSE per Iteration: RGB vs CIELAB")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()

    # --- 3D cubes: RGB and CIELAB-sphere ---
    plot_rgb_cube(sampled_pixels, final_palette_rgb)
    plot_lab_sphere(sampled_pixels, final_palette_lab)

    # Single global show for all figures
    plt.show()
