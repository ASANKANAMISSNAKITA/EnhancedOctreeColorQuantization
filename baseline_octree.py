# ======================================================
# RGB Cube + Fast K-Means Palette Generation System
# (Existing Algorithm based on Huang: Stage 1 + Stage 2)
# ======================================================

from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import random
import math

# ----------------- PARAMETERS -----------------

IMAGE_PATH = "4.2.03.tiff"   # <-- your image
K = 7                        # desired number of palette colors
CUBE_BINS = 16               # RGB cube division per axis
COUNT_THRESHOLD = 1          # Thr: minimum pixel count for a cube to be considered
SAMPLE_RATE = 0.1            # fraction of pixels used for K-Means refinement
MAX_ITER = 10                # Max_cycle in the paper

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

# ----------------- STAGE 2: FAST K-MEANS PALETTE REFINEMENT -----------------

def fast_kmeans_palette_refinement(sampled_pixels, initial_palette, max_iter=10):
    """
    Stage 2: Fast K-Means Algorithm (existing algorithm)

    Step 1: SCP_i sampling, Iter = 0, StopF = 0.
    Step 2: assign each SCP_i to nearest palette color CCP_i.
    Step 3: compute mean of K groups and sort palette by length.
    Step 4: MSE1(Iter) = (1 / SPN) * sum_i SED(SCP_i, CCP_i).
            If Iter > 0 and MSE1(Iter) >= MSE1(Iter - 1), StopF = 1.
    Step 5: Iter = Iter + 1; if Iter == Max_cycle or StopF == 1, stop.
    Step 6: output final palette and Iter.
    """
    if not sampled_pixels:
        print("No sampled pixels, skipping Stage 2 refinement.")
        return initial_palette

    palette = [list(c) for c in initial_palette]
    K = len(palette)
    SPN = len(sampled_pixels)

    # Step 1
    Iter = 0
    StopF = 0
    prev_mse = None

    while Iter < max_iter and not StopF:
        # Step 2: assignment step
        clusters = [[] for _ in range(K)]
        mse_accum = 0.0

        for (r, g, b) in sampled_pixels:
            dists = [squared_euclidean((r, g, b), (p[0], p[1], p[2])) for p in palette]
            k_idx = min(range(K), key=lambda idx: dists[idx])
            clusters[k_idx].append((r, g, b))
            mse_accum += dists[k_idx]  # SED(SCP_i, CCP_i)

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

        # sort K palette colors in ascending order of length
        palette.sort(key=lambda c: c[0]**2 + c[1]**2 + c[2]**2)

        # Step 5: Iter = Iter + 1
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

def plot_rgb_cube(sampled_pixels, palette, max_points=500):
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

    # Stage 1: build RGB cubes and generate initial palette
    initc, initn, sampled_pixels = build_rgb_cubes(
        img,
        bins=CUBE_BINS,
        count_threshold=COUNT_THRESHOLD,
        sample_rate=SAMPLE_RATE
    )

    initial_palette = initial_palette_generation(initc, initn, K=K)

    # Stage 2: refine palette with fast K-Means on sampled pixels
    final_palette = fast_kmeans_palette_refinement(
        sampled_pixels,
        initial_palette,
        max_iter=MAX_ITER
    )

    # show palette as swatch
    show_palette_swatch(final_palette)

    # show RGB cube with palette points
    plot_rgb_cube(sampled_pixels, final_palette)
