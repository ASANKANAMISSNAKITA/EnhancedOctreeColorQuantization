# ======================================================
# Stage 1 ONLY – RGB Cube Initial Palette Generation
# (Based on Huang, 2021)
# ======================================================

from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import random
import math

# ----------------- PARAMETERS -----------------

IMAGE_PATH = "flower.jpg"   # <-- your image
K = 7                        # desired number of palette colors
CUBE_BINS = 16               # RGB cube division per axis
COUNT_THRESHOLD = 1          # Thr: minimum pixel count for a cube to be considered

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

def build_rgb_cubes(img, bins, count_threshold):
    """
    Build initc(i) and initn(i) using an RGB cube.

    - Divide RGB space into bins^3 cubes.
    - For each cube, accumulate count and sum of RGB.
    - For cubes with count >= Thr, compute mean color (initc)
      and frequency (initn).
    """
    width, height = img.size
    cube_stats = {}   # key: (rb,gb,bb), value: dict(count, sum_r, sum_g, sum_b)
    all_pixels = []

    print("Scanning image and building RGB cubes (Stage 1)...")

    for y in tqdm(range(height), desc="Rows processed"):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            all_pixels.append((r, g, b))

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
            mean_r = stats["sum_r"] // c_count
            mean_g = stats["sum_g"] // c_count
            mean_b = stats["sum_b"] // c_count
            initc.append((mean_r, mean_g, mean_b))
            initn.append(c_count)

    print(f"Number of candidate colors (initc): {len(initc)}")

    return initc, initn, all_pixels

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

    print("Showing Stage 1 palette swatch...")
    palette_img.show()

def plot_rgb_cube(all_pixels, palette, max_points=5000):
    """
    Plot image pixels in RGB cube and highlight Stage 1 palette colors as 'X'.
    """
    print("Plotting RGB cube with Stage 1 palette colors...")

    if len(all_pixels) > max_points:
        plot_points = random.sample(all_pixels, max_points)
    else:
        plot_points = all_pixels

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    ax.set_title('RGB Cube - Image Colors and Stage 1 Palette')

    # plot image pixels
    for (r, g, b) in plot_points:
        ax.scatter(r, g, b, color=(r/255, g/255, b/255), s=5)

    # highlight palette colors
    for color in palette:
        r, g, b = color
        ax.scatter(r, g, b, color=(r/255, g/255, b/255),
                   s=80, marker='X', edgecolors='black', linewidths=1.5)

    plt.show()
    print("Plotting complete. Stage 1 palette colors are marked as 'X'.")

# ----------------- MAIN -----------------

if __name__ == "__main__":
    # load image
    img = Image.open(IMAGE_PATH).convert("RGB")

    # Stage 1: build RGB cubes and generate initial palette
    initc, initn, all_pixels = build_rgb_cubes(
        img,
        bins=CUBE_BINS,
        count_threshold=COUNT_THRESHOLD
    )

    initial_palette = initial_palette_generation(initc, initn, K=K)

    # Visualize Stage 1 result
    show_palette_swatch(initial_palette)
    plot_rgb_cube(all_pixels, initial_palette)
