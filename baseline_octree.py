# ===============================================
# 3D RGB Cube Visualization with Octree & Progress
# ===============================================
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm  # progress bar
import random

# ---------- Octree Node ----------
class OctreeNode:
    def __init__(self):
        self.child = [None] * 8
        self.color_count = 0
        self.color_sum = [0, 0, 0]

def get_bit(value, bit_index):
    return (value >> bit_index) & 1

def store_color(color, root, num_levels):
    if root is None:
        root = OctreeNode()
    temp = root
    R, G, B = color
    for i in reversed(range(8 - num_levels, 8)):
        r_bit = get_bit(R, i)
        g_bit = get_bit(G, i)
        b_bit = get_bit(B, i)
        index = (r_bit << 2) | (g_bit << 1) | b_bit
        if temp.child[index] is None:
            temp.child[index] = OctreeNode()
        temp = temp.child[index]
    temp.color_count += 1
    temp.color_sum[0] += R
    temp.color_sum[1] += G
    temp.color_sum[2] += B
    return root

def get_all_colors(node):
    if node is None:
        return []
    if node.color_count > 0:
        avg_color = tuple(int(c / node.color_count) for c in node.color_sum)
        return [avg_color]
    palette = []
    for child in node.child:
        palette.extend(get_all_colors(child))
    return palette

def get_palette(colors, n=7):
    colors_sorted = sorted(colors, key=lambda c: sum(c), reverse=True)
    return colors_sorted[:n]

# ---------- MAIN ----------
if __name__ == "__main__":
    image_path = "Mona_Lisa.jpg"  # <-- your image
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    num_levels = 8
    root = None

    print("Building octree...")

    # ---------- Progress bar for inserting pixels ----------
    for y in tqdm(range(height), desc="Rows processed"):
        for x in range(width):
            color = img.getpixel((x, y))
            root = store_color(color, root, num_levels)

    print("Octree built.")

    # Extract colors
    colors = get_all_colors(root)
    print(f"Number of leaf colors: {len(colors)}")

    # ---------- Sample colors for fast plotting ----------
    max_points = 5000
    if len(colors) > max_points:
        colors_sample = random.sample(colors, max_points)
    else:
        colors_sample = colors

    palette = get_palette(colors, n=7)

    # ---------- Plot 3D RGB Cube ----------
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    ax.set_title('RGB Cube - Color Distribution')

    # Plot sampled colors
    for color in colors_sample:
        r, g, b = color
        ax.scatter(r, g, b, color=(r/255, g/255, b/255), s=5)

    # Highlight palette colors
    for color in palette:
        r, g, b = color
        ax.scatter(r, g, b, color=(r/255, g/255, b/255), s=50, marker='X', edgecolors='black')

    print("Plotting complete. Palette colors are marked as 'X'.")
    plt.show()
