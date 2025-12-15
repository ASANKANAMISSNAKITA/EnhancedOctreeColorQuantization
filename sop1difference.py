# ============================================================
# Demo: RGB vs CIELAB – same test on 4 strips
# Rows 1–2: equal steps in RGB (dark & light)
# Rows 3–4: equal steps in Lab (dark & light ranges)
# For every row we print:
#   - adjacent RGB squared distances
#   - adjacent CIELAB squared distances
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb   # pip install scikit-image

N_STEPS = 11

# ---------------------- RGB STRIPS --------------------------

# Same RGB increment for each step
step_rgb = np.array([20, 0, 0], dtype=np.int32)

start_dark  = np.array([20, 20, 20], dtype=np.int32)
start_light = np.array([200, 200, 200], dtype=np.int32)

def build_rgb_strip(start_color, step):
    colors = []
    for i in range(N_STEPS):
        c = start_color + i * step
        c = np.clip(c, 0, 255)
        colors.append(c)
    return np.array(colors, dtype=np.uint8).reshape(1, N_STEPS, 3)

strip_rgb_dark  = build_rgb_strip(start_dark,  step_rgb)
strip_rgb_light = build_rgb_strip(start_light, step_rgb)

# ---------------------- LAB STRIPS --------------------------
# Equal steps in Lab: only L* changes, a*, b* fixed
# ➜ start Lab colors are chosen so that the first tile
#    of row 3 matches row 1, and row 4 matches row 2.

def rgb_to_lab_single(rgb_uint8):
    """Convert a single RGB uint8 color [R,G,B] to Lab [L*,a*,b*]."""
    rgb = np.array(rgb_uint8, dtype=np.float32) / 255.0
    lab = rgb2lab(rgb.reshape(1, 1, 3)).reshape(3,)
    return lab.astype(np.float32)

start_lab_dark  = rgb_to_lab_single(start_dark)   # aligned with Row 1 start
start_lab_light = rgb_to_lab_single(start_light)  # aligned with Row 2 start

# Same perceptual step in Lab (ΔE ≈ 6, only L* changes)
step_lab = np.array([6.0, 0.0, 0.0], dtype=np.float32)

def build_lab_strip(start_lab_color, lab_step):
    labs = []
    for i in range(N_STEPS):
        L_a_b = start_lab_color + i * lab_step
        L_a_b[0] = np.clip(L_a_b[0], 0.0, 100.0)  # keep L* valid
        labs.append(L_a_b.copy())

    labs = np.array(labs, dtype=np.float32)          # (N_STEPS, 3)
    labs_img = labs.reshape(1, N_STEPS, 3)

    rgb_float = lab2rgb(labs_img)                    # 0..1
    rgb_uint8 = np.clip(rgb_float * 255, 0, 255).astype(np.uint8)

    return rgb_uint8, labs   # strip in RGB + Lab coords

strip_lab_rgb_dark,  labs_dark  = build_lab_strip(start_lab_dark,  step_lab)
strip_lab_rgb_light, labs_light = build_lab_strip(start_lab_light, step_lab)

# ---------------------- DISTANCE HELPERS --------------------

def sq_euclid_rgb(c1, c2):
    d = c1.astype(np.int32) - c2.astype(np.int32)
    return int(np.dot(d, d))

def sq_euclid_lab(l1, l2):
    d = l1 - l2
    return float(np.dot(d, d))

def rgb_strip_to_lab_coords(strip_rgb):
    """Convert a 1 x N x 3 uint8 strip to N x 3 Lab coordinates."""
    rgb_float = strip_rgb.astype(np.float32) / 255.0
    lab_img   = rgb2lab(rgb_float)          # shape (1, N, 3)
    return lab_img.reshape(N_STEPS, 3)

# ---------------------- RUN THE TESTS -----------------------

def print_distances(name, strip_rgb, lab_coords):
    print(f"\n{name}")
    print("  Adjacent RGB squared distances:")
    for i in range(N_STEPS - 1):
        d = sq_euclid_rgb(strip_rgb[0, i], strip_rgb[0, i+1])
        print(f"    step {i} → {i+1}: {d}")

    print("  Adjacent CIELAB squared distances:")
    for i in range(N_STEPS - 1):
        d = sq_euclid_lab(lab_coords[i], lab_coords[i+1])
        print(f"    step {i} → {i+1}: {d:.2f}")

# For RGB-based rows, Lab coords come from converting the RGB strip
lab_dark_from_rgb  = rgb_strip_to_lab_coords(strip_rgb_dark)
lab_light_from_rgb = rgb_strip_to_lab_coords(strip_rgb_light)

print_distances("Row 1: RGB steps (dark strip)",  strip_rgb_dark,  lab_dark_from_rgb)
print_distances("Row 2: RGB steps (light strip)", strip_rgb_light, lab_light_from_rgb)

# For Lab-based rows, Lab coords are the ones we used to construct them
print_distances("Row 3: Lab steps (dark range)",  strip_lab_rgb_dark,  labs_dark)
print_distances("Row 4: Lab steps (light range)", strip_lab_rgb_light, labs_light)

# ---------------------- VISUALIZATION -----------------------

combined = np.vstack([
    strip_rgb_dark,
    strip_rgb_light,
    strip_lab_rgb_dark,
    strip_lab_rgb_light
])

plt.figure(figsize=(8, 4))
plt.imshow(combined)
plt.axis("off")
plt.title(
    "Rows 1–2: equal RGB steps (dark & light)\n"
    "Rows 3–4: equal CIELAB steps (dark & light), same starting colors"
)
plt.show()
