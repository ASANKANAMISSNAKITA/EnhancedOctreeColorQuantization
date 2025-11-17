# ===============================================
# Baseline Octree Color Quantization (Original)
# Reference: https://arxiv.org/pdf/1412.1945
# Contains: STORE COLOR, MERGE TREES, CHECK COLOR
# ===============================================

# ---------- Octree Node Definition ----------
class OctreeNode:
    def __init__(self):
        self.child = [None] * 8  # Each node has 8 children

# ---------- Bit Helper Function ----------
def get_bit(value, bit_index):
    """Return the bit at position bit_index of value (0-255)."""
    return (value >> bit_index) & 1

# ===============================================
# Algorithm 1: STORE COLOR
# ===============================================
def store_color(color, root, num_levels):
    """
    Insert a color into the octree.
    color: (R, G, B) tuple 0-255
    root: OctreeNode (can be None)
    num_levels: depth of octree (typically 8)
    """
    if root is None:
        root = OctreeNode()

    temp = root
    R, G, B = color

    # Traverse from MSB down to num_levels
    for i in reversed(range(8 - num_levels, 8)):
        r_bit = get_bit(R, i)
        g_bit = get_bit(G, i)
        b_bit = get_bit(B, i)

        # Index = binary combination of R,G,B bits
        index = (r_bit << 2) | (g_bit << 1) | b_bit

        if temp.child[index] is None:
            temp.child[index] = OctreeNode()

        temp = temp.child[index]

    return root

# ===============================================
# Algorithm 2: MERGE TREES
# ===============================================
def merge_trees(roots, threshold, num_trees, num_levels):
    """
    Merge multiple octrees into a single octree based on threshold.
    roots: list of OctreeNode roots
    threshold: fraction of trees that must have a node (0-1)
    num_trees: number of octrees
    num_levels: depth of octree
    """
    if not roots or all(r is None for r in roots):
        return None

    result = OctreeNode()

    # Step 1: check each child index
    for i in range(8):
        count = 0
        for j in range(num_trees):
            if roots[j] is not None and roots[j].child[i] is not None:
                count += 1
        if count / num_trees >= threshold:
            result.child[i] = OctreeNode()

    # Step 2: recurse for children
    for i in range(8):
        if result.child[i] is not None:
            temp_roots = []
            for j in range(num_trees):
                if roots[j] is None:
                    temp_roots.append(None)
                else:
                    temp_roots.append(roots[j].child[i])
            result.child[i] = merge_trees(temp_roots, threshold, num_trees, num_levels)

    return result

# ===============================================
# Algorithm 3: CHECK COLOR
# ===============================================
def check_color(color, root, num_levels):
    """
    Check if a color exists in the octree.
    Returns True if present, False otherwise.
    """
    if root is None:
        return False

    temp = root
    R, G, B = color

    for i in reversed(range(8 - num_levels, 8)):
        r_bit = get_bit(R, i)
        g_bit = get_bit(G, i)
        b_bit = get_bit(B, i)

        index = (r_bit << 2) | (g_bit << 1) | b_bit

        if temp.child[index] is None:
            return False

        temp = temp.child[index]

    return True

# ===============================================
# Example Usage
# ===============================================
if __name__ == "__main__":
    num_levels = 8  # standard 8-level octree
    root1 = None
    root2 = None

    # Store colors in two octrees
    root1 = store_color((255, 0, 0), root1, num_levels)
    root1 = store_color((0, 255, 0), root1, num_levels)
    root2 = store_color((0, 255, 0), root2, num_levels)
    root2 = store_color((0, 0, 255), root2, num_levels)

    # Merge octrees with threshold 0.5
    merged_root = merge_trees([root1, root2], threshold=0.5, num_trees=2, num_levels=num_levels)

    # Check colors
    print("Red exists in merged tree:", check_color((255, 0, 0), merged_root, num_levels))  # True
    print("Green exists in merged tree:", check_color((0, 255, 0), merged_root, num_levels))  # True
    print("Blue exists in merged tree:", check_color((0, 0, 255), merged_root, num_levels))  # False (below threshold)
