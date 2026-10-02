import numpy as np


def solve_smallest_coloring(
    points: np.ndarray, s: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies the smallest coloring of the point set P given float s > 1.

    Args:
        points: Array-like of shape (N, 2) containing the 2D coordinates.
        s: Float strictly greater than 1 representing the separation factor.

    Returns:
        trivial_colors: 1D numpy array of the absolute minimal coloring (1 color).
        nontrivial_colors: 1D numpy array of the optimal non-trivial partitioning.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)

    if n == 0:
        return np.array([]), np.array([])
    if n == 1:
        return np.array([1]), np.array([1])

    # 1. Vectorized Distance Matrix Calculation
    # points[:, None, :] shapes to (N, 1, 2), points[None, :, :] shapes to (1, N, 2)
    # diff shapes to (N, N, 2), norm computes Euclidean distance over the last axis.
    diff = points[:, None, :] - points[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    # Fill diagonal with 0.0 to ensure self-distances don't affect diameters
    np.fill_diagonal(dist_matrix, 0.0)

    # Extract upper triangle indices to form edges
    edges_i, edges_j = np.triu_indices(n, k=1)
    edges_w = dist_matrix[edges_i, edges_j]

    # 2. Build the Hierarchical Tree (Single-Linkage/Kruskal via DSU)
    sort_idx = np.argsort(edges_w)

    parent = np.arange(n)

    def find(i):
        # Iterative path-compression find
        root = i
        while root != parent[root]:
            root = parent[root]
        curr = i
        while curr != root:
            nxt = parent[curr]
            parent[curr] = root
            curr = nxt
        return root

    # Tree structures mapped by node ID
    node_points = {i: [i] for i in range(n)}
    node_children = {i: [] for i in range(n)}
    active_node = {i: i for i in range(n)}

    next_node_id = n

    for idx in sort_idx:
        u, v = edges_i[idx], edges_j[idx]
        ru, rv = find(u), find(v)

        if ru != rv:
            parent[ru] = rv

            new_id = next_node_id
            next_node_id += 1

            left_node, right_node = active_node[ru], active_node[rv]
            node_children[new_id] = [left_node, right_node]
            node_points[new_id] = node_points[left_node] + node_points[right_node]

            active_node[rv] = new_id

    # 3. Dynamic Programming (Validation via Numpy Array Masking)
    is_valid = np.zeros(next_node_id, dtype=bool)
    dp = np.zeros(next_node_id, dtype=int)

    # Base cases: individual points are always valid valid 1-color blocks
    is_valid[:n] = True
    dp[:n] = 1

    # Topological DP: Since parent IDs are always > child IDs, a simple loop processes bottom-up
    for node in range(n, next_node_id):
        pts = node_points[node]
        pts_arr = np.array(pts)

        # Diameter: Max distance inside the cluster
        diam = np.max(dist_matrix[pts_arr][:, pts_arr])

        # Separation: Min distance to any point outside the cluster
        if len(pts) == n:
            sep = np.inf
        else:
            mask = np.zeros(n, dtype=bool)
            mask[pts] = True
            # Slice only rows inside the cluster and columns outside
            sep = np.min(dist_matrix[pts_arr][:, ~mask])

        # Logic check: 'sep > 0' prevents arbitrarily splitting perfectly superimposed identical points
        valid = (sep >= s * diam) and (sep > 0 or len(pts) == n)
        is_valid[node] = valid

        child_sum = sum(dp[child] for child in node_children[node])
        dp[node] = 1 if valid else child_sum

    # 4. Color Assignment (Top-Down)
    colors = np.zeros(n, dtype=int)

    def assign_colors(node, color_offset, force_split=False):
        # If the block is valid and we aren't forcing it to split, color the whole block
        if is_valid[node] and not force_split:
            colors[node_points[node]] = color_offset
            return color_offset + 1

        # Otherwise, recursively color the children
        for child in node_children[node]:
            color_offset = assign_colors(child, color_offset, False)
        return color_offset

    root_node = next_node_id - 1

    # Trivial coloring: Mathematical minimum (always k=1)
    assign_colors(root_node, 1, force_split=False)
    trivial_colors = colors.copy()

    # Non-trivial coloring: Force the root to split into meaningful sub-components (k>=2)
    assign_colors(root_node, 1, force_split=True)
    nontrivial_colors = colors.copy()

    return trivial_colors, nontrivial_colors


print(
    solve_smallest_coloring(
        np.array(
            [[0, 0], [1, 0], [0, 1], [1, 1], [5, 5], [6, 5], [5, 6], [6, 6]],
            dtype=np.float16,
        ),
        s=2.0,
    )
)

print(
    solve_smallest_coloring(
        np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=np.float16,
        ),
        s=2.0,
    )
)
