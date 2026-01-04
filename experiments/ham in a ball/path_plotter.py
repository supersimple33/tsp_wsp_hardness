import math
import numpy as np
import matplotlib.pyplot as plt


# ---------- Distance helpers ----------


def dist_matrix(points):
    n = len(points)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dd = math.hypot(xi - xj, yi - yj)
            d[i][j] = d[j][i] = dd
    return d


def diameter(d):
    return max(max(row) for row in d)


# ---------- DP: all endpoint-optimal paths + reconstruction ----------


def all_endpoint_optimal_paths_with_paths(d):
    """
    L[s][t] = length of shortest Hamiltonian path that starts at s, ends at t.
    paths[s][t] = list of vertex indices in that path.
    """
    n = len(d)
    INF = float("inf")
    full_mask = (1 << n) - 1

    L = [[INF] * n for _ in range(n)]
    paths = [[None] * n for _ in range(n)]

    for s in range(n):
        size = 1 << n
        dp = [[INF] * n for _ in range(size)]
        parent = [[-1] * n for _ in range(size)]

        start_mask = 1 << s
        dp[start_mask][s] = 0.0

        for mask in range(size):
            if not (mask & start_mask):
                continue
            for j in range(n):
                if not (mask & (1 << j)):
                    continue
                cost = dp[mask][j]
                if math.isinf(cost):
                    continue
                if mask == full_mask:
                    continue
                for k in range(n):
                    if mask & (1 << k):
                        continue
                    new_mask = mask | (1 << k)
                    new_cost = cost + d[j][k]
                    if new_cost < dp[new_mask][k]:
                        dp[new_mask][k] = new_cost
                        parent[new_mask][k] = j

        # finalize for this start s
        for t in range(n):
            if t == s:
                continue
            if math.isinf(dp[full_mask][t]):
                continue

            L[s][t] = dp[full_mask][t]

            # reconstruct path s -> ... -> t
            path_rev = []
            mask = full_mask
            cur = t
            while cur != -1:
                path_rev.append(cur)
                if cur == s:
                    break
                prev = parent[mask][cur]
                if prev == -1:
                    break
                mask ^= 1 << cur
                cur = prev

            path = list(reversed(path_rev))
            if path[0] != s or path[-1] != t or len(path) != n:
                # Safety check; in theory shouldn't happen
                paths[s][t] = None
            else:
                paths[s][t] = path

    return L, paths


# ---------- Max gap ratio + retrieve witnessing paths ----------


def max_gap_ratio_with_paths(points):
    """
    Returns:
      ratio, diam, max_gap,
      (a, b), (c, e),
      path_ab, path_ce, dist_matrix
    where path_ab / path_ce are lists of indices.
    """
    d = dist_matrix(points)
    diam = diameter(d)
    L, all_paths = all_endpoint_optimal_paths_with_paths(d)
    n = len(points)

    max_gap = 0.0
    best_ab = None
    best_ce = None

    for a in range(n):
        for b in range(n):
            if a == b or math.isinf(L[a][b]):
                continue
            for c in range(n):
                for e in range(n):
                    if c == e or math.isinf(L[c][e]):
                        continue
                    gap = abs(L[a][b] - L[c][e])
                    if gap > max_gap:
                        max_gap = gap
                        best_ab = (a, b)
                        best_ce = (c, e)

    if diam == 0 or best_ab is None or best_ce is None:
        return float("nan"), diam, max_gap, None, None, None, None, d

    a, b = best_ab
    c, e = best_ce
    path_ab = all_paths[a][b]
    path_ce = all_paths[c][e]

    ratio = max_gap / diam if diam > 0 else float("nan")
    return ratio, diam, max_gap, best_ab, best_ce, path_ab, path_ce, d


# ---------- Plotting ----------


def plot_points_and_paths(points, path_ab, path_ce, ab_label=None, ce_label=None):
    """
    points: list of (x, y)
    path_ab, path_ce: lists of vertex indices
    """
    xs, ys = zip(*points)

    plt.figure(figsize=(6, 6))
    # Plot all points
    plt.scatter(xs, ys, s=40, zorder=3)

    # Label points by index
    for idx, (x, y) in enumerate(points):
        plt.text(x + 0.01, y + 0.01, str(idx), fontsize=8)

    def plot_path(path, style, label):
        if path is None:
            return
        xs_p = [points[i][0] for i in path]
        ys_p = [points[i][1] for i in path]
        plt.plot(xs_p, ys_p, style, linewidth=2, label=label, zorder=2)
        # Mark start/end
        plt.scatter(xs_p[0], ys_p[0], s=70, marker="s", zorder=4)
        plt.scatter(xs_p[-1], ys_p[-1], s=70, marker="X", zorder=4)

    plot_path(path_ab, "-", ab_label or "Path AB")
    plot_path(path_ce, "--", ce_label or "Path CE")

    plt.axis("equal")
    plt.title("Points with two extremal Hamiltonian paths")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------- Your base points ----------

base_points = [
    (-1.0, 1.0),
    (-0.9781476007338057, 0.9567727288213006),
    (-0.913545457642601, 0.8345653031794292),
    (-0.8090169943749473, 0.6545084971874736),
    (-0.6691306063588579, 0.44773576836617285),
    (-0.49999999999999983, 0.24999999999999983),
    (-0.30901699437494734, 0.09549150281252623),
    (-0.10452846326765333, 0.010926199633097152),
    (0.10452846326765346, 0.010926199633097178),
    (0.30901699437494745, 0.0954915028125263),
    (0.5000000000000001, 0.2500000000000001),
    (0.6691306063588582, 0.4477357683661733),
    (0.8090169943749475, 0.6545084971874737),
    (0.9135454576426009, 0.834565303179429),
    (0.9781476007338057, 0.9567727288213006),
    (1.0, 1.0),
]

# ---------- Choose a new point to analyze ----------

# Replace this with whatever point you want to test (e.g. on your semicircle)
new_point = (0.0, 0.0)

points = base_points + [new_point]

ratio, diam, max_gap, (a, b), (c, e), path_ab, path_ce, d = max_gap_ratio_with_paths(
    points
)

print(f"Diameter: {diam:.6f}")
print(f"Max gap: {max_gap:.6f}")
print(f"Ratio (max_gap / diam): {ratio:.6f}")
print(f"Endpoints AB: {a} -> {b}")
print(f"Endpoints CE: {c} -> {e}")


# Print actual paths as coordinates
def path_coords(path):
    return [points[i] for i in path] if path is not None else None


print("Path AB (indices):", path_ab)
print("Path AB (coords):", path_coords(path_ab))
print("Path CE (indices):", path_ce)
print("Path CE (coords):", path_coords(path_ce))

# ---------- Plot ----------

plot_points_and_paths(
    points,
    path_ab,
    path_ce,
    ab_label=f"AB {a}->{b}",
    ce_label=f"CE {c}->{e}",
)
