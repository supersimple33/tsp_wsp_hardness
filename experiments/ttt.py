import math
import numpy as np
import matplotlib.pyplot as plt


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


def all_endpoint_optimal_paths(d):
    n = len(d)
    INF = float("inf")
    full_mask = (1 << n) - 1
    L = [[INF] * n for _ in range(n)]
    for s in range(n):
        dp = [[INF] * n for _ in range(1 << n)]
        dp[1 << s][s] = 0.0
        for mask in range(1 << n):
            if not (mask & (1 << s)):
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
        for t in range(n):
            if t != s:
                L[s][t] = dp[full_mask][t]
    return L


def max_gap_ratio(points):
    d = dist_matrix(points)
    diam = diameter(d)
    L = all_endpoint_optimal_paths(d)
    n = len(points)
    max_gap = 0.0
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
    return (max_gap / diam) if diam > 0 else float("nan")


base_points = [
    (0.255691, 0.152402),
    (-0.242313, 0.038200),
    (0.281690, 0.080247),
    (-0.124978, -0.764436),
    (0.719447, -0.235692),
    (-0.089001, 0.232604),
    (0.058391, 0.211552),
    (0.162232, 0.178466),
    (-0.265936, -0.103417),
    (-0.211332, 0.128331),
]

xs = np.linspace(-0.99, 0.99, 30)
ys = np.linspace(-0.99, 0.99, 30)
heat = np.zeros((len(ys), len(xs)))

for iy, y in enumerate(ys):
    for ix, x in enumerate(xs):
        pts = base_points + [(float(x), float(y))]
        heat[iy, ix] = max_gap_ratio(pts)

plt.figure()
plt.imshow(
    heat,
    origin="lower",
    extent=[xs[0], xs[-1], ys[0], ys[-1]],
    aspect="auto",
)
plt.colorbar()
plt.title("Ratio heatmap (coarse grid) for adding one new point")
plt.xlabel("x")
plt.ylabel("y")

# ---- overlay original coordinates ----
bx, by = zip(*base_points)
plt.scatter(bx, by, s=40, edgecolors="white", facecolors="none", linewidths=1.5)

plt.show()
