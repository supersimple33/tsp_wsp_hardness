import math
import numpy as np


# ---------- your existing core ----------


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


# ---------- semicircle helpers ----------


def semicircle_points(num_points, radius=1.0, center=(0.0, 0.0), orientation="upper"):
    """
    Return a list of points on a semicircle.
    Also returns the angle list so you can analyze structure if useful.
    """
    cx, cy = center
    if orientation == "upper":
        angles = np.linspace(0.0, math.pi, num_points)
    elif orientation == "full":
        angles = np.linspace(0.0, 2 * math.pi, num_points)
    elif orientation == "quarter":
        angles = np.linspace(0.0, math.pi / 2, num_points)
    elif orientation == "tripleQuarter":
        angles = np.linspace(0.0, 3 * math.pi / 2, num_points)
    else:
        angles = np.linspace(-math.pi, 0.0, num_points)

    pts = [(cx + radius * math.cos(t), cy + radius * math.sin(t)) for t in angles]
    return pts, angles


def max_ratio_with_moving_point_on_semicircle(
    base_points,
    radius=1.0,
    center=(0.0, 0.0),
    orientation="upper",
    num_candidates=200,
):
    """
    Sweep a moving point along the semicircle.
    Returns:
      best_ratio, best_theta, best_point, all_test_points (list of (theta, point, ratio))
    """
    cx, cy = center
    if orientation == "upper":
        thetas = np.linspace(0.0, math.pi, num_candidates)
    else:
        thetas = np.linspace(-math.pi, 0.0, num_candidates)

    results = []
    best_ratio = -1.0
    best_theta = None
    best_point = None

    for theta in thetas:
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        pts = base_points + [(x, y)]
        ratio = max_gap_ratio(pts)
        results.append((theta, (x, y), ratio, pts))

        if ratio > best_ratio:
            best_ratio = ratio
            best_theta = theta
            best_point = (x, y)

    return best_ratio, best_theta, best_point, results


# ---------- experiments ----------

if __name__ == "__main__":
    num_points = 10
    R = 1.0

    # (A) All points fixed on a semicircle
    semicircle_pts, angles = semicircle_points(
        num_points, radius=R, orientation="tripleQuarter"
    )
    ratio_semicircle = max_gap_ratio(semicircle_pts)

    print("\n=== Fixed semicircle configuration ===")
    print("points:")
    for p in semicircle_pts:
        print(f" {p},")
    print("ratio:", ratio_semicircle)

    # (B) One moving point along the semicircle
    base_pts, _ = semicircle_points(
        num_points - 1, radius=R, orientation="tripleQuarter"
    )
    best_ratio, best_theta, best_point, sweep_results = (
        max_ratio_with_moving_point_on_semicircle(
            base_pts, radius=R, orientation="tripleQuarter"
        )
    )

    print("\n=== Moving-point semicircle sweep ===")
    print("base points:")
    for p in base_pts:
        print(f" {p},")

    print("\nBEST configuration:")
    print(" best_ratio:", best_ratio)
    print(" best_theta:", best_theta)
    print(" best_point:", best_point)

    # Optional: dump all evaluated configs if you want to analyze offline
    # (theta, point, ratio, full_point_set)
    # Example: write to CSV / pickle later if you want
    print("\nSample of sweep results (first 5):")
    for theta, p, r, _ in sweep_results[:5]:
        print(f" theta={theta:.3f}, point={p}, ratio={r}")
