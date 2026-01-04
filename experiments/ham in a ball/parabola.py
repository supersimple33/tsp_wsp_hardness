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


# ---------- Chebyshev helpers ----------


def chebyshev_lobatto_nodes(num_points, x_min=-1.0, x_max=1.0):
    """
    Chebyshev–Lobatto nodes on [x_min, x_max].

    Standard Lobatto nodes on [-1,1] are cos(pi*k/(n-1)), k=0..n-1,
    which include the endpoints ±1 and cluster near them.
    """
    if num_points == 1:
        return np.array([(x_min + x_max) * 0.5])

    k = np.arange(num_points)
    # nodes in [-1, 1], from 1 down to -1
    xs_std = np.cos(math.pi * k / (num_points - 1))

    # map [-1,1] -> [x_min, x_max]
    xs = 0.5 * (x_max + x_min) + 0.5 * (x_max - x_min) * xs_std

    # sort so xs are increasing left-to-right
    xs = np.sort(xs)
    return xs


# ---------- parabola helpers ----------


def parabola_points(
    num_points,
    x_min=-1.0,
    x_max=1.0,
    a=1.0,
    b=0.0,
    c=0.0,
):
    """
    Generate num_points on the parabola y = a x^2 + b x + c
    using Chebyshev–Lobatto x-nodes in [x_min, x_max].

    Returns:
      pts: list[(x, y)]
      xs:  numpy array of x-coordinates (for analysis)
    """
    xs = chebyshev_lobatto_nodes(num_points, x_min=x_min, x_max=x_max)
    pts = [(float(x), float(a * x * x + b * x + c)) for x in xs]
    return pts, xs


def max_ratio_with_moving_point_on_parabola(
    base_points,
    x_min=-1.0,
    x_max=1.0,
    a=1.0,
    b=0.0,
    c=0.0,
    num_candidates=200,
):
    """
    Keep base_points fixed. Add one extra point that moves along the parabola
    y = a x^2 + b x + c, x in [x_min, x_max].

    Returns:
      best_ratio
      best_x
      best_point = (x, y)
      results: list of (x, point, ratio, full_point_set)
    """
    # still do a uniform sweep for the moving point;
    # base_points themselves are Chebyshev-based.
    xs = np.linspace(x_min, x_max, num_candidates)

    results = []
    best_ratio = -1.0
    best_x = None
    best_point = None

    for x in xs:
        y = a * x * x + b * x + c
        p = (float(x), float(y))
        pts = base_points + [p]
        ratio = max_gap_ratio(pts)

        results.append((float(x), p, ratio, pts))

        if ratio > best_ratio:
            best_ratio = ratio
            best_x = float(x)
            best_point = p

    return best_ratio, best_x, best_point, results


# ---------- experiments ----------

if __name__ == "__main__":
    num_points = 16
    # Choose a simple parabola y = x^2 on [-1, 1]
    a, b, c = 1, 0.0, 0.0
    x_min, x_max = -1.0, 1.0

    # (A) All points fixed on the parabola (Chebyshev–Lobatto nodes)
    parabola_pts, xs = parabola_points(
        num_points,
        x_min=x_min,
        x_max=x_max,
        a=a,
        b=b,
        c=c,
    )
    ratio_parabola = max_gap_ratio(parabola_pts)

    print("\n=== Fixed parabola configuration (Chebyshev–Lobatto) ===")
    print("x-coordinates (Chebyshev nodes):")
    print(xs)
    print("points:")
    for p in parabola_pts:
        print(f"{p},")
    print("ratio:", ratio_parabola)

    # (B) One moving point along the parabola with (num_points - 1) as base
    base_pts, base_xs = parabola_points(
        num_points - 1,
        x_min=x_min,
        x_max=x_max,
        a=a,
        b=b,
        c=c,
    )

    raise  # keep your early stop here if you want to inspect base_pts

    best_ratio, best_x, best_point, sweep_results = (
        max_ratio_with_moving_point_on_parabola(
            base_pts,
            x_min=x_min,
            x_max=x_max,
            a=a,
            b=b,
            c=c,
            num_candidates=200,
        )
    )

    print("\n=== Moving-point parabola sweep ===")
    print("base points (Chebyshev–Lobatto):")
    for p in base_pts:
        print(f"{p},")

    print("\nBEST configuration:")
    print(" best_ratio:", best_ratio)
    print(" best_x:", best_x)
    print(" best_point:", best_point)

    print("\nSample of sweep results (first 5):")
    for x, p, r, pts in sweep_results[:5]:
        print(f" x={x:.3f}, point={p}, ratio={r}")
        # pts is the full point set used for that ratio
