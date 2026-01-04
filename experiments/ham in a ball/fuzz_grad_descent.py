import itertools
import math
import random
from typing import List, Tuple, Optional

Point = Tuple[float, float]

# ---------- TSP / ratio core ----------


def dist_matrix(points: List[Point]) -> List[List[float]]:
    n = len(points)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dd = math.hypot(xi - xj, yi - yj)
            d[i][j] = d[j][i] = dd
    return d


def diameter(d: List[List[float]]) -> float:
    return max(max(row) for row in d)


def all_endpoint_optimal_paths(d: List[List[float]]) -> List[List[float]]:
    """
    L[s][t] = length of shortest Hamiltonian path that
    starts at s, ends at t, and visits every vertex exactly once.

    Uses Held–Karp–style DP:
      For each fixed start s:
        dp[mask][j] = best cost of a path that starts at s,
                      visits exactly the vertices in 'mask',
                      and ends at j (s ∈ mask, j ∈ mask).
    """
    n = len(d)
    INF = float("inf")
    full_mask = (1 << n) - 1

    # Initialize output matrix with +inf (no path)
    L = [[INF] * n for _ in range(n)]

    for s in range(n):
        # dp[mask][j]
        dp = [[INF] * n for _ in range(1 << n)]
        dp[1 << s][s] = 0.0

        for mask in range(1 << n):
            # Must contain the start vertex
            if not (mask & (1 << s)):
                continue

            for j in range(n):
                if not (mask & (1 << j)):
                    continue

                cost = dp[mask][j]
                if math.isinf(cost):
                    continue

                # Extend path by going to a new vertex k not yet in mask
                if mask == full_mask:
                    # Already visited everyone; can't extend further
                    continue

                for k in range(n):
                    if mask & (1 << k):
                        continue
                    new_mask = mask | (1 << k)
                    new_cost = cost + d[j][k]
                    if new_cost < dp[new_mask][k]:
                        dp[new_mask][k] = new_cost

        # After the DP, dp[full_mask][t] is the best s→...→t Hamiltonian path
        for t in range(n):
            if t == s:
                continue
            L[s][t] = dp[full_mask][t]

    return L


def max_gap_ratio(points: List[Point]):
    """
    Compute:
      max_gap = max_{a!=b, c!=d} |L(a,b) - L(c,d)|
      diam    = diameter
      ratio   = max_gap / diam
    Return (max_gap, diam, ratio, (a,b,c,d)) where (a,b,c,d) achieves the max gap.
    """
    d = dist_matrix(points)
    diam = diameter(d)
    L = all_endpoint_optimal_paths(d)
    n = len(points)

    max_gap = 0.0
    argpairs = None
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
                        argpairs = (a, b, c, e)

    ratio = max_gap / diam if diam > 0 else float("nan")
    return max_gap, diam, ratio, argpairs


# ---------- Geometry / instance gen ----------


def random_point_in_disk(rng: random.Random, radius: float = 1.0) -> Point:
    r = radius * math.sqrt(rng.random())
    theta = 2 * math.pi * rng.random()
    return r * math.cos(theta), r * math.sin(theta)


def random_instance(rng: random.Random, n: int, radius: float = 1.0) -> List[Point]:
    return [random_point_in_disk(rng, radius) for _ in range(n)]


def normalize_to_unit_diameter(points: List[Point]) -> List[Point]:
    d = dist_matrix(points)
    diam = diameter(d)
    if diam == 0:
        return points
    scale = 1.0 / diam
    return [(x * scale, y * scale) for (x, y) in points]


# ---------- Finite-difference gradient ----------


def finite_diff_gradient(
    points: List[Point],
    eps: float = 1e-3,
) -> List[Tuple[float, float]]:
    """
    Approximate gradient of ratio w.r.t. each (x_i, y_i) using central differences.
    Returns a list grad[i] = (d/dx_i, d/dy_i).
    """
    n = len(points)
    _, _, base_ratio, _ = max_gap_ratio(points)
    # (We don't actually *need* base_ratio for central differences, but we might
    #  want it if we switch to forward differences later.)

    grads: List[Tuple[float, float]] = []

    for i in range(n):
        x, y = points[i]

        # d/dx: evaluate at x+eps and x-eps
        pts_plus = list(points)
        pts_minus = list(points)

        pts_plus[i] = (x + eps, y)
        pts_minus[i] = (x - eps, y)

        # normalize to keep scale invariant
        pts_plus = normalize_to_unit_diameter(pts_plus)
        pts_minus = normalize_to_unit_diameter(pts_minus)

        _, _, ratio_plus, _ = max_gap_ratio(pts_plus)
        _, _, ratio_minus, _ = max_gap_ratio(pts_minus)

        grad_x = (ratio_plus - ratio_minus) / (2.0 * eps)

        # d/dy: evaluate at y+eps and y-eps
        pts_plus_y = list(points)
        pts_minus_y = list(points)

        pts_plus_y[i] = (x, y + eps)
        pts_minus_y[i] = (x, y - eps)

        pts_plus_y = normalize_to_unit_diameter(pts_plus_y)
        pts_minus_y = normalize_to_unit_diameter(pts_minus_y)

        _, _, ratio_plus_y, _ = max_gap_ratio(pts_plus_y)
        _, _, ratio_minus_y, _ = max_gap_ratio(pts_minus_y)

        grad_y = (ratio_plus_y - ratio_minus_y) / (2.0 * eps)

        grads.append((grad_x, grad_y))

    return grads


# ---------- Gradient-based search (ascent) ----------


def gradient_search(
    n_points: int = 6,
    radius: float = 1.0,
    iters: int = 200,
    lr: float = 0.05,
    eps: float = 1e-3,
    seed: Optional[int] = None,
    start_points: Optional[List[Point]] = None,
    jitter: float = 0.0,  # small random noise per step if you want
):

    rng = random.Random(seed)

    # ----- choose starting configuration -----
    if start_points is not None:
        if len(start_points) != n_points:
            raise ValueError(
                f"start_points has {len(start_points)} points but n_points={n_points}"
            )
        pts = normalize_to_unit_diameter(start_points)
        print("Using user-supplied starting configuration.")
    else:
        pts = normalize_to_unit_diameter(random_instance(rng, n_points, radius))
        print("Using random starting configuration.")

    best_pts = pts
    best_gap, best_diam, best_ratio, best_pairs = max_gap_ratio(pts)

    for t in range(iters):
        # Compute (approximate) gradient of ratio at current pts
        grads = finite_diff_gradient(pts, eps=eps)

        # Gradient ascent step
        new_pts = []
        for (x, y), (gx, gy) in zip(pts, grads):
            # Optional tiny jitter to avoid stagnation
            jx = jitter * (2 * rng.random() - 1) if jitter > 0 else 0.0
            jy = jitter * (2 * rng.random() - 1) if jitter > 0 else 0.0
            new_x = x + lr * gx + jx
            new_y = y + lr * gy + jy
            new_pts.append((new_x, new_y))

        # Normalize to unit diameter again (scale invariance)
        new_pts = normalize_to_unit_diameter(new_pts)

        # Evaluate new configuration
        new_gap, new_diam, new_ratio, new_pairs = max_gap_ratio(new_pts)

        # Pure gradient ascent: accept unconditionally
        pts = new_pts

        if new_ratio > best_ratio:
            best_ratio = new_ratio
            best_gap = new_gap
            best_diam = new_diam
            best_pairs = new_pairs
            best_pts = new_pts

        if (t + 1) % max(1, iters // 10) == 0:
            print(
                f"[iter {t+1:4d}] current_ratio = {new_ratio:.6f}, "
                f"best_ratio = {best_ratio:.6f}"
            )

    print("\n=== Gradient search result ===")
    print(f"best_ratio = {best_ratio:.6f}")
    print(f"best_gap   = {best_gap:.6f}")
    print(f"best_diam  = {best_diam:.6f}")
    print(f"best_pairs = {best_pairs}")
    print("\nBest configuration:")
    for i, (x, y) in enumerate(best_pts):
        print(f"  {i}: ({x:.6f}, {y:.6f})")

    return best_pts, (best_gap, best_diam, best_ratio, best_pairs)


if __name__ == "__main__":
    # Example: start from your 10-point config
    START = [
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

    gradient_search(
        n_points=10,
        radius=1.0,
        iters=100,  # each iter does a bunch of DP calls; tune as needed
        lr=0.05,
        eps=1e-3,
        seed=123,
        start_points=START,
        jitter=0.0,
    )
