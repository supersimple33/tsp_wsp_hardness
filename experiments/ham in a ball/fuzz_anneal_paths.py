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


# ---------- Hill climbing / annealing ----------


def local_search(
    n_points: int = 6,
    radius: float = 1.0,
    iters: int = 5000,
    step_size: float = 0.05,
    temperature_start: float = 0.1,
    temperature_end: float = 0.001,
    seed: Optional[int] = None,
    start_points: Optional[List[Point]] = None,  # <-- NEW
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

    current_pts = pts
    current_ratio = best_ratio

    def schedule(t):
        return temperature_start + (temperature_end - temperature_start) * (
            t / max(1, iters - 1)
        )

    for t in range(iters):
        temp = schedule(t)

        i = rng.randrange(n_points)
        x, y = current_pts[i]

        dx = step_size * (2 * rng.random() - 1)
        dy = step_size * (2 * rng.random() - 1)

        new_pts = list(current_pts)
        new_pts[i] = (x + dx, y + dy)
        new_pts = normalize_to_unit_diameter(new_pts)

        new_gap, new_diam, new_ratio, new_pairs = max_gap_ratio(new_pts)

        delta = new_ratio - current_ratio
        accept = (delta >= 0) or (rng.random() < math.exp(delta / max(temp, 1e-8)))

        if accept:
            current_pts = new_pts
            current_ratio = new_ratio

            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_gap = new_gap
                best_diam = new_diam
                best_pairs = new_pairs
                best_pts = new_pts

        if (t + 1) % max(1, iters // 10) == 0:
            print(f"[iter {t+1:5d}] current={current_ratio:.6f}, best={best_ratio:.6f}")

    print("\n=== Local search result ===")
    print(f"best_ratio = {best_ratio:.6f}")
    print(f"best_gap   = {best_gap:.6f}")
    print(f"best_diam  = {best_diam:.6f}")
    print(f"best_pairs = {best_pairs}")
    print("\nBest configuration:")
    for i, (x, y) in enumerate(best_pts):
        print(f"  ({x:.6f}, {y:.6f}),")

    return best_pts, (best_gap, best_diam, best_ratio, best_pairs)


if __name__ == "__main__":
    # Example: start from a known configuration
    # START = [
    #    (0.278862, 0.286847),
    #    (-0.484354, 0.418137),
    #    (0.389020, 0.170291),
    #    (0.403116, -0.019424),
    #    (-0.247903, -0.553506),
    #    (0.300714, -0.177158),
    #    (0.180302, 0.350063),
    #    (0.193857, -0.268223),
    # ]

    START = [
        (-1.0, 1.0),
        (-0.7777777777777778, 0.6049382716049383),
        (-0.5555555555555556, 0.308641975308642),
        (-0.33333333333333337, 0.11111111111111113),
        (-0.11111111111111116, 0.01234567901234569),
        (0.11111111111111116, 0.01234567901234569),
        (0.33333333333333326, 0.11111111111111106),
        (0.5555555555555554, 0.30864197530864174),
        (0.7777777777777777, 0.6049382716049381),
        (1.0, 1.0),
    ]

    local_search(
        n_points=10,
        iters=2000,
        step_size=0.01,
        temperature_start=0.005,
        temperature_end=0.0005,
        seed=12,
        start_points=START,  # <-- supply or set to None
    )
