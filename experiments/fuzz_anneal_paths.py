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
    n = len(d)
    INF = float("inf")
    L = [[INF] * n for _ in range(n)]
    for perm in itertools.permutations(range(n)):
        length = 0.0
        for i in range(n - 1):
            length += d[perm[i]][perm[i + 1]]
        s, t = perm[0], perm[-1]
        if s != t and length < L[s][t]:
            L[s][t] = length
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
        print(f"  {i}: ({x:.6f}, {y:.6f})")

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

    local_search(
        n_points=8,
        iters=2000,
        step_size=0.05,
        temperature_start=0.05,
        temperature_end=0.005,
        seed=123,
        # start_points=START,  # <-- supply or set to None
    )
