import itertools
import math
import random
from typing import List, Tuple


Point = List[float]


# =========================
#  Geometry / instance gen
# =========================


def random_point_in_ball(dim: int, radius: float = 1.0) -> Point:
    """
    Sample a point ~ uniform in the k-dimensional Euclidean ball of given radius.

    Algorithm:
      - sample a Gaussian vector g ~ N(0, I_dim)
      - normalize to get a random direction on the unit sphere
      - sample a radius r = R * U^(1/dim), where U ~ Uniform[0,1]
    """
    # Random direction (normal vector normalized)
    g = [random.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in g))
    if norm == 0.0:
        # extremely unlikely, but just in case; resample
        return random_point_in_ball(dim, radius)

    direction = [x / norm for x in g]

    # Radius with correct radial distribution for uniform volume
    u = random.random()
    r = radius * (u ** (1.0 / dim))

    return [r * x for x in direction]


def generate_instance(n: int, dim: int, radius: float = 1.0) -> List[Point]:
    """Generate n points in a k-dimensional Euclidean ball of given radius."""
    return [random_point_in_ball(dim, radius) for _ in range(n)]


def euclidean_distance_matrix(points: List[Point]) -> List[List[float]]:
    """Compute pairwise Euclidean distance matrix for a list of k-D points."""
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(
                sum(
                    (points[i][coord] - points[j][coord]) ** 2
                    for coord in range(len(points[0]))
                )
            )
            dist[i][j] = dist[j][i] = d
    return dist


def diameter(dist: List[List[float]]) -> float:
    """Compute diameter from a distance matrix."""
    n = len(dist)
    d = 0.0
    for i in range(n):
        for j in range(n):
            if dist[i][j] > d:
                d = dist[i][j]
    return d


# =========================
#  Hamiltonian path lengths
# =========================


def all_endpoint_optimal_paths(dist: List[List[float]]) -> List[List[float]]:
    """
    Given a metric (as a distance matrix) on n points, compute L(a,b) for all ordered pairs (a,b),
    where L(a,b) is the length of a shortest Hamiltonian path from a to b visiting all vertices.

    Brute force: iterate over all permutations of vertices; each permutation is a Hamiltonian path
    with endpoints (perm[0], perm[-1]). For each such pair, keep the minimum length seen.
    """
    n = len(dist)
    INF = float("inf")
    L = [[INF] * n for _ in range(n)]

    for perm in itertools.permutations(range(n)):
        # Compute path length for this permutation
        length = 0.0
        for i in range(n - 1):
            length += dist[perm[i]][perm[i + 1]]

        s = perm[0]
        t = perm[-1]
        if s != t and length < L[s][t]:
            L[s][t] = length

    return L


def max_endpoint_gap_ratio(dist: List[List[float]]) -> Tuple[float, float]:
    """
    Given a metric, compute:
        max_gap = max_{a!=b, c!=d} |L(a,b) - L(c,d)|
        ratio  = max_gap / diameter
    Returns (max_gap, ratio).
    """
    n = len(dist)
    diam = diameter(dist)
    L = all_endpoint_optimal_paths(dist)

    max_gap = 0.0
    for a in range(n):
        for b in range(n):
            if a == b or math.isinf(L[a][b]):
                continue
            for c in range(n):
                for d in range(n):
                    if c == d or math.isinf(L[c][d]):
                        continue
                    gap = abs(L[a][b] - L[c][d])
                    if gap > max_gap:
                        max_gap = gap

    ratio = max_gap / diam if diam > 0 else float("nan")
    return max_gap, ratio


# =========================
#  Fuzzing harness
# =========================


def fuzz_instances(
    n_points: int = 7,
    dim: int = 2,
    num_trials: int = 200,
    radius: float = 1.0,
    seed: int | None = None,
) -> None:
    """
    Fuzz random point sets in a k-dimensional ball and print the worst ratio observed of
        max_{a,b,c,d} |L(a,b) - L(c,d)| / diam(S).

    n_points   : number of points per instance (keep small, brute force is O(n!))
    dim        : dimension k of the Euclidean space (R^k)
    num_trials : how many random instances to test
    radius     : radius of the ball for point generation
    seed       : RNG seed for reproducibility (optional)
    """
    if seed is not None:
        random.seed(seed)

    best_ratio = -1.0
    best_gap = 0.0
    best_instance: List[Point] | None = None
    best_diameter = 0.0

    for trial in range(1, num_trials + 1):
        points = generate_instance(n_points, dim=dim, radius=radius)
        dist = euclidean_distance_matrix(points)
        diam = diameter(dist)
        max_gap, ratio = max_endpoint_gap_ratio(dist)

        if ratio > best_ratio:
            best_ratio = ratio
            best_gap = max_gap
            best_instance = points
            best_diameter = diam

        print(
            f"[trial {trial:3d}] dim = {dim}, max_gap = {max_gap:.6f}, "
            f"diam = {diam:.6f}, ratio = {ratio:.6f}, best_ratio = {best_ratio:.6f}",
        )

    print("\n=== Summary ===")
    print(f"n_points       = {n_points}")
    print(f"dim            = {dim}")
    print(f"num_trials     = {num_trials}")
    print(f"best_ratio     = {best_ratio:.6f}")
    print(f"best_gap       = {best_gap:.6f}")
    print(f"best_diameter  = {best_diameter:.6f}")
    if best_instance is not None:
        print("\nExample configuration achieving best_ratio:")
        for i, p in enumerate(best_instance):
            coords = ", ".join(f"{x:.6f}" for x in p)
            print(f"  {i}: ({coords})")


if __name__ == "__main__":
    # Adjust these as you like
    N_POINTS = 8  # 6–8 is typically safe; 9 is slow
    DIM = 5  # dimension k: 2 for plane, 3 for R^3, etc.
    NUM_TRIALS = 20000  # increase if you want more exploration
    RADIUS = 1.0
    SEED = 41  # or None for non-deterministic

    fuzz_instances(
        n_points=N_POINTS,
        dim=DIM,
        num_trials=NUM_TRIALS,
        radius=RADIUS,
        seed=SEED,
    )
