import os
import time
import math
from dataclasses import dataclass
from itertools import product
from string import ascii_lowercase as ALPHABET
from typing import Optional, Tuple, List

import numpy as np
from concorde.tsp import TSPSolver
from xxhash import xxh64

# results
# 50k u p=25 s=0.33 -> 0
# 50k u p=10 s=0.33 -> 3
# 50k u p=6  s=0.33 -> 10
# 50k u p=5  s=0.33 -> 81

# 50k u     p=5 s=0.5 -> 0
# 50k n     p=5 s=0.5 -> 0
# 50k p0.33 p=5 s=0.5 -> 0

# 50k u p=10 s=0.5 -> 0
# 50k c p=10 s=0.5 -> 0
# 50k cr p=10 s=0.5 -> 0
# 50k ann0.8 p=10 s=0.5 -> 0
# 50k ann0.5 p=10 s=0.5 -> 0

# 50k u p=20 s=0.5 -> 0
# c cr ann0.8 ann0.5

# 50k u p=40 s=0.5 -> 0
# c cr ann0.8 ann0.5

# p=80 -> 0

# -----------------------
# Parameters (edit freely)
# -----------------------

SCALE_SIZE = 10000
NUM_POINTS = 5
DIMENSION = 3  # 2 or 3
METRIC = "manhattan"  # "euclidean" or "manhattan"
NA = NUM_POINTS // 2
NB = NUM_POINTS - NA

TAKE = 10_000  # how many random problems to generate+solve
START_INDEX = 0

DISTRIB_CODE = "cr"  # "u", "n", or "pX"
S_FACTOR = 0.5  # diameter-based separation factor s
CONCORDE_SEED = 44  # single seed per problem (but problems differ)

PRINT_FIRST_K_BAD = 1


# -----------------------
# Silence Concorde output (your pattern)
# -----------------------

STDOUT = 1
STDERR = 2
saved_fd = os.dup(STDOUT)
error_fd = os.dup(STDERR)
null_fd = os.open(os.devnull, os.O_WRONLY)


# -----------------------
# Distributions (your style)
# -----------------------


def _random_unit_vectors(
    rng: np.random.Generator, count: int, dimension: int, metric: str
) -> np.ndarray:
    """Sample unit vectors under the chosen metric."""
    if metric == "euclidean":
        direction = rng.normal(size=(count, dimension))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        return direction
    if metric == "manhattan":
        base = rng.exponential(scale=1.0, size=(count, dimension))
        signs = rng.choice([-1.0, 1.0], size=(count, dimension))
        direction = signs * base
        direction /= np.sum(np.abs(direction), axis=1, keepdims=True)
        return direction
    raise ValueError(f"Unknown metric: {metric}")


def _fibonacci_sphere(count: int) -> np.ndarray:
    """Even-ish spread on the unit sphere (3D)."""
    k = np.arange(count, dtype=np.float64) + 0.5
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * k / count
    radius = np.sqrt(1.0 - y * y)
    theta = phi * k
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.stack([x, y, z], axis=1)


def _diamond_boundary_2d(count: int, R: float, rng: np.random.Generator) -> np.ndarray:
    """Points evenly along the L1 unit diamond boundary scaled by R."""
    t = (np.arange(count, dtype=np.float64) + rng.random()) / count * 4.0
    x = np.empty(count)
    y = np.empty(count)
    for i, ti in enumerate(t):
        if ti < 1.0:
            x[i] = R * (1.0 - ti)
            y[i] = R * ti
        elif ti < 2.0:
            x[i] = -R * (ti - 1.0)
            y[i] = R * (2.0 - ti)
        elif ti < 3.0:
            x[i] = -R * (3.0 - ti)
            y[i] = -R * (ti - 2.0)
        else:
            x[i] = R * (ti - 3.0)
            y[i] = -R * (4.0 - ti)
    return np.stack([x, y], axis=1)


def get_points(
    rng: np.random.Generator,
    num_points: int = NUM_POINTS,
    distrib_code: str = DISTRIB_CODE,
    scale_size: int = SCALE_SIZE,
    dimension: int = DIMENSION,
    metric: str = METRIC,
) -> np.ndarray:
    """
    distrib_code options:
      "u", "n", "pX", "c", "cr", "annX"
    """
    match distrib_code:
        case "u":
            return rng.integers(0, scale_size, size=(num_points, dimension)).astype(
                np.float64
            )
        case "n":
            return rng.normal(size=(num_points, dimension), scale=scale_size)
        case x if x.startswith("p"):
            power = float(x[1:])
            direction = _random_unit_vectors(rng, num_points, dimension, metric)
            r = rng.power(power, num_points) * scale_size
            return direction * r[:, None]
        case "c":
            R = float(scale_size)
            if metric == "euclidean" and dimension == 2:
                theta0 = 2.0 * np.pi * rng.random()
                k = np.arange(num_points, dtype=np.float64)
                theta = theta0 + 2.0 * np.pi * k / num_points
                return np.stack([R * np.cos(theta), R * np.sin(theta)], axis=1)
            if metric == "euclidean" and dimension == 3:
                return _fibonacci_sphere(num_points) * R
            if metric == "manhattan" and dimension == 2:
                return _diamond_boundary_2d(num_points, R, rng)
            if metric == "manhattan" and dimension == 3:
                direction = _random_unit_vectors(rng, num_points, dimension, metric)
                return direction * R
            raise ValueError("circle distribution requires dimension >= 2")
        case "cr":
            R = float(scale_size)
            if metric == "euclidean" and dimension == 2:
                theta = 2.0 * np.pi * rng.random(num_points)
                return np.stack([R * np.cos(theta), R * np.sin(theta)], axis=1)
            direction = _random_unit_vectors(rng, num_points, dimension, metric)
            return direction * R
        case x if x.startswith("ann"):
            R = float(scale_size)
            alpha = float(x[3:])
            if not (0.0 < alpha < 1.0):
                raise ValueError("annX requires 0 < X < 1, e.g. ann0.9")
            direction = _random_unit_vectors(rng, num_points, dimension, metric)
            u = rng.random(num_points)
            r = R * (alpha**dimension + (1.0 - alpha**dimension) * u) ** (
                1.0 / dimension
            )
            return direction * r[:, None]
        case _:
            raise ValueError(f"Unknown distribution code: {distrib_code}")


# -----------------------
# Geometry helpers (numpy)
# -----------------------


def pairwise_dists(P: np.ndarray, metric: str) -> np.ndarray:
    dif = P[:, None, :] - P[None, :, :]
    if metric == "euclidean":
        return np.sqrt(np.sum(dif * dif, axis=-1))
    if metric == "manhattan":
        return np.sum(np.abs(dif), axis=-1)
    raise ValueError(f"Unknown metric: {metric}")


def diameter(P: np.ndarray, metric: str) -> float:
    return float(np.max(pairwise_dists(P, metric)))


def set_distance(A: np.ndarray, B: np.ndarray, metric: str) -> float:
    dif = A[:, None, :] - B[None, :, :]
    if metric == "euclidean":
        d = np.sqrt(np.sum(dif * dif, axis=-1))
    elif metric == "manhattan":
        d = np.sum(np.abs(dif), axis=-1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return float(np.min(d))


def enforce_diameter_separation(
    A: np.ndarray, B: np.ndarray, s: float, metric: str
) -> Tuple[np.ndarray, float, float, float]:
    """
    Shift B in +x so that dist(A,B) >= s * D where D=max(diam(A),diam(B)).
    We do this by forcing x-gap = s*D, which implies Euclidean gap >= s*D.
    """
    dA = diameter(A, metric)
    dB = diameter(B, metric)
    D = max(dA, dB)
    if D <= 0:
        return B, D, 0.0, 0.0

    max_xA = float(np.max(A[:, 0]))
    min_xB = float(np.min(B[:, 0]))
    shift = (max_xA - min_xB) + s * D

    B2 = B.copy()
    B2[:, 0] += shift

    distAB = set_distance(A, B2, metric)
    achieved_s = distAB / D
    return B2, D, distAB, achieved_s


# -----------------------
# Concorde solve + crossing test
# -----------------------


def build_concorde_solver(points: np.ndarray, metric: str) -> TSPSolver:
    n = points.shape[0]
    dist_matrix = pairwise_dists(points, metric)
    ltri = np.round(dist_matrix[np.tril_indices(n, k=-1)]).astype(np.int32)
    return TSPSolver.from_lower_tri(shape=n, edges=ltri)


def solve_concorde_once(solver: TSPSolver, random_seed: int) -> Tuple[np.ndarray, int]:
    os.dup2(null_fd, STDOUT) and os.dup2(null_fd, STDERR)
    sol = solver.solve(verbose=False, random_seed=random_seed)
    os.dup2(saved_fd, STDOUT) and os.dup2(error_fd, STDERR)

    assert sol.found_tour, "Concorde did not find a tour"
    assert sol.success, "Concorde did not certify optimality"
    return np.array(sol.tour, dtype=np.int32), int(sol.optimal_value)


def tour_cross_edges(tour: np.ndarray, labels: np.ndarray) -> int:
    n = len(tour)
    c = 0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        c += int(labels[u] != labels[v])
    return c


@dataclass
class BadInstance:
    name: str
    achieved_s: float
    D: float
    dist_ab: float
    opt_value: int
    cross_edges: int
    points: np.ndarray
    labels: np.ndarray
    tour: np.ndarray


def run_experiment(
    *,
    scale_size: int,
    num_points: int,
    dimension: int,
    metric: str,
    take: int,
    start_index: int,
    distrib_code: str,
    s_factor: float,
    concorde_seed: int,
    print_every: int = 0,
) -> Tuple[int, List[BadInstance]]:
    """
    Returns:
      bad_count, bad_instances (kept in-memory; you can choose to ignore for sweeps)
    """
    NA = num_points // 2
    NB = num_points - NA

    ids = ["".join(x) for x in product(ALPHABET, repeat=4)]
    ids = ids[start_index : start_index + take]

    bad: List[BadInstance] = []

    for k, id_ in enumerate(ids):
        name = f"{id_}_{num_points}_{distrib_code}_s{s_factor}_{dimension}d_{metric}"

        rng = np.random.default_rng(seed=xxh64(name).intdigest())
        A = get_points(rng, NA, distrib_code, scale_size, dimension, metric)
        B = get_points(rng, NB, distrib_code, scale_size, dimension, metric)

        A = A - np.mean(A, axis=0, keepdims=True)
        B = B - np.mean(B, axis=0, keepdims=True)

        B2, D, dist_ab, achieved_s = enforce_diameter_separation(A, B, s_factor, metric)
        if D <= 0:
            continue

        points = np.vstack([A, B2]).astype(np.float64)
        labels = np.array([0] * NA + [1] * NB, dtype=np.int8)

        solver = build_concorde_solver(points, metric)
        tour, optv = solve_concorde_once(solver, random_seed=concorde_seed)

        crosses = tour_cross_edges(tour, labels)
        if crosses > 2:
            bad.append(
                BadInstance(
                    name=name,
                    achieved_s=achieved_s,
                    D=D,
                    dist_ab=dist_ab,
                    opt_value=optv,
                    cross_edges=crosses,
                    points=points,
                    labels=labels,
                    tour=tour,
                )
            )

        if print_every and (k + 1) % print_every == 0:
            print(f"progress {k+1}/{len(ids)} | bad_so_far={len(bad)}")

    return len(bad), bad


# -----------------------
# Main experiment loop
# -----------------------


def main():
    t0 = time.time()
    bad_count, bad = run_experiment(
        scale_size=SCALE_SIZE,
        num_points=NUM_POINTS,
        dimension=DIMENSION,
        metric=METRIC,
        take=TAKE,
        start_index=START_INDEX,
        distrib_code=DISTRIB_CODE,
        s_factor=S_FACTOR,
        concorde_seed=CONCORDE_SEED,
        print_every=250,
    )

    print("\nRESULTS")
    print(f"s={S_FACTOR} | problems={TAKE} | NOT TSP-separated tours={bad_count}")
    print(f"elapsed={time.time() - t0:.2f}s")

    for ex in bad[:PRINT_FIRST_K_BAD]:
        print("\n--- BAD INSTANCE ---")
        print(ex.name)
        print(f"achieved_s={ex.achieved_s:.6f} (distAB={ex.dist_ab:.6f}, D={ex.D:.6f})")
        print(f"opt_value(int)={ex.opt_value}, cross_edges={ex.cross_edges}")
        print("A points:")
        for p in ex.points[:NA]:
            coords = ", ".join(f"{v:.6f}" for v in p)
            print(f"  ({coords})")
        print("B points:")
        for p in ex.points[NA:]:
            coords = ", ".join(f"{v:.6f}" for v in p)
            print(f"  ({coords})")
        print("tour (0-index):", ex.tour.tolist())


if __name__ == "__main__":
    main()
