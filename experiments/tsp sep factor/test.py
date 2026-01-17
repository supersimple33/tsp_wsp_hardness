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
NUM_POINTS = 80
NA = NUM_POINTS // 2
NB = NUM_POINTS - NA

TAKE = 50_000  # how many random problems to generate+solve
START_INDEX = 0

DISTRIB_CODE = "cr"  # "u", "n", or "pX"
S_FACTOR = 0.5  # diameter-based separation factor s
CONCORDE_SEED = 43  # single seed per problem (but problems differ)

PRINT_FIRST_K_BAD = 0


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


def get_points(rng: np.random.Generator, num_points: int) -> np.ndarray:
    """
    DISTRIB_CODE options:
      "u"          : uniform integer square [0,SCALE_SIZE)^2
      "n"          : normal(0, SCALE_SIZE)
      "pX"         : power radial with exponent X (your old)
      "c"          : circularly-uniform on a circle (equal angles + random rotation)
      "cr"         : circularly-uniform on a circle (random angles i.i.d.)
      "annX"       : annulus with r in [X*R, R], angle uniform; e.g. ann0.8
                     (X in (0,1), closer to 1 => thin ring)
    """
    match DISTRIB_CODE:
        case "u":
            return rng.integers(0, SCALE_SIZE, size=(num_points, 2)).astype(np.float64)
        case "n":
            return rng.normal(size=(num_points, 2), scale=SCALE_SIZE)
        case x if x.startswith("p"):
            phi = 2.0 * np.pi * rng.random(num_points)
            r = rng.power(float(DISTRIB_CODE[1:]), num_points) * SCALE_SIZE
            return np.array([r * np.cos(phi), r * np.sin(phi)]).T
        # ---- NEW: circularly uniform schemes ----
        case "c":
            # "circularly uniform" in the strong sense:
            # equally spaced angles, random global rotation
            # (gives a very structured, symmetric set; good for stress testing)
            R = float(SCALE_SIZE)
            theta0 = 2.0 * np.pi * rng.random()
            k = np.arange(num_points, dtype=np.float64)
            theta = theta0 + 2.0 * np.pi * k / num_points
            return np.stack([R * np.cos(theta), R * np.sin(theta)], axis=1)
        case "cr":
            # "circularly uniform" in the weaker sense:
            # angles i.i.d. uniform; all points on the circle
            R = float(SCALE_SIZE)
            theta = 2.0 * np.pi * rng.random(num_points)
            return np.stack([R * np.cos(theta), R * np.sin(theta)], axis=1)
        case x if x.startswith("ann"):
            # uniform angle; radius in [alpha*R, R]
            # (thin ring if alpha ~ 0.9)
            R = float(SCALE_SIZE)
            alpha = float(x[3:])
            if not (0.0 < alpha < 1.0):
                raise ValueError("annX requires 0 < X < 1, e.g. ann0.9")
            theta = 2.0 * np.pi * rng.random(num_points)
            # sample radius uniformly in area over the annulus:
            # r^2 uniform in [alpha^2 R^2, R^2]
            u = rng.random(num_points)
            r = R * np.sqrt(alpha * alpha + (1.0 - alpha * alpha) * u)
            return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        case _:
            raise ValueError(f"Unknown distribution code: {DISTRIB_CODE}")


# -----------------------
# Geometry helpers (numpy)
# -----------------------


def pairwise_dists(P: np.ndarray) -> np.ndarray:
    dif = P[:, None, :] - P[None, :, :]
    return np.sqrt(np.sum(dif * dif, axis=-1))


def diameter(P: np.ndarray) -> float:
    return float(np.max(pairwise_dists(P)))


def set_distance(A: np.ndarray, B: np.ndarray) -> float:
    dif = A[:, None, :] - B[None, :, :]
    d = np.sqrt(np.sum(dif * dif, axis=-1))
    return float(np.min(d))


def enforce_diameter_separation(
    A: np.ndarray, B: np.ndarray, s: float
) -> Tuple[np.ndarray, float, float, float]:
    """
    Shift B in +x so that dist(A,B) >= s * D where D=max(diam(A),diam(B)).
    We do this by forcing x-gap = s*D, which implies Euclidean gap >= s*D.
    """
    dA = diameter(A)
    dB = diameter(B)
    D = max(dA, dB)
    if D <= 0:
        return B, D, 0.0, 0.0

    max_xA = float(np.max(A[:, 0]))
    min_xB = float(np.min(B[:, 0]))
    shift = (max_xA - min_xB) + s * D

    B2 = B.copy()
    B2[:, 0] += shift

    distAB = set_distance(A, B2)
    achieved_s = distAB / D
    return B2, D, distAB, achieved_s


# -----------------------
# Concorde solve + crossing test
# -----------------------


def build_concorde_solver(points: np.ndarray) -> TSPSolver:
    n = points.shape[0]
    dist_matrix = pairwise_dists(points)
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
    distAB: float
    opt_value: int
    cross_edges: int
    points: np.ndarray
    labels: np.ndarray
    tour: np.ndarray


# -----------------------
# Main experiment loop
# -----------------------


def main():
    t0 = time.time()

    # Use your ID scheme but just take TAKE of them
    ids = ["".join(x) for x in product(ALPHABET, repeat=4)]
    ids = ids[START_INDEX : START_INDEX + TAKE]

    bad: List[BadInstance] = []

    for k, id_ in enumerate(ids):
        name = f"{id_}_{NUM_POINTS}_{DISTRIB_CODE}_s{S_FACTOR}"

        rng = np.random.default_rng(seed=xxh64(name).intdigest())
        A = get_points(rng, NA)
        B = get_points(rng, NB)

        # optional recenter clusters (helps with "u")
        A = A - np.mean(A, axis=0, keepdims=True)
        B = B - np.mean(B, axis=0, keepdims=True)

        B2, D, distAB, achieved_s = enforce_diameter_separation(A, B, S_FACTOR)
        if D <= 0:
            continue

        points = np.vstack([A, B2]).astype(np.float64)
        labels = np.array([0] * NA + [1] * NB, dtype=np.int8)

        solver = build_concorde_solver(points)
        tour, optv = solve_concorde_once(solver, random_seed=CONCORDE_SEED)

        crosses = tour_cross_edges(tour, labels)
        if crosses > 2:
            bad.append(
                BadInstance(
                    name=name,
                    achieved_s=achieved_s,
                    D=D,
                    distAB=distAB,
                    opt_value=optv,
                    cross_edges=crosses,
                    points=points,
                    labels=labels,
                    tour=tour,
                )
            )

        if (k + 1) % 250 == 0:
            print(f"progress {k+1}/{len(ids)} | bad_so_far={len(bad)}")

    print("\nRESULTS")
    print(f"s={S_FACTOR} | problems={len(ids)} | NOT TSP-separated tours={len(bad)}")
    print(f"elapsed={time.time() - t0:.2f}s")

    # Print first few bad instances with raw points so you can analyze
    for ex in bad[:PRINT_FIRST_K_BAD]:
        print("\n--- BAD INSTANCE ---")
        print(ex.name)
        print(f"achieved_s={ex.achieved_s:.6f} (distAB={ex.distAB:.6f}, D={ex.D:.6f})")
        print(f"opt_value(int)={ex.opt_value}, cross_edges={ex.cross_edges}")
        print("A points:")
        for p in ex.points[:NA]:
            print(f"  ({p[0]:.6f}, {p[1]:.6f})")
        print("B points:")
        for p in ex.points[NA:]:
            print(f"  ({p[0]:.6f}, {p[1]:.6f})")
        print("tour (0-index):", ex.tour.tolist())


if __name__ == "__main__":
    main()
