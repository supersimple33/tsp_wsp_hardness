# Create a bunch of problems following some probability distribution
# Then solve the problem and save its solution
# Then generate n subproblems by removing each point and solve those
# to identify which points are most responsible for increasing the size of the solution

import os
import math
import time
from itertools import product, combinations, chain
from string import ascii_lowercase as ALPHABET

import numpy as np
import tsplib95
from concorde.tsp import TSPSolver  # DC
import matplotlib.pyplot as plt

# NOTE: We need constant hashes so cant us python's hash
from xxhash import xxh64

from wsp import tsp, ds

SCALE_SIZE = 10000
NUM_POINTS = 50
START_INDEX = 0
TAKE = 2
DISTRIB_CODE = "p0.33"
# DISTRIB_CODE = "u"
EXIST_OK = True
NUM_REMOVED = 2

# Silencing stuff
STDOUT = 1
STDERR = 2
saved_fd = os.dup(STDOUT)
error_fd = os.dup(STDERR)
null_fd = os.open(os.devnull, os.O_WRONLY)

t = time.time()


def power_subset(ss, k=None):
    """Generate all subsets of a set with size k."""
    if k is None:
        k = len(ss) + 1
    return chain(*map(lambda x: combinations(ss, x), range(1, k + 1)))


def get_points(rng: np.random.Generator, num_points: int) -> np.ndarray:
    match DISTRIB_CODE:
        case "u":  # Uniform distribution with bounds
            return rng.integers(0, SCALE_SIZE, size=(num_points, 2)).astype(
                dtype=np.float64
            )  # how do we decide this also typing is a factor
        case "n":  # Normal distribution
            return rng.normal(
                size=(num_points, 2), scale=SCALE_SIZE
            )  # TODO: Add scaling here (how should i decide????)
        case x if x.startswith("p"):  # Power distribution
            phi = 2.0 * np.pi * rng.random(num_points)
            r = rng.power(float(DISTRIB_CODE[1:]), num_points) * SCALE_SIZE
            return np.array([r * np.cos(phi), r * np.sin(phi)]).T
        case _:
            raise ValueError(f"Unknown distribution code: {DISTRIB_CODE}")


ids = ["".join(x) for x in product(ALPHABET, repeat=3)]
assert len(ids) >= START_INDEX + TAKE
ids = ids[START_INDEX : START_INDEX + TAKE]

# TODO: MULTI-PROCESSING
for i, id in enumerate(ids):
    # Save the problem
    name = f"{id}{NUM_POINTS}{DISTRIB_CODE}"
    rng: np.random.Generator = np.random.default_rng(seed=xxh64(name).intdigest())
    np_points = get_points(rng, NUM_POINTS)
    points = [ds.Point(x, y) for x, y in np_points]
    # plt.scatter([p.x for p in points], [p.y for p in points])
    # plt.show()

    # TODO: check if this already exists

    # mainProblem = tsp.TravellingSalesmanProblem[TREE_TYPE](
    #     TREE_TYPE, points, AX, s=S_FACTOR
    # )

    lib_problem = tsplib95.models.StandardProblem(
        name=name,
        comment=f"{NUM_POINTS} points generated from {DISTRIB_CODE} distribution (#{i + START_INDEX})",
        type="TSP",
        dimension=NUM_POINTS,  # CHECK TYPES
        edge_weight_type="EUC_2D",
        node_coords={i + 1: (p.x, p.y) for i, p in enumerate(points)},
    )

    # BEGIN INITIAL SOLUTION (NEW CODE IS 2x faster)

    dist_matrix = np.sqrt(
        np.sum(
            (np_points[:, np.newaxis, :] - np_points[np.newaxis, :, :]) ** 2, axis=-1
        )
    )
    ltri = np.round(dist_matrix[np.tril_indices(NUM_POINTS, k=-1)]).astype(np.int32)
    solver = TSPSolver.from_lower_tri(shape=NUM_POINTS, edges=ltri)
    os.dup2(null_fd, STDOUT) and os.dup2(null_fd, STDERR)
    solution = solver.solve(verbose=False, random_seed=42)
    os.dup2(saved_fd, STDOUT) and os.dup2(error_fd, STDERR)

    # lib_problem.save("solution area/temp.tsp")
    # os.dup2(null_fd, STDOUT) and os.dup2(null_fd, STDERR)
    # solver = TSPSolver.from_tspfile("solution area/temp.tsp")
    # solution = solver.solve(verbose=False, random_seed=42)
    # os.dup2(saved_fd, STDOUT) and os.dup2(error_fd, STDERR)

    assert solution.success  # Check that the solution is optimal
    assert solution.found_tour  # DEBUG: If this fails a different seed should be tried
    parent_solution = solution.tour

    lib_problem.tours = [
        [x + 1 for x in list(solution.tour)],
    ]
    lib_problem.comment = (
        f"Concorde optimal: {solution.optimal_value} {lib_problem.comment}"
    )

    os.makedirs(f"DATA_GEN_{NUM_POINTS}{DISTRIB_CODE}/{name}", exist_ok=EXIST_OK)
    lib_problem.save(f"DATA_GEN_{NUM_POINTS}{DISTRIB_CODE}/{name}/{name}.tsp")

    # MARK: - Work on Sub Problems
    for removed_points in power_subset(range(NUM_POINTS), NUM_REMOVED):
        # sub_problem = tsp.TravellingSalesmanProblem[TREE_TYPE]

        sub_name = f"{name}_" + "_".join([str(x + 1) for x in removed_points])

        lib_problem = tsplib95.models.StandardProblem(
            name=sub_name,
            comment=f"Removed points {[x+1 for x in removed_points]} from {NUM_POINTS} points generated from {DISTRIB_CODE} distribution",
            type="TSP",
            dimension=NUM_POINTS - len(removed_points),  # CHECK TYPES
            edge_weight_type="EUC_2D",
            node_coords={
                idx + 1: (p.x, p.y)
                for idx, (_, p) in enumerate(
                    (j, p) for j, p in enumerate(points) if j not in removed_points
                )
            },
        )

        # BEGIN INITIAL SOLUTION (NEW CODE IS 2x faster)
        mask = np.ones(NUM_POINTS, dtype=bool)
        mask[list(removed_points)] = False
        sub_dist_matrix = dist_matrix[np.ix_(mask, mask)]
        ltri = sub_dist_matrix[np.tril_indices(len(sub_dist_matrix), k=-1)]
        solver = TSPSolver.from_lower_tri(
            shape=NUM_POINTS - len(removed_points), edges=ltri.astype(np.int32)
        )
        os.dup2(null_fd, STDOUT) and os.dup2(null_fd, STDERR)
        # this is all wrong below
        in_tour = np.array(
            [
                i - sum(1 if i > rem_point else 0 for rem_point in removed_points)
                for i in parent_solution
                if i not in removed_points
            ]
        )
        solution = solver.solve(in_tour=in_tour, verbose=False, random_seed=42)
        os.dup2(saved_fd, STDOUT) and os.dup2(error_fd, STDERR)

        # lib_problem.save("solution area/temp.tsp")
        # os.dup2(null_fd, STDOUT) and os.dup2(null_fd, STDERR)
        # solver = TSPSolver.from_tspfile("solution area/temp.tsp")
        # solution = solver.solve(verbose=False, random_seed=42)
        # os.dup2(saved_fd, STDOUT) and os.dup2(error_fd, STDERR)

        assert solution.found_tour, "Concorde did not find a tour for the subproblem"
        assert solution.success, "Concorde did not optimally solve the subproblem"

        lib_problem.tours = [
            [x + 1 for x in list(solution.tour)],
        ]
        lib_problem.comment = (
            f"Concorde optimal: {solution.optimal_value} {lib_problem.comment}"
        )

        lib_problem.save(f"DATA_GEN_{NUM_POINTS}{DISTRIB_CODE}/{name}/{sub_name}.tsp")

    if i % 50 == 0:
        print(i)

print(
    f"Generated {TAKE}x{NUM_POINTS} points from {DISTRIB_CODE} distribution in {time.time() - t:.2f} seconds"
)
