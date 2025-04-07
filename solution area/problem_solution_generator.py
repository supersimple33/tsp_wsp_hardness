# Create a bunch of problems following some probability distribution
# Then solve the problem and save its solution
# Then generate n subproblems by removing each point and solve those
# to identify which points are most responsible for increasing the size of the solution

import os
import math
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
NUM_POINTS = 30
START_INDEX = 0
TAKE = 3
DISTRIB_CODE = "p0.33"
# DISTRIB_CODE = "u"
EXIST_OK = True
NUM_REMOVED = 3


def power_subset(ss, k=None):
    """Generate all subsets of a set with size k."""
    if k is None:
        k = len(ss)
    return chain(*map(lambda x: combinations(ss, x), range(1, k)))


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

    # BEGIN INITIAL SOLUTION

    dist_matrix = np.sqrt(
        np.sum(
            (np_points[:, np.newaxis, :] - np_points[np.newaxis, :, :]) ** 2, axis=-1
        )
    )
    utri = dist_matrix[np.triu_indices(NUM_POINTS, k=1)].astype(np.int32)
    solver = TSPSolver.from_upper_tri(shape=NUM_POINTS, edges=utri)
    solution = solver.solve(verbose=False, random_seed=42)

    assert solution.success  # Check that the solution is optimal
    assert solution.found_tour  # DEBUG: If this fails a different seed should be tried

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

        sub_name = f"{name}_" + "_".join([str(x) for x in removed_points])
        mask = np.ones(NUM_POINTS, dtype=bool)
        mask[list(removed_points)] = False
        sub_dist_matrix = dist_matrix[np.ix_(mask, mask)]

        lib_problem = tsplib95.models.StandardProblem(
            name=sub_name,
            comment=f"Removed points {removed_points} from {NUM_POINTS} points generated from {DISTRIB_CODE} distribution",
            type="TSP",
            dimension=NUM_POINTS - len(removed_points),  # CHECK TYPES
            edge_weight_type="EUC_2D",
            node_coords={
                j + 1: (p.x, p.y)
                for j, p in enumerate(points)
                if j not in removed_points
            },
        )

        # BEGIN INITIAL SOLUTION
        utri = sub_dist_matrix[np.triu_indices(len(sub_dist_matrix), k=1)]
        solver = TSPSolver.from_upper_tri(
            shape=NUM_POINTS - len(removed_points), edges=utri.astype(np.int32)
        )
        solution = solver.solve(verbose=False, random_seed=42)

        assert solution.success
        assert solution.found_tour

        lib_problem.tours = [
            [x + 1 for x in list(solution.tour)],
        ]
        lib_problem.comment = (
            f"Concorde optimal: {solution.optimal_value} {lib_problem.comment}"
        )

        lib_problem.save(f"DATA_GEN_{NUM_POINTS}{DISTRIB_CODE}/{name}/{sub_name}.tsp")
