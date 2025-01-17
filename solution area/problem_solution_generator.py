# Create a bunch of problems following some probability distribution
# Then solve the problem and save its solution
# Then generate n subproblems by removing each point and solve those
# to identify which points are most responsible for increasing the size of the solution

import os

import itertools
import math
import numpy as np
import tsplib95
from concorde.tsp import TSPSolver  # DC

from wsp import tsp, ds

ALPHABET = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]

NUM_POINTS = 50
START_INDEX = 0
TAKE = 50
DISTRIB_CODE = "Uniform"


def get_points(rng: np.random.Generator, num_points: int) -> np.ndarray:
    match DISTRIB_CODE:
        case "Uniform":  # integers
            return rng.integers(0, 10000, size=(num_points, 2)).astype(
                dtype=np.float64
            )  # how do we decide this also typing is a factor
        case "Normal":
            return rng.normal(
                size=(num_points, 2), scale=1000
            )  # TODO: Add scaling here (how should i decide????)
        case _:
            raise ValueError(f"Unknown distribution code: {DISTRIB_CODE}")


ids = ["".join(x) for x in itertools.product(ALPHABET, repeat=2)][
    START_INDEX : START_INDEX + NUM_POINTS
]


# TODO: MULTI-PROCESSING
for id in ids:  # TODO: SIGINT Handling
    # Save the problem
    name = id + f"{NUM_POINTS}{DISTRIB_CODE}"
    rng: np.random.Generator = np.random.default_rng(
        seed=abs(hash(name))
    )  # CHECK NO OVERFLOW HERE
    points = [ds.Point(x, y) for x, y in get_points(rng, NUM_POINTS)]

    # TODO: check if this already exists

    # mainProblem = tsp.TravellingSalesmanProblem[TREE_TYPE](
    #     TREE_TYPE, points, AX, s=S_FACTOR
    # )

    lib_problem = tsplib95.models.StandardProblem(
        name=name,
        comment=f"{NUM_POINTS} points generated from {DISTRIB_CODE} distribution",
        type="TSP",
        dimension=NUM_POINTS,  # CHECK TYPES
        edge_weight_type="EUC_2D",
        node_coords={i + 1: (p.x, p.y) for i, p in enumerate(points)},
    )

    lib_problem.save("solution area/temp.tsp")

    # BEGIN INITIAL SOLUTION

    solver = TSPSolver.from_tspfile("solution area/temp.tsp")
    solution = solver.solve(verbose=False, random_seed=42)

    assert solution.success  # Check that the solution is optimal
    assert solution.found_tour  # DEBUG: If this fails a different seed should be tried

    lib_problem.tours = [
        [x + 1 for x in list(solution.tour)],
    ]

    os.makedirs(f"DATA_GEN_50/{id}")
    lib_problem.save(f"DATA_GEN_50/{id}/{name}.tsp")

    # MARK: - Work on Sub Problems
    for i in range(NUM_POINTS):
        # sub_problem = tsp.TravellingSalesmanProblem[TREE_TYPE](
        #     TREE_TYPE, points[:i] + points[i + 1 :], AX, s=S_FACTOR
        # )

        lib_problem = tsplib95.models.StandardProblem(
            name=name + f"_{i+1}",
            comment=f"Removed point {i+1} from {NUM_POINTS} points generated from {DISTRIB_CODE} distribution",
            type="TSP",
            dimension=NUM_POINTS - 1,  # CHECK TYPES
            edge_weight_type="EUC_2D",
            node_coords={
                j + 1: (p.x, p.y) for j, p in enumerate(points[:i] + points[i + 1 :])
            },
        )

        lib_problem.save("solution area/temp.tsp")

        solver = TSPSolver.from_tspfile("solution area/temp.tsp")
        solution = solver.solve(verbose=False, random_seed=42)

        assert solution.success
        assert solution.found_tour

        lib_problem.tours = [
            [x + 1 for x in list(solution.tour)],
        ]

        lib_problem.save(f"DATA_GEN_50/{id}/{name}_{i+1}.tsp")
