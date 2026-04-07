from typing import Literal

import numpy as np
from numba import njit

from concorde.tsp import TSPSolver


type ListOfBool = np.ndarray[tuple[int], np.dtype[np.bool_]]
type ListOfInt = np.ndarray[tuple[int], np.dtype[np.integer]]
#type ListOfEnterExit = np.ndarray[tuple[int], np.dtype[np.void]] # TODO: update to better numpy typing
type ListOfEnterExit = np.ndarray[tuple[int, Literal[2]], np.dtype[np.integer]]
type ListOfPoints = np.ndarray[tuple[int, ...], np.dtype[np.floating]]
type DistMatrix = np.ndarray[tuple[int, int], np.dtype[np.floating]] # TODO: use numpydantic here


def roll_to_node(tour: ListOfInt, node: int) -> ListOfInt:
    """Roll the tour so that it starts with the given node"""
    return np.roll(tour, -np.nonzero(tour == node)[0][0])

@njit(inline="always", cache= True, nogil=True)
def _euclidean(points: ListOfPoints, u: int, v: int) -> float:
    acc = (points[u, 0] - points[v, 0]) ** 2
    for k in range(1, points.shape[1]):
        dv = points[u, k] - points[v, k]
        acc += dv * dv
    return np.round(np.sqrt(acc), decimals=0)

@njit(inline="always", cache=True, nogil=True)
def build_dist_matrix(points: ListOfPoints) -> DistMatrix:
    """Build a distance matrix from the points using Euclidean distance"""
    n = points.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean(points, i, j)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix

@njit(inline="always", cache=True, nogil=True)
def calc_tour_len_euc(points: ListOfPoints, tour: ListOfInt) -> float:
    """Calculate the length of a tour given the points and the tour order. Done in just numpy"""
    acc = _euclidean(points, tour[-1], tour[0])
    for i in range(len(tour) - 1):
        u = tour[i]
        v = tour[i + 1]
        acc += _euclidean(points, u, v)
    return acc

@njit(inline="always", cache=True, nogil=True)
def valid_tour(tour: ListOfInt, n: int) -> bool:
    """Check if a tour is valid (contains all nodes exactly once)"""
    if tour.size != n:
        return False
    seen = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        node = tour[i]
        if node < 0 or node >= n or seen[node]:
            return False
        seen[node] = True
    return True

def build_concorde_solver(dist_matrix: DistMatrix) -> TSPSolver:
    n = dist_matrix.shape[0]
    ltri = np.round(dist_matrix[np.tril_indices(n, k=-1)]).astype(np.int32)
    return TSPSolver.from_lower_tri(shape=n, edges=ltri)


def solve_concorde_once(solver: TSPSolver, random_seed: int, dtype: type[np.integer] = np.int32, timeout: int = -1) -> tuple[ListOfInt, int]:
    sol = solver.solve(verbose=False, random_seed=random_seed, time_bound=timeout)

    if sol.hit_timebound:
        raise TimeoutError("Concorde hit time bound without finding a solution")
    assert sol.found_tour, "Concorde did not find a tour"
    assert sol.success, "Concorde did not certify optimality"
    return np.array(sol.tour, dtype=dtype), int(sol.optimal_value)