import os
import sys
import ctypes
import threading
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Iterator

import numpy as np
from numba import njit

from concorde.tsp import TSPSolver

STDOUT = 1
STDERR = 2
_libc = ctypes.CDLL(None)
_OUTPUT_LOCK = threading.Lock()


def _flush_all_streams() -> None:
    # Flush both Python-level and C stdio buffers before fd swaps.
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    try:
        _libc.fflush(None)
    except Exception:
        pass


@contextmanager
def _silence_process_output() -> Iterator[None]:
    # Redirect both Python streams and raw FDs to suppress Cython/C output in notebooks.
    with _OUTPUT_LOCK:
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            saved_stdout_fd = os.dup(STDOUT)
            saved_stderr_fd = os.dup(STDERR)
            try:
                _flush_all_streams()
                os.dup2(devnull.fileno(), STDOUT)
                os.dup2(devnull.fileno(), STDERR)
                yield
            finally:
                _flush_all_streams()
                os.dup2(saved_stdout_fd, STDOUT)
                os.dup2(saved_stderr_fd, STDERR)
                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)

def roll_to_node(tour: np.ndarray, node: int) -> np.ndarray:
    """Roll the tour so that it starts with the given node"""
    return np.roll(tour, -np.nonzero(tour == node)[0][0])

@njit(inline="always", cache= True, nogil=True)
def _euclidean(points: np.ndarray, u: int, v: int) -> float:
    acc = (points[u, 0] - points[v, 0]) ** 2
    for k in range(1, points.shape[1]):
        dv = points[u, k] - points[v, k]
        acc += dv * dv
    return np.round(np.sqrt(acc), decimals=0)

@njit(inline="always", cache=True, nogil=True)
def build_dist_matrix(points: np.ndarray) -> np.ndarray:
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
def calc_tour_len_euc(points: np.ndarray, tour: np.ndarray) -> float:
    """Calculate the length of a tour given the points and the tour order. Done in just numpy"""
    acc = _euclidean(points, tour[-1], tour[0])
    for i in range(len(tour) - 1):
        u = tour[i]
        v = tour[i + 1]
        acc += _euclidean(points, u, v)
    return acc

@njit(inline="always", cache=True, nogil=True)
def valid_tour(tour: np.ndarray, n: int) -> bool:
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

def build_concorde_solver(dist_matrix: np.ndarray) -> TSPSolver:
    n = dist_matrix.shape[0]
    ltri = np.round(dist_matrix[np.tril_indices(n, k=-1)]).astype(np.int32)
    return TSPSolver.from_lower_tri(shape=n, edges=ltri)


def solve_concorde_once(solver: TSPSolver, random_seed: int, dtype: type[np.integer] = np.int32) -> tuple[np.ndarray, int]:
    with _silence_process_output():
        sol = solver.solve(verbose=False, random_seed=random_seed)
        found_tour = sol.found_tour
        success = sol.success
        tour, opt_val = sol.tour, sol.optimal_value

    assert found_tour, "Concorde did not find a tour"
    assert success, "Concorde did not certify optimality"
    return np.array(tour, dtype=dtype), int(opt_val)