import numpy as np
import numba as nb

from .helpers import _euclidean

@nb.njit(cache=True, inline="always", boundscheck=False, fastmath=True, nogil=True)
def solve_nn(dist_mat: np.ndarray, start: int = 0) -> np.ndarray:
    """Solves a TSP using the nearest neighbor heuristic."""
    n = dist_mat.shape[0]
    tour = np.empty(n, dtype=np.int64)
    tour[0] = start
    visited = np.zeros(n, dtype=bool)
    visited[start] = True

    for i in range(1, n):
        last_city = tour[i - 1]
        next_city = -1
        min_dist = np.inf
        for j in range(n):
            if not visited[j]:
                dist = dist_mat[last_city, j]
                if dist < min_dist:
                    min_dist = dist
                    next_city = j
        tour[i] = next_city
        visited[next_city] = True

    return tour

@nb.njit(cache=True, inline="always", boundscheck=False, fastmath=True, nogil=True)
def solve_nn_euc(points: np.ndarray, start: int = 0, dtype=np.intp) -> np.ndarray:
    """Solves a TSP using the nearest neighbor heuristic for euclidean points by computing distances on the fly."""
    n = points.shape[0]
    tour = np.empty(n, dtype=dtype)
    tour[0] = start
    visited = np.zeros(n, dtype=bool)
    visited[start] = True

    for i in range(1, n):
        last_city = tour[i - 1]
        next_city = -1
        min_dist = np.inf
        for j in range(n):
            if not visited[j]:
                dist = _euclidean(points, last_city, j)
                if dist < min_dist:
                    min_dist = dist
                    next_city = j
        tour[i] = next_city
        visited[next_city] = True

    return tour
