import numpy as np
from numba import njit

@njit(inline="always")
def _euclidean(points: np.ndarray, u: int, v: int) -> float:
    acc = (points[u, 1] - points[v, 1]) ** 2
    for k in range(1, points.shape[1]):
        dv = points[u, k] - points[v, k]
        acc += dv * dv
    return np.sqrt(acc)

@njit(inline="always", cache=True)
def calc_tour_len_euc(points: np.ndarray, tour: np.ndarray) -> float:
    """Calculate the length of a tour given the points and the tour order. Done in just numpy"""
    acc = _euclidean(points, tour[-1], tour[0])
    for i in range(len(tour) - 1):
        u = tour[i]
        v = tour[i + 1]
        acc += _euclidean(points, u, v)
    return acc

@njit(inline="always", cache=True)
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