from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import List, Tuple, Optional


Point3D = Tuple[float, float, float, float]


def dist(a: Point3D, b: Point3D) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    di = a[3] - b[3]
    return sqrt(dx * dx + dy * dy + dz * dz + di * di)


def solve_tsp_3d(points: List[Point3D], start: int = 0) -> Tuple[List[int], float]:
    """
    Exact TSP solver (minimum Hamiltonian cycle) for small n using Held–Karp DP.

    Args:
        points: list of (x, y, z, w) coordinates; node id is its index in this list.
        start: starting node id (default 0). Tour is returned starting at this node.

    Returns:
        (tour, length) where tour is a list of node ids in visit order including
        the start again at the end, e.g. [0, 3, 2, 1, 0].
    """
    n = len(points)
    if n < 2:
        return ([start, start], 0.0)
    if not (0 <= start < n):
        raise ValueError("start must be a valid node index")

    # Precompute distances
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            D[i][j] = dist(points[i], points[j])

    # Remap nodes so that `start` becomes 0 for convenience
    # inv_map[new_id] = original_id
    # map[original_id] = new_id
    inv_map = [start] + [i for i in range(n) if i != start]
    fwd_map = {orig: new for new, orig in enumerate(inv_map)}

    # D2 is distance matrix in remapped index space
    D2 = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            D2[i][j] = D[inv_map[i]][inv_map[j]]

    # DP[mask][j] = best cost to start at 0, visit mask (over nodes 1..n-1), and end at j
    # where j in 1..n-1 and j is included in mask.
    # mask ranges over subsets of {1..n-1}, represented on (n-1) bits.
    size = 1 << (n - 1)
    INF = float("inf")
    dp = [[INF] * n for _ in range(size)]
    parent: List[List[Optional[int]]] = [[None] * n for _ in range(size)]

    # Initialize: paths that go directly from start (0) to j
    for j in range(1, n):
        mask = 1 << (j - 1)
        dp[mask][j] = D2[0][j]
        parent[mask][j] = 0

    # Fill DP
    for mask in range(size):
        for j in range(1, n):
            if not (mask & (1 << (j - 1))):
                continue
            prev_mask = mask ^ (1 << (j - 1))
            if prev_mask == 0:
                continue
            best = dp[mask][j]
            best_k = parent[mask][j]
            # Try coming to j from k
            for k in range(1, n):
                if not (prev_mask & (1 << (k - 1))):
                    continue
                cand = dp[prev_mask][k] + D2[k][j]
                if cand < best:
                    best = cand
                    best_k = k
            dp[mask][j] = best
            parent[mask][j] = best_k

    # Close the tour: return to start (0)
    full_mask = size - 1
    best_cost = INF
    best_last = None
    for j in range(1, n):
        cand = dp[full_mask][j] + D2[j][0]
        if cand < best_cost:
            best_cost = cand
            best_last = j

    assert best_last is not None

    # Reconstruct tour in remapped ids
    tour_remap = [0]
    mask = full_mask
    j = best_last
    order_rev = []
    while j != 0:
        order_rev.append(j)
        pj = parent[mask][j]
        assert pj is not None
        mask ^= 1 << (j - 1)
        j = pj

    tour_remap.extend(reversed(order_rev))
    tour_remap.append(0)

    # Convert back to original node ids
    tour = [inv_map[i] for i in tour_remap]
    return tour, best_cost


def main() -> None:
    # This confirms that all entrance / exit out of just A is possible
    points = [
        (0.0, 0.0, 0.0, 0.0),    # node 0 A
        (0.0, 1.0, 0.0, 0.0),    # node 1
        (-1.0, 0.0, 0.0, 0.0),   # node 2
        (-1.0, 1.0, 0.0, 0.0),   # node 3

        (-15.0, 0.0, 0.0, 0.0),  # node 4 D
        (-15.0, 1.0, 0.0, 0.0),  # node 5

        (40.0, 0.0, 0.0, 0.0),   # node 6 E

        (-0.5, 0.0, 1.1, 0.0),   # node 7 B
    ]

    tour, length = solve_tsp_3d(points, start=0)
    print("Optimal tour (node ids):", " -> ".join(map(str, tour)))
    print("Total length:", length)

# C has 4 exits and 2 edges
# A has 6 exits

if __name__ == "__main__":
    main()