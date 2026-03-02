from __future__ import annotations

from math import sqrt
from typing import Dict, List, Tuple, Optional, Sequence

Point3D = Tuple[float, float, float, float]


def dist(a: Point3D, b: Point3D) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    di = a[3] - b[3]
    return sqrt(dx * dx + dy * dy + dz * dz + di * di)


def solve_tsp_3d(points: Sequence[Point3D], start: int = 0) -> Tuple[List[int], float]:
    """
    Exact TSP solver (minimum Hamiltonian cycle) for small n using Held–Karp DP.

    Args:
        points: list/sequence of (x, y, z, w) coordinates; node id is its index.
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
    inv_map = [start] + [i for i in range(n) if i != start]

    # D2 is distance matrix in remapped index space
    D2 = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            D2[i][j] = D[inv_map[i]][inv_map[j]]

    # DP over subsets of {1..n-1}
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
    best_last: Optional[int] = None
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
    order_rev: List[int] = []
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


def solve_tsp_named(
    named_points: Dict[str, Point3D],
    start: str | None = None,
) -> Tuple[List[str], float]:
    """
    Wrapper around solve_tsp_3d that accepts {name: point} and returns a tour of names.

    Ordering notes:
      - Python dicts preserve insertion order, so we keep that order by default.
      - If you want a deterministic order independent of insertion, replace the
        `names = list(named_points.keys())` line with `names = sorted(named_points)`.
    """
    if not named_points:
        raise ValueError("named_points must be non-empty")

    names = list(named_points.keys())
    points = [named_points[n] for n in names]

    if start is None:
        start_idx = 0
    else:
        try:
            start_idx = names.index(start)
        except ValueError as e:
            raise ValueError(f"start name {start!r} is not a key in named_points") from e

    tour_idx, cost = solve_tsp_3d(points, start=start_idx)
    tour_names = [names[i] for i in tour_idx]
    return tour_names, cost

def main() -> None:
    named_points: Dict[str, Point3D] = {
        # Achieves a tour with Abs Flux = 6 for A
        "A1": (0.0, 0.0, 0.0, 0.0),
        "A2": (0.0, 0.0, 0.5, 0.0),
        "A3": (0.0, 0.1, 0.0, 0.0),
        "A4": (0.0, 0.1, 0.5, -0.5),
        "B": (-2.0, 1.0, 0.0, -1.0),
        "C": (-2.0, -1.0, 0.0, -1.0),
        "D": (2.0, 0.0, 0.0, -1.0),
        "E": (2.0, 0.0, 0.6, -1.0),
        "F": (0.0, 0.1, 0.5, 0.75),
    }

    tour, length = solve_tsp_named(named_points, start="D")
    print("Tour:", " -> ".join(tour))
    print("Length:", length)


if __name__ == "__main__":
    main()


## C set of 4 exits
#(0.0, 0.0, 0.0, 0.0),    # node 0 A
#(0.0, 0.0, 0.5, 0.0),    # node 1 A
#(-2.0, 1.0, 0.0, 0.0),    # node 2
#(-2.0, -1.0, 0.0, 0.0),    # node 3
#(2.0, 0.0, 0.0, 0.0),    # node 4 B
#(2.0, 0.0, 0.6, 0.0),    # node 5 C


### C set of 4
#(0.0, 0.0, 0.0, 0.0),    # node 0 A
#(1.0, 0.0, 0.0, 0.0),    # node 1
#(0.0, 1.5, 0.0, 0.0),   # node 2
#(1.0, 1.5, 0.0, 0.0),   # node 3

#(-15.0, 0.0, 0.0, 0.0),  # node 4 D
#(-15.0, 1.5, 0.0, 0.0),  # node 5

#(40.0, 0.0, 0.0, 0.0),   # node 6 E

#(0.5, 0.75, 1.7, 0.0),   # node 7 B




#(0.0,0.0,0,0),(1.0,0.0,0,0),(0.0,1.5,0,0),(1.0,1.5,0,0),
#(-15.0,0.0,0,0),(-15.0,1.5,0,0),
#(40.0,0.0,0,0),(40.0,0.5,0,0),
#(200.0,0.0,0,0),(200.0,0.5,0,0)