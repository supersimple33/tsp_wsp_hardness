import numpy as np
import numba as nb

from .helpers import _euclidean


@nb.njit(inline="always")
def _reverse_segment(tour: np.ndarray, start: int, end: int) -> None:
    while start < end:
        tmp = tour[start]
        tour[start] = tour[end]
        tour[end] = tmp
        start += 1
        end -= 1


@nb.njit(inline="always")
def _is_mutable_edge(tour: np.ndarray, idx: int, in_ab: np.ndarray) -> bool:
    n = tour.size
    j = idx + 1
    if j == n:
        j = 0
    return bool(in_ab[tour[idx]] and in_ab[tour[j]])


@nb.njit(inline="always")
def _edge_delta(
    tour: np.ndarray,
    points: np.ndarray,
    i: int,
    i2: int,
    j: int,
    j2: int,
) -> float:
    a = tour[i]
    b = tour[i2]
    c = tour[j]
    d = tour[j2]
    return (_euclidean(points, a, c) + _euclidean(points, b, d)) - (
        _euclidean(points, a, b) + _euclidean(points, c, d)
    )


@nb.njit(cache=True)
def _collect_mutable_edges(tour: np.ndarray, in_ab: np.ndarray) -> np.ndarray:
    n = tour.size
    idxs = np.empty(n, dtype=np.int64)
    count = 0
    for i in range(n):
        if _is_mutable_edge(tour, i, in_ab):
            idxs[count] = i
            count += 1
    return idxs[:count]


@nb.njit(inline="always")
def _valid_edge_pair(i: int, j: int, n: int) -> bool:
    if j <= i + 1:
        return False
    if i == 0 and j == n - 1:
        return False
    return True


@nb.njit(cache=True)
def _find_first_improving_move(
    tour: np.ndarray,
    in_ab: np.ndarray,
    points: np.ndarray,
) -> tuple[int, int]:
    n = tour.size
    mutable_edges = _collect_mutable_edges(tour, in_ab)

    for p in range(mutable_edges.size):
        i = mutable_edges[p]

        i2 = i + 1
        if i2 == n:
            i2 = 0

        for q in range(p + 1, mutable_edges.size):
            j = mutable_edges[q]
            if not _valid_edge_pair(i, j, n):
                continue

            j2 = j + 1
            if j2 == n:
                j2 = 0

            if _edge_delta(tour, points, i, i2, j, j2) < -1e-12:
                return i2, j

    return -1, -1


@nb.njit(cache=True)
def _constrained_two_opt(
    tour: np.ndarray,
    in_ab: np.ndarray,
    points: np.ndarray,
    max_moves: int,
) -> np.ndarray:
    moves = 0

    while moves < max_moves:
        start, end = _find_first_improving_move(tour, in_ab, points)
        if start < 0:
            break

        _reverse_segment(tour, start, end)
        moves += 1

    return tour


@nb.njit(cache=True)
def _exact_block_fixed_endpoints(
    block_nodes: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    k = block_nodes.size
    if k <= 2:
        return block_nodes.copy()

    # Endpoints are fixed to preserve outside-incident edges.
    # We optimize only the interior permutation.
    interior = block_nodes[1 : k - 1]
    t = interior.size
    if t == 0:
        return block_nodes.copy()

    full = 1 << t
    inf = 1e308
    dp = np.full((full, t), inf, dtype=np.float64)
    parent = np.full((full, t), -1, dtype=np.int64)

    start_node = block_nodes[0]
    end_node = block_nodes[k - 1]

    for i in range(t):
        dp[1 << i, i] = _euclidean(points, start_node, interior[i])

    for mask in range(full):
        for last in range(t):
            if mask & (1 << last):
                _path_relax_path_state(dp, parent, mask, last, interior, points, inf)

    all_mask = full - 1
    best_last = -1
    best_cost = inf
    for last in range(t):
        cand = dp[all_mask, last] + _euclidean(points, interior[last], end_node)
        if cand < best_cost:
            best_cost = cand
            best_last = last

    order = np.empty(k, dtype=block_nodes.dtype)
    order[0] = start_node
    order[k - 1] = end_node

    mask = all_mask
    last = best_last
    for pos in range(k - 2, 0, -1):
        order[pos] = interior[last]
        prev = parent[mask, last]
        mask ^= 1 << last
        last = prev

    return order


@nb.njit(cache=True)
def _exact_cycle_order(
    cycle_nodes: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    k = cycle_nodes.size
    if k <= 1:
        return cycle_nodes.copy()

    full = 1 << k
    inf = 1e308
    dp = np.full((full, k), inf, dtype=np.float64)
    parent = np.full((full, k), -1, dtype=np.int64)

    dp[1, 0] = 0.0
    for mask in range(full):
        if not (mask & 1):
            continue
        for last in range(k):
            if mask & (1 << last):
                _cycle_relax_cycle_state(dp, parent, mask, last, cycle_nodes, points, inf)

    all_mask = full - 1
    best_last = 0
    best_cost = inf
    for last in range(1, k):
        cand = dp[all_mask, last] + _euclidean(points, cycle_nodes[last], cycle_nodes[0])
        if cand < best_cost:
            best_cost = cand
            best_last = last

    order = np.empty(k, dtype=cycle_nodes.dtype)
    mask = all_mask
    last = best_last
    for pos in range(k - 1, 0, -1):
        order[pos] = cycle_nodes[last]
        prev = parent[mask, last]
        mask ^= 1 << last
        last = prev
    order[0] = cycle_nodes[0]
    return order


@nb.njit(inline="always")
def _path_relax_path_state(
    dp: np.ndarray,
    parent: np.ndarray,
    mask: int,
    last: int,
    seq_nodes: np.ndarray,
    points: np.ndarray,
    inf: float,
) -> None:
    cur = dp[mask, last]
    if cur >= inf:
        return
    k = seq_nodes.size
    for nxt in range(k):
        bit = 1 << nxt
        if mask & bit:
            continue
        new_mask = mask | bit
        cand = cur + _euclidean(points, seq_nodes[last], seq_nodes[nxt])
        if cand < dp[new_mask, nxt]:
            dp[new_mask, nxt] = cand
            parent[new_mask, nxt] = last


@nb.njit(inline="always")
def _cycle_relax_cycle_state(
    dp: np.ndarray,
    parent: np.ndarray,
    mask: int,
    last: int,
    cycle_nodes: np.ndarray,
    points: np.ndarray,
    inf: float,
) -> None:
    cur = dp[mask, last]
    if cur >= inf:
        return
    k = cycle_nodes.size
    for nxt in range(k):
        bit = 1 << nxt
        if mask & bit:
            continue
        new_mask = mask | bit
        cand = cur + _euclidean(points, cycle_nodes[last], cycle_nodes[nxt])
        if cand < dp[new_mask, nxt]:
            dp[new_mask, nxt] = cand
            parent[new_mask, nxt] = last


def _build_in_ab_mask(n_points: int, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    in_ab = np.zeros(n_points, dtype=np.bool_)
    in_ab[A] = True
    in_ab[B] = True
    return in_ab


def _rotate_tour(tour: np.ndarray, start_idx: int) -> np.ndarray:
    if start_idx == 0:
        return tour.copy()
    return np.concatenate((tour[start_idx:], tour[:start_idx]))


def _unrotate_tour(rotated: np.ndarray, start_idx: int) -> np.ndarray:
    if start_idx == 0:
        return rotated.copy()
    n = rotated.size
    shift = n - start_idx
    return np.concatenate((rotated[shift:], rotated[:shift]))


def _exact_repair_blocks(tour: np.ndarray, in_ab: np.ndarray, points: np.ndarray) -> np.ndarray:
    n = tour.size
    if n == 0:
        return tour.copy()

    if np.count_nonzero(in_ab[tour]) == n:
        return _exact_cycle_order(tour.astype(np.int64, copy=False), points).astype(tour.dtype, copy=False)

    first_outside = -1
    for idx in range(n):
        if not in_ab[tour[idx]]:
            first_outside = idx
            break
    if first_outside < 0:
        return _exact_cycle_order(tour.astype(np.int64, copy=False), points).astype(tour.dtype, copy=False)

    rotated = _rotate_tour(tour, first_outside).astype(np.int64, copy=True)

    idx = 0
    while idx < n:
        if not in_ab[rotated[idx]]:
            idx += 1
            continue
        start = idx
        while idx + 1 < n and in_ab[rotated[idx + 1]]:
            idx += 1
        end = idx

        block_nodes = rotated[start : end + 1]
        repaired = _exact_block_fixed_endpoints(block_nodes, points)
        rotated[start : end + 1] = repaired
        idx += 1

    repaired_tour = _unrotate_tour(rotated, first_outside)
    return repaired_tour.astype(tour.dtype, copy=False)


def repair_tour_euc(
    tour: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """
    Repair a Euclidean TSP tour by re-optimizing only edges fully internal to A union B.

    Constraints enforced:
    - Any edge touching a node outside A union B is never removed.
    - Only A/A, A/B, B/A, and B/B edges may be disconnected/reconnected.
    """
    if tour.ndim != 1:
        raise ValueError("tour must be a 1D array of node ids")
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of coordinates")
    if tour.size == 0:
        return tour.copy()

    n_points = points.shape[0]
    if np.any(tour < 0) or np.any(tour >= n_points):
        raise ValueError("tour contains node ids outside points")

    in_ab = _build_in_ab_mask(n_points, A, B)
    mutable_count = np.count_nonzero(in_ab[tour])

    # Fewer than 2 points in A union B means there is no mutable edge.
    if mutable_count < 2:
        return tour.copy()

    if mutable_count <= 14:
        return _exact_repair_blocks(tour.astype(np.int64, copy=False), in_ab, np.ascontiguousarray(points)).astype(tour.dtype, copy=False)

    work_tour = np.ascontiguousarray(tour.astype(np.int64, copy=True))
    work_points = np.ascontiguousarray(points)
    work_mask = np.ascontiguousarray(in_ab)

    # Conservative cap to avoid pathological runtimes.
    max_moves = max(8, 4 * work_tour.size)
    repaired = _constrained_two_opt(work_tour, work_mask, work_points, max_moves)

    if repaired.dtype != tour.dtype:
        repaired = repaired.astype(tour.dtype, copy=False)
    return repaired
