from typing import Literal
import itertools
from collections import namedtuple

import numpy as np
import numba as nb

from .helpers import _euclidean, calc_tour_len_euc

BRUTE_FORCE_THRESHOLD = 10  # If there are fewer than this many mutable edges, just brute force all possibilities

type ListOfBool = np.ndarray[tuple[int], np.dtype[np.bool_]]
type ListOfInt = np.ndarray[tuple[int], np.dtype[np.signedinteger]]
type ListOfEnterExit = np.ndarray[tuple[int], np.dtype[np.void]] # TODO: update to better numpy typing
type ListOfPoints = np.ndarray[tuple[int, ...], np.dtype[np.floating]]

def signed_cyclic_permutations(n):
    nums = list(range(1, n))
    
    for perm in itertools.permutations(nums):
        for signs in itertools.product((-1, 1), repeat=n-1):
            yield (0,) + tuple(s * x for s, x in zip(signs, perm))

def _entrance_exit_masks(
    tour: ListOfInt, A: ListOfInt, B: ListOfInt
) -> tuple[ListOfBool, ListOfBool]:
    """Return boolean masks of which nodes in the tour are entrances and exits to !(A union B)"""
    n = tour.size
    in_a = np.isin(tour, A, assume_unique=True)
    in_b = np.isin(tour, B, assume_unique=True)
    in_ab = in_a | in_b


    # A node is an exit if it is outside A union B and the next node in the tour is inside A union B
    exit_mask = np.empty(n, dtype=np.bool_)
    exit_mask[:-1] = ~in_ab[:-1] & in_ab[1:]
    exit_mask[-1] = ~in_ab[-1] & in_ab[0]
    # A node is an entrance if the node before it is inside A union B and the node itself is outside A union B
    entrance_mask = np.empty(n, dtype=np.bool_)
    entrance_mask[1:] = in_ab[:-1] & ~in_ab[1:]
    entrance_mask[0] = in_ab[-1] & ~in_ab[0]

    return entrance_mask, exit_mask

@nb.njit(inline="always")
def _entrance_exit_pairs(
    tour: ListOfInt, entrance_indices: ListOfInt, exit_indices: ListOfInt
) -> ListOfEnterExit:
    """Return a list of (exit_index, entrance_index) pairs in the order they appear in the tour"""
    assert exit_indices.size == entrance_indices.size, "Number of exit edges must equal number of entrance edges"

    starts_entrance = exit_indices[0] < entrance_indices[0]
    entrance_exit_pairs = np.empty(exit_indices.size, dtype=[("entrance", tour.dtype), ("exit", tour.dtype)])
    if starts_entrance:
        entrance_exit_pairs["entrance"] = entrance_indices
        entrance_exit_pairs["exit"] = exit_indices
    else:
        entrance_exit_pairs["entrance"] = entrance_indices
        entrance_exit_pairs[:-1]["exit"] = exit_indices[1:]
        entrance_exit_pairs[-1]["exit"] = exit_indices[0]
    return entrance_exit_pairs

def _dp_for_endpoints(paths: list[tuple[int, int, list]], AB: ListOfInt, points: ListOfPoints) -> float:
    """Fills in paths with the points of AB so that it is the optimal solution and returns the total cost of the reconnection strategy"""
    M = len(AB)
    K = len(paths)

    active_nodes = np.concatenate((AB, [p[0] for p in paths]))
    L = len(active_nodes)

    # map from node id of start to index in dp table
    start_to_idx = {p[0]: M + i for i, p in enumerate(paths)}

    # main loop
    dp = np.full((1 << M, L, K), np.inf, dtype=points.dtype)

    # reverse pointers # this could also just be a structured array
    parent_mask = np.full((1 << M, L, K), -1, dtype=AB.dtype)
    parent_uidx = np.full((1 << M, L, K), -1, dtype=AB.dtype)
    parent_vidx = np.full((1 << M, L, K), -1, dtype=AB.dtype)

    # initialize dp
    dp[0, start_to_idx[paths[0][0]], 0] = 0.0
    for i in range(K): # iterate over paths in order, we must finish path i before starting path i+1
        valid_uidxs = list(range(M)) + [start_to_idx[paths[i][0]]] # TODO: this doesnt need to be in memory

        for mask in range(1 << M):
            for u_idx in valid_uidxs:
                cost_u = dp[mask, u_idx, i]
                if np.isinf(cost_u):
                    continue # REVIEW: why?

                u_orig = active_nodes[u_idx]

                # Option 1: continue path i visiting an unvisited node in AB  
                for v_bit in range(M):
                    if mask & (1 << v_bit):
                        continue
                    next_mask = mask | (1 << v_bit)
                    v_orig = AB[v_bit]
                    new_cost = cost_u + _euclidean(points, u_orig, v_orig)

                    if new_cost < dp[next_mask, v_bit, i]:
                        dp[next_mask, v_bit, i] = new_cost
                        parent_mask[next_mask, v_bit, i] = mask
                        parent_uidx[next_mask, v_bit, i] = u_idx
                        parent_vidx[next_mask, v_bit, i] = i

                # Option 2: end path i and jump to start of path i+1
                if i < K - 1:
                    new_cost = cost_u + _euclidean(points, u_orig, paths[i][1])
                    next_u_idx = start_to_idx[paths[i + 1][0]]

                    if new_cost < dp[mask, next_u_idx, i + 1]:
                        dp[mask, next_u_idx, i + 1] = new_cost
                        parent_mask[mask, next_u_idx, i + 1] = mask
                        parent_uidx[mask, next_u_idx, i + 1] = u_idx
                        parent_vidx[mask, next_u_idx, i + 1] = i

    # reconstruct solution
    final_mask = (1 << M) - 1
    best_cost = np.inf
    best_last_uidx = -1

    valid_last_uidxs = list(range(M)) + [start_to_idx[paths[K - 1][0]]]

    for u_idx in valid_last_uidxs:
        cost_u = dp[final_mask, u_idx, K - 1]
        if np.isinf(cost_u):
            continue

        u_orig = active_nodes[u_idx]
        total_cost = cost_u + _euclidean(points, u_orig, paths[K - 1][1])

        if total_cost < best_cost:
            best_cost = total_cost
            best_last_uidx = u_idx

    if np.isinf(best_cost):
        raise ValueError("No valid reconnection strategy found")
    
    curr_mask = final_mask
    curr_uidx = best_last_uidx
    curr_i = K - 1

    paths[curr_i][2].append(paths[curr_i][1]) # add entrance node to path

    while curr_i >= 0 and parent_uidx[curr_mask, curr_uidx, curr_i] != -1:
        curr_u_orig = active_nodes[curr_uidx]
        paths[curr_i][2].append(curr_u_orig)

        p_mask = parent_mask[curr_mask, curr_uidx, curr_i]
        p_uidx = parent_uidx[curr_mask, curr_uidx, curr_i]
        p_i = parent_vidx[curr_mask, curr_uidx, curr_i]

        if p_i == curr_i:
            paths[curr_i][2].append(paths[curr_i][1])

        curr_mask, curr_uidx, curr_i = p_mask, p_uidx, p_i

    paths[0][2].append(paths[0][0]) # add entrance node to first path

    for i in range(K):
        paths[i][2].reverse() # reverse each path to be in the correct order

    return best_cost


def _brute_force_repair(tour: ListOfInt, entrance_exit_pairs: ListOfEnterExit, AB: ListOfInt, points: ListOfPoints) -> ListOfInt:
    # iterate over all endpoint combinations then do a dp style search to find the best reconnection strategy
    # (ab, cd, ef) (ab, cd fe), (ab, dc, ef) (ab, dc fe) (ab, ef, cd) (ab, ef, dc) (ab, fe, cd) (ab, fe, dc)
    best_paths = None
    best_cost = float("inf")
    for perm in signed_cyclic_permutations(len(entrance_exit_pairs)): # TODO: parallelize
        # entrance, exit, reversed, internal edges that have been collected
        paths: list[tuple[int, int, list]] = [(0, 0, [])] * len(entrance_exit_pairs)
        for i in perm:
            last_path_ind = (i - 1) % len(entrance_exit_pairs)

            entrance_node, exit_node = entrance_exit_pairs[abs(i)]
            if i < 0:
                entrance_node, exit_node = exit_node, entrance_node

            paths[i] = (exit_node, paths[i][1], paths[i][2])
            paths[last_path_ind] = (paths[last_path_ind][0], entrance_node, paths[last_path_ind][2])
        
        min_cost = _dp_for_endpoints(paths, AB, points)
        if min_cost < best_cost:
            best_paths = paths
            best_cost = min_cost

    if best_paths is None:
        raise ValueError("entrance exit pairs was empty")
    

def repair_tour_euc(
    tour: ListOfInt,
    A: ListOfInt,
    B: ListOfInt,
    points: ListOfPoints,
) -> np.ndarray:
    r"""
    Repair a Euclidean TSP tour by re-optimizing only edges fully internal to :math:`A \cup B`
    Note :math:`A \cap B = \emptyset`

    Constraints enforced:
    - Any edge touching a node outside A union B is never removed.
    - Only A/A, A/B, B/A, and B/B edges may be disconnected/reconnected.
    """
    if tour.ndim != 1:
        raise ValueError("tour must be a 1D array of node ids")
    if points.ndim < 1:
        raise ValueError("points must be an array with shape (n_points, ...)")
    if tour.size == 0:
        return tour.copy()

    n_points = points.shape[0]
    if np.any(tour < 0) or np.any(tour >= n_points):
        raise ValueError("tour contains node ids outside points")

    entrance_mask, exit_mask = _entrance_exit_masks(tour, A, B)

    entrance_indices = np.nonzero(entrance_mask)[0]
    exit_indices = np.nonzero(exit_mask)[0]

    entrance_exit_pairs = _entrance_exit_pairs(tour, entrance_indices, exit_indices)

    if A.size + B.size < 2:
        print("Cannot repair tour: not enough mutable edges")
        return tour.copy()
    elif A.size + B.size < 14:
        print("Few mutable edges, using brute force search")
        return _brute_force_repair(tour, entrance_exit_pairs, A, B, points)
    else:
        print("Many mutable edges, using greedy repair")
        raise NotImplementedError("Greedy repair not implemented yet")
