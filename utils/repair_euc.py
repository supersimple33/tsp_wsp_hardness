from typing import Literal
import itertools
from collections import namedtuple

import numpy as np
import numba as nb

from .helpers import _euclidean, calc_tour_len_euc

BRUTE_FORCE_THRESHOLD = 10  # If there are fewer than this many mutable edges, just brute force all possibilities

Path = namedtuple("Path", ["start", "end", "internal_nodes"])

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

def _unified_dp_repair(entrance_exit_pairs: ListOfEnterExit, AB: ListOfInt, points: ListOfPoints) -> tuple[float, list[Path]]:
    """
    Simultaneously finds the optimal segment sequence, traversal direction, and AB point distribution.
    Returns: (best_cost, list_of_paths)
    """
    K = len(entrance_exit_pairs)
    M = len(AB)
    
    if K == 0:
        raise ValueError("entrance exit pairs was empty")

    L = M + 2 * K  # Total physical nodes in our state space

    # Map indices to their corresponding node IDs:
    # 0 to M-1: AB nodes
    # M to M+2K-1: Segments (A_k is entrance, B_k is exit)
    active_nodes = np.zeros(L, dtype=AB.dtype)
    if M > 0:
        active_nodes[:M] = AB
        
    for k in range(K):
        active_nodes[M + 2*k] = entrance_exit_pairs[k]["entrance"] # A_k
        active_nodes[M + 2*k + 1] = entrance_exit_pairs[k]["exit"]   # B_k

    # Precompute distance matrix to avoid recalculating in inner loops
    dist_matrix = np.zeros((L, L), dtype=points.dtype)
    for i in range(L):
        for j in range(L):
            if i != j:
                dist_matrix[i, j] = _euclidean(points, active_nodes[i], active_nodes[j])

    # DP dimensions: (mask_AB, mask_K, current_node)
    # mask_K has K-1 bits (since we always start at segment 0, we only track segments 1 to K-1)
    shape = (1 << M, 1 << max(0, K - 1), L)
    
    dp = np.full(shape, np.inf, dtype=points.dtype)
    parent_mask_AB = np.empty(shape, dtype=AB.dtype)
    parent_mask_K = np.empty(shape, dtype=AB.dtype)
    parent_u = np.empty(shape, dtype=AB.dtype)

    # Initialize: We break the cyclic symmetry by fixing Segment 0 to be traversed forward.
    # Therefore, we start exactly at the EXIT of segment 0 (B_0).
    A0 = M              # Entrance of Segment 0
    start_u = M + 1     # Exit of Segment 0
    dp[0, 0, start_u] = 0.0

    # DP Transitions
    for mask_K in range(1 << max(0, K - 1)):
        for mask_AB in range(1 << M):
            for u in range(L):
                cost_u = dp[mask_AB, mask_K, u]
                if np.isinf(cost_u): # ensures we only expand reachable states
                    continue

                # Option 1: Jump to an unvisited AB node
                for v in range(M):
                    if not (mask_AB & (1 << v)):
                        nxt_mask_AB = mask_AB | (1 << v)
                        new_cost = cost_u + dist_matrix[u, v]
                        if new_cost < dp[nxt_mask_AB, mask_K, v]:
                            dp[nxt_mask_AB, mask_K, v] = new_cost
                            parent_mask_AB[nxt_mask_AB, mask_K, v] = mask_AB
                            parent_mask_K[nxt_mask_AB, mask_K, v] = mask_K
                            parent_u[nxt_mask_AB, mask_K, v] = u

                # Option 2: Jump to an unvisited Segment (k in 1..K-1)
                for k in range(1, K):
                    k_bit = k - 1
                    if not (mask_K & (1 << k_bit)):
                        nxt_mask_K = mask_K | (1 << k_bit)
                        Ak, Bk = M + 2*k, M + 2*k + 1
                        
                        # 2A: Traverse Segment Forward (Enter A_k, Exit B_k -> we land at B_k)
                        new_cost_fwd = cost_u + dist_matrix[u, Ak]
                        if new_cost_fwd < dp[mask_AB, nxt_mask_K, Bk]:
                            dp[mask_AB, nxt_mask_K, Bk] = new_cost_fwd
                            parent_mask_AB[mask_AB, nxt_mask_K, Bk] = mask_AB
                            parent_mask_K[mask_AB, nxt_mask_K, Bk] = mask_K
                            parent_u[mask_AB, nxt_mask_K, Bk] = u

                        # 2B: Traverse Segment Backward (Enter B_k, Exit A_k -> we land at A_k)
                        new_cost_bwd = cost_u + dist_matrix[u, Bk]
                        if new_cost_bwd < dp[mask_AB, nxt_mask_K, Ak]:
                            dp[mask_AB, nxt_mask_K, Ak] = new_cost_bwd
                            parent_mask_AB[mask_AB, nxt_mask_K, Ak] = mask_AB
                            parent_mask_K[mask_AB, nxt_mask_K, Ak] = mask_K
                            parent_u[mask_AB, nxt_mask_K, Ak] = u

    # Find the best valid cycle closure back to the entrance of Segment 0 (A_0)
    final_mask_AB = (1 << M) - 1
    final_mask_K = (1 << max(0, K - 1)) - 1
    
    best_cost = np.inf
    best_last_u = -1

    for u in range(L):
        if not np.isinf(dp[final_mask_AB, final_mask_K, u]):
            # Close the loop by connecting the final node 'u' back to A0
            cost = dp[final_mask_AB, final_mask_K, u] + dist_matrix[u, A0]
            if cost < best_cost:
                best_cost = cost
                best_last_u = u

    if np.isinf(best_cost):
        raise ValueError("No valid reconnection strategy found")

    # --- Reconstruct the Paths ---
    paths: list[Path] = []

    curr_start = best_last_u
    curr_path: list[int] = []

    curr_mask_AB, curr_mask_K, curr_u = parent_mask_AB[final_mask_AB, final_mask_K, best_last_u], parent_mask_K[final_mask_AB, final_mask_K, best_last_u], parent_u[final_mask_AB, final_mask_K, best_last_u]

    while curr_u != start_u:
        prev_u = parent_u[curr_mask_AB, curr_mask_K, curr_u]
        prev_mask_AB = parent_mask_AB[curr_mask_AB, curr_mask_K, curr_u]
        prev_mask_K = parent_mask_K[curr_mask_AB, curr_mask_K, curr_u]

        if curr_mask_K == prev_mask_K:  # we took an AB node
            curr_path.append(active_nodes[curr_u])
        else:  # we took a segment
            k = (curr_u - M) // 2
            is_B = (curr_u - M) % 2 == 1
            entrance = M + 2*k + (0 if is_B else 1)

            paths.append(Path(start=active_nodes[curr_start], end=active_nodes[entrance], internal_nodes=curr_path))

            curr_start = curr_u
            curr_path = []

        curr_mask_AB, curr_mask_K, curr_u = prev_mask_AB, prev_mask_K, prev_u

    # Add the final path back to the start of Segment 0
    paths.append(Path(start=active_nodes[curr_start], end=active_nodes[A0], internal_nodes=curr_path))

    return best_cost, paths


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
