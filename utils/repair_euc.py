from typing import NamedTuple, Literal

import numpy as np
import numba as nb
from numba.typed import List, Dict

from .helpers import _euclidean

BRUTE_FORCE_THRESHOLD = 10  # If there are fewer than this many mutable edges, just brute force all possibilities
NB_INT_TYPE_GUIDE = nb.uint32  # Guide for numba list type inference, should be set to the integer type used for node IDs in the tour

ENTRANCE = 0
EXIT = 1

type ListOfBool = np.ndarray[tuple[int], np.dtype[np.bool_]]
type ListOfInt = np.ndarray[tuple[int], np.dtype[np.integer]]
#type ListOfEnterExit = np.ndarray[tuple[int], np.dtype[np.void]] # TODO: update to better numpy typing
type ListOfEnterExit = np.ndarray[tuple[int, Literal[2]], np.dtype[np.integer]]
type ListOfPoints = np.ndarray[tuple[int, ...], np.dtype[np.floating]]

class Path(NamedTuple):
    start: int
    end: int
    internal_nodes: list[int]

@nb.njit(inline="always", cache=True, nogil=True)
def _entrance_exit_masks(
    tour: ListOfInt, AB: ListOfInt
) -> tuple[ListOfBool, ListOfBool]:
    """Return boolean masks of which nodes in the tour are entrances and exits to !(A union B)"""
    n = tour.size
    in_ab = np.isin(tour, AB, assume_unique=True)

    # A node is an exit if it is outside A union B and the next node in the tour is inside A union B
    exit_mask = np.empty(n, dtype=np.bool_)
    exit_mask[:-1] = ~in_ab[:-1] & in_ab[1:]
    exit_mask[-1] = ~in_ab[-1] & in_ab[0]
    # A node is an entrance if the node before it is inside A union B and the node itself is outside A union B
    entrance_mask = np.empty(n, dtype=np.bool_)
    entrance_mask[1:] = in_ab[:-1] & ~in_ab[1:]
    entrance_mask[0] = in_ab[-1] & ~in_ab[0]

    return entrance_mask, exit_mask

@nb.njit(inline="always", cache=True, nogil=True)
def _entrance_exit_inds(
    tour: ListOfInt, entrance_indices: ListOfInt, exit_indices: ListOfInt
) -> ListOfEnterExit:
    """Return a list of (exit_index, entrance_index) pairs in the order they appear in the tour"""
    assert exit_indices.size == entrance_indices.size, "Number of exit edges must equal number of entrance edges"

    starts_entrance = entrance_indices[0] < exit_indices[0] 
    entrance_exit_inds = np.empty((exit_indices.size, 2), dtype=tour.dtype) # [("entrance", tour.dtype), ("exit", tour.dtype)])
    if starts_entrance:
        entrance_exit_inds[:, ENTRANCE] = entrance_indices
        entrance_exit_inds[:, EXIT] = exit_indices
    else:
        entrance_exit_inds[:, ENTRANCE] = entrance_indices
        entrance_exit_inds[:-1, EXIT] = exit_indices[1:]
        entrance_exit_inds[-1, EXIT] = exit_indices[0]
    return entrance_exit_inds

@nb.njit(inline="always", cache=True, nogil=True)
def _unified_dp_repair(entrance_exit_nodes: ListOfEnterExit, AB: ListOfInt, points: ListOfPoints) -> tuple[float, list[Path]]:
    """
    Simultaneously finds the optimal segment sequence, traversal direction, and AB point distribution.
    Returns: (best_cost, list_of_paths)
    """
    K = len(entrance_exit_nodes)
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
        active_nodes[M + 2*k] = entrance_exit_nodes[k, ENTRANCE] # A_k
        active_nodes[M + 2*k + 1] = entrance_exit_nodes[k, EXIT]   # B_k

    # Precompute distance matrix to avoid recalculating in inner loops
    dist_matrix = np.zeros((L, L), dtype=points.dtype)
    for i in range(L):
        for j in range(i+1, L):
            if i != j:
                dist_matrix[i, j] = _euclidean(points, active_nodes[i], active_nodes[j])
                dist_matrix[j, i] = dist_matrix[i, j]

    # DP dimensions: (mask_AB, mask_K, current_node)
    # mask_K has K-1 bits (since we always start at segment 0, we only track segments 1 to K-1)
    TOTAL_BITS = M + max(0, K - 1)
    shape = (1 << TOTAL_BITS, L)
    
    dp = np.full(shape, np.inf, dtype=points.dtype)
    parent = np.empty(shape, dtype=AB.dtype)

    # Initialize: We break the cyclic symmetry by fixing Segment 0 to be traversed forward.
    # Therefore, we start exactly at the EXIT of segment 0 (B_0).
    A0 = M              # Entrance of Segment 0
    start_u = M + 1     # Exit of Segment 0
    dp[0, start_u] = 0.0

    # DP Transitions
    for mask in range(1 << TOTAL_BITS):
        for u in range(L):
            cost_u = dp[mask, u]
            if np.isinf(cost_u): # ensures we only expand reachable states
                continue

            # Option 1: Jump to an unvisited AB node
            for v in range(M):
                if not (mask & (1 << v)):
                    nxt_mask = mask | (1 << v)
                    new_cost = cost_u + dist_matrix[u, v]
                    if new_cost < dp[nxt_mask, v]:
                        dp[nxt_mask, v] = new_cost
                        parent[nxt_mask, v] = (mask << 8) | u

            # Option 2: Jump to an unvisited Segment (k in 1..K-1)
            for k in range(1, K):
                bit_k = 1 << (M + k - 1)
                if not (mask & bit_k):
                    nxt_mask = mask | bit_k
                    Ak, Bk = M + 2*k, M + 2*k + 1
                    
                    # 2A: Traverse Segment Forward (Enter A_k, Exit B_k -> we land at B_k)
                    new_cost_fwd = cost_u + dist_matrix[u, Ak]
                    if new_cost_fwd < dp[nxt_mask, Bk]:
                        dp[nxt_mask, Bk] = new_cost_fwd
                        parent[nxt_mask, Bk] = (mask << 8) | u

                    # 2B: Traverse Segment Backward (Enter B_k, Exit A_k -> we land at A_k)
                    new_cost_bwd = cost_u + dist_matrix[u, Bk]
                    if new_cost_bwd < dp[nxt_mask, Ak]:
                        dp[nxt_mask, Ak] = new_cost_bwd
                        parent[nxt_mask, Ak] = (mask << 8) | u

    # Find the best valid cycle closure back to the entrance of Segment 0 (A_0)
    final_mask = (1 << TOTAL_BITS) - 1
    
    best_cost = np.inf
    best_last_u = 0

    for u in range(L):
        if not np.isinf(dp[final_mask, u]):
            # Close the loop by connecting the final node 'u' back to A0
            cost = dp[final_mask, u] + dist_matrix[u, A0]
            if cost < best_cost:
                best_cost = cost
                best_last_u = u

    if np.isinf(best_cost):
        raise ValueError("No valid reconnection strategy found")

   # --- Reconstruct the Paths ---
    paths: list[Path] = []

    # In forward time, the sequence is: 
    # Segment 0 (Exit B0) -> Path 1 -> Segment K1 (Entrance) ... (Exit) -> Path Last -> A0
    
    # We backtrack from the end (the jump to A0)
    curr_path_end_node = A0
    curr_path_internal: list[int] = List.empty_list(NB_INT_TYPE_GUIDE)  # Collect internal nodes for the current path segment
    
    curr_u = best_last_u
    curr_mask = final_mask

    while True:
        if curr_u < M:
            # It's an AB node: add to the internal nodes of the current gap
            curr_path_internal.append(active_nodes[curr_u])
        else:
            # It's a Segment Exit: This is where a forward-moving path STARTS.
            # We close the current path gap here.
            paths.append(Path(
                start=active_nodes[curr_u],
                end=active_nodes[curr_path_end_node],
                internal_nodes=curr_path_internal[::-1] # Reverse because we collected backwards
            ))
            
            # The NEXT path (going backwards) will end at the ENTRANCE of this segment
            k = (curr_u - M) // 2
            is_B = (curr_u - M) % 2 == 1
            # If we are at the Exit (B_k), the entrance was A_k, and vice versa.
            entrance_idx = M + 2*k + (0 if is_B else 1)
            
            curr_path_end_node = entrance_idx
            curr_path_internal = List.empty_list(NB_INT_TYPE_GUIDE)

        # If we have reached the very first exit (B0), we are done.
        if curr_u == start_u:
            break

        # Move to the parent state
        p = parent[curr_mask, curr_u]
        prev_mask = p >> 8
        prev_u = p & 0xFF

        curr_u, curr_mask = prev_u, prev_mask

    # The paths are currently in reverse chronological order (Last Path -> First Path)
    return best_cost, paths[::-1] 

@nb.njit(inline="always", cache=True, nogil=True)
def _exhaustive_repair(tour: ListOfInt, entrance_exit_inds: ListOfEnterExit, AB: ListOfInt, points: ListOfPoints) -> np.ndarray:
    entrance_exit_nodes = np.empty_like(entrance_exit_inds)
    entrance_exit_nodes[:, ENTRANCE] = tour[entrance_exit_inds[:, ENTRANCE]]
    entrance_exit_nodes[:, EXIT] = tour[entrance_exit_inds[:, EXIT]]

    _, best_paths = _unified_dp_repair(entrance_exit_nodes, AB, points)

    # maps nodes to their segments
    enter_exit_segments: dict[int, ListOfInt] = Dict.empty(key_type=NB_INT_TYPE_GUIDE, value_type=NB_INT_TYPE_GUIDE[:])
    for i in range(entrance_exit_inds.shape[0]):
        enter_ind = entrance_exit_inds[i, ENTRANCE]
        exit_ind = entrance_exit_inds[i, EXIT]
        if enter_ind <= exit_ind:
            segment = tour[enter_ind:exit_ind+1]
        else:
            segment = np.concatenate((tour[enter_ind:], tour[:exit_ind+1]))

        enter_exit_segments[entrance_exit_nodes[i, ENTRANCE]] = segment
        enter_exit_segments[entrance_exit_nodes[i, EXIT]] = segment[::-1]

    # Reconstruct the tour by following the best paths
    new_tour = np.empty_like(tour)
    idx = 0

    #print(best_paths)
    #print(entrance_exit_inds)
    #print(entrance_exit_nodes)
    for _, end, internal in best_paths:
        for node in internal:
            new_tour[idx] = node
            idx += 1
        
        outside_segment = enter_exit_segments[end]

        new_tour[idx:idx+outside_segment.size] = outside_segment
        idx += outside_segment.size

    return new_tour

@nb.njit(cache=True, nogil=True)
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

    AB = np.concatenate((A, B))
    entrance_mask, exit_mask = _entrance_exit_masks(tour, AB)

    entrance_indices = np.nonzero(entrance_mask)[0]
    exit_indices = np.nonzero(exit_mask)[0]

    entrance_exit_inds = _entrance_exit_inds(tour, entrance_indices, exit_indices)

    if AB.size < 2:
        raise ValueError("At least two nodes must be in A union B for there to be any mutable edges")
    elif AB.size + len(entrance_exit_inds) <= 24:
        return _exhaustive_repair(tour, entrance_exit_inds, AB, points)
    else:
        raise NotImplementedError(f"Greedy repair not implemented yet for large problems (|AB|={AB.size}, K={len(entrance_exit_inds)})")
