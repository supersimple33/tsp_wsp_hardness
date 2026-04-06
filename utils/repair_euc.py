from typing import NamedTuple, Literal

import numpy as np
import numba as nb
from numba.typed import List, Dict

from .helpers import _euclidean, build_concorde_solver, solve_concorde_once, ListOfInt, ListOfPoints, ListOfEnterExit, ListOfBool

BRUTE_FORCE_THRESHOLD = 10  # If there are fewer than this many mutable edges, just brute force all possibilities
NB_INT_TYPE_GUIDE = nb.uint32  # Guide for numba list type inference, should be set to the integer type used for node IDs in the tour

ENTRANCE = 0
EXIT = 1


class Path(NamedTuple):
    start: int
    end: int
    internal_nodes: list[int]

# MARK: - Helpers

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

# MARK: - Dynamic Programming Repair Function

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
                start=curr_u - M,  # Segment index (0-based)
                end=curr_path_end_node - M,  # Segment index (0-based)
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
def _exhaustive_repair(tour: ListOfInt, entrance_exit_inds: ListOfEnterExit, AB: ListOfInt, points: ListOfPoints) -> ListOfInt:
    entrance_exit_nodes = np.empty_like(entrance_exit_inds)
    entrance_exit_nodes[:, ENTRANCE] = tour[entrance_exit_inds[:, ENTRANCE]]
    entrance_exit_nodes[:, EXIT] = tour[entrance_exit_inds[:, EXIT]]

    _, best_paths = _unified_dp_repair(entrance_exit_nodes, AB, points)

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
        
        segment_idx = end // 2
        enter_ind = entrance_exit_inds[segment_idx, ENTRANCE]
        exit_ind = entrance_exit_inds[segment_idx, EXIT]
        if enter_ind <= exit_ind:
            outside_segment = tour[enter_ind:exit_ind+1]
        else:
            outside_segment = np.concatenate((tour[enter_ind:], tour[:exit_ind+1]))

        if end % 2 == 1:
            outside_segment = outside_segment[::-1]

        new_tour[idx:idx+outside_segment.size] = outside_segment
        idx += outside_segment.size

    return new_tour

# MARK: - Approximate Repair Function

@nb.njit(inline="always", cache=True, nogil=True)
def _beam_search_repair(
    entrance_exit_nodes: ListOfEnterExit, 
    AB: ListOfInt, 
    points: ListOfPoints, 
    BEAM_WIDTH: int = 150
) -> tuple[float, list[Path]]:
    """
    Beam Search heuristic for TSP segment reconnection.
    Explores the top `BEAM_WIDTH` parallel paths to avoid greedy local optima
    while remaining strictly memory-bounded and fast.
    """
    K = len(entrance_exit_nodes)
    M = len(AB)

    if K == 0:
        raise ValueError("entrance exit pairs was empty")

    L = M + 2 * K

    active_nodes = np.zeros(L, dtype=AB.dtype)
    if M > 0:
        active_nodes[:M] = AB

    for k in range(K):
        active_nodes[M + 2 * k] = entrance_exit_nodes[k, ENTRANCE]
        active_nodes[M + 2 * k + 1] = entrance_exit_nodes[k, EXIT]

    # Distance table over all active nodes (AB nodes + segment endpoints)
    dist_matrix = np.zeros((L, L), dtype=points.dtype)
    for i in range(L):
        for j in range(i + 1, L):
            dist = _euclidean(points, active_nodes[i], active_nodes[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    num_segment_bits = max(0, K - 1)
    total_steps = M + num_segment_bits

    beam_cap = BEAM_WIDTH
    if beam_cap < 1:
        beam_cap = 1

    branch_factor = M + 2 * num_segment_bits
    cand_cap = beam_cap * max(1, branch_factor)

    A0 = M
    start_u = M + 1

    cost_curr = np.full(beam_cap, np.inf, dtype=points.dtype)
    cost_next = np.full(beam_cap, np.inf, dtype=points.dtype)
    u_curr = np.full(beam_cap, -1, dtype=AB.dtype)
    u_next = np.full(beam_cap, -1, dtype=AB.dtype)

    visited_ab_curr = np.zeros((beam_cap, M), dtype=np.bool_)
    visited_ab_next = np.zeros((beam_cap, M), dtype=np.bool_)
    visited_seg_curr = np.zeros((beam_cap, num_segment_bits), dtype=np.bool_)
    visited_seg_next = np.zeros((beam_cap, num_segment_bits), dtype=np.bool_)

    # Backtracking tables indexed by [depth, beam_position]
    hist_u = np.full((total_steps + 1, beam_cap), -1, dtype=AB.dtype)
    hist_parent = np.full((total_steps + 1, beam_cap), -1, dtype=AB.dtype)

    cand_cost = np.empty(cand_cap, dtype=points.dtype)
    cand_u = np.empty(cand_cap, dtype=AB.dtype)
    cand_parent = np.empty(cand_cap, dtype=AB.dtype)
    cand_type = np.empty(cand_cap, dtype=AB.dtype)  # 0: AB jump, 1: segment traversal
    cand_id = np.empty(cand_cap, dtype=AB.dtype)

    num_curr = 1
    cost_curr[0] = 0.0
    u_curr[0] = start_u
    hist_u[0, 0] = start_u
    hist_parent[0, 0] = -1

    for depth in range(total_steps):
        cand_count = 0

        for i in range(num_curr):
            u = u_curr[i]
            cost_u = cost_curr[i]

            # Option 1: visit an unvisited AB node
            for v in range(M):
                if not visited_ab_curr[i, v]:
                    cand_cost[cand_count] = cost_u + dist_matrix[u, v]
                    cand_u[cand_count] = v
                    cand_parent[cand_count] = i
                    cand_type[cand_count] = 0
                    cand_id[cand_count] = v
                    cand_count += 1

            # Option 2: traverse an unvisited segment (k in 1..K-1), either direction
            for s in range(num_segment_bits):
                if not visited_seg_curr[i, s]:
                    k = s + 1
                    Ak = M + 2 * k
                    Bk = Ak + 1

                    cand_cost[cand_count] = cost_u + dist_matrix[u, Ak]
                    cand_u[cand_count] = Bk
                    cand_parent[cand_count] = i
                    cand_type[cand_count] = 1
                    cand_id[cand_count] = s
                    cand_count += 1

                    cand_cost[cand_count] = cost_u + dist_matrix[u, Bk]
                    cand_u[cand_count] = Ak
                    cand_parent[cand_count] = i
                    cand_type[cand_count] = 1
                    cand_id[cand_count] = s
                    cand_count += 1

        if cand_count == 0:
            raise ValueError("Beam search failed: no candidates")

        order = np.argsort(cand_cost[:cand_count])

        num_next = beam_cap
        if cand_count < num_next:
            num_next = cand_count

        for j in range(num_next):
            cidx = order[j]
            p = cand_parent[cidx]

            cost_next[j] = cand_cost[cidx]
            u_next[j] = cand_u[cidx]

            if M > 0:
                visited_ab_next[j, :] = visited_ab_curr[p, :]
            if num_segment_bits > 0:
                visited_seg_next[j, :] = visited_seg_curr[p, :]

            if cand_type[cidx] == 0:
                visited_ab_next[j, cand_id[cidx]] = True
            else:
                visited_seg_next[j, cand_id[cidx]] = True

            hist_u[depth + 1, j] = u_next[j]
            hist_parent[depth + 1, j] = p

        # Reset unused tail in next-beam buffers to keep state clean between depths
        for j in range(num_next, beam_cap):
            cost_next[j] = np.inf
            u_next[j] = -1
            if M > 0:
                visited_ab_next[j, :] = False
            if num_segment_bits > 0:
                visited_seg_next[j, :] = False

        cost_curr, cost_next = cost_next, cost_curr
        u_curr, u_next = u_next, u_curr
        visited_ab_curr, visited_ab_next = visited_ab_next, visited_ab_curr
        visited_seg_curr, visited_seg_next = visited_seg_next, visited_seg_curr
        num_curr = num_next

    best_cost = np.inf
    best_last_pos = -1

    for i in range(num_curr):
        u = u_curr[i]
        cost = cost_curr[i] + dist_matrix[u, A0]
        if cost < best_cost:
            best_cost = cost
            best_last_pos = i

    if best_last_pos < 0 or np.isinf(best_cost):
        raise ValueError("No valid reconnection strategy found")

    paths: list[Path] = []

    curr_depth = total_steps
    curr_pos = best_last_pos
    curr_u = hist_u[curr_depth, curr_pos]

    curr_path_end_node = A0
    curr_path_internal: list[int] = List.empty_list(NB_INT_TYPE_GUIDE)

    while True:
        if curr_u < M:
            curr_path_internal.append(int(active_nodes[curr_u]))
        else:
            paths.append(Path(
                start=int(active_nodes[curr_u]),
                end=int(active_nodes[curr_path_end_node]),
                internal_nodes=curr_path_internal[::-1]
            ))

            k = (curr_u - M) // 2
            is_B = (curr_u - M) % 2 == 1
            entrance_idx = M + 2 * k + (0 if is_B else 1)

            curr_path_end_node = entrance_idx
            curr_path_internal = List.empty_list(NB_INT_TYPE_GUIDE)

        if curr_u == start_u:
            break

        parent_pos = hist_parent[curr_depth, curr_pos]
        if parent_pos < 0:
            raise ValueError("Beam reconstruction failed")

        curr_depth -= 1
        curr_pos = parent_pos
        curr_u = hist_u[curr_depth, curr_pos]

    return best_cost, paths[::-1]


@nb.njit(inline="always", cache=True, nogil=True)
def _approximate_repair(
    tour: ListOfInt, 
    entrance_exit_inds: ListOfEnterExit, 
    AB: ListOfInt, 
    points: ListOfPoints
) -> np.ndarray:
    
    entrance_exit_nodes = np.empty_like(entrance_exit_inds)
    entrance_exit_nodes[:, ENTRANCE] = tour[entrance_exit_inds[:, ENTRANCE]]
    entrance_exit_nodes[:, EXIT] = tour[entrance_exit_inds[:, EXIT]]

    # Routes to the robust Beam Search solver instead of blowing up exact DP
    _, best_paths = _beam_search_repair(entrance_exit_nodes, AB, points, BEAM_WIDTH=150)

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

    new_tour = np.empty_like(tour)
    idx = 0

    for _, end, internal in best_paths:
        for node in internal:
            new_tour[idx] = node
            idx += 1
        
        outside_segment = enter_exit_segments[end]
        new_tour[idx:idx+outside_segment.size] = outside_segment
        idx += outside_segment.size

    return new_tour

# MARK: - Concorde Repair (for Euclidean TSP)

@nb.njit(inline="always", cache=True, nogil=True)
def build_mini_problem(entrance_exit_nodes: ListOfEnterExit, AB: ListOfInt, points: ListOfPoints) -> np.ndarray:
    entrance_exit_flat = np.ravel(entrance_exit_nodes)

    dist_mat = np.zeros((AB.size + entrance_exit_nodes.size, AB.size + entrance_exit_nodes.size), dtype=np.float64)
    for i in range(AB.size):
        for j in range(i + 1, AB.size):
            d = _euclidean(points, AB[i], AB[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d
        for j in range(entrance_exit_flat.size):
            d = _euclidean(points, AB[i], entrance_exit_flat[j])
            dist_mat[i, AB.size + j] = d
            dist_mat[AB.size + j, i] = d
    for i in range(entrance_exit_flat.size):
        for j in range(i + 1, entrance_exit_flat.size):
            d = _euclidean(points, entrance_exit_flat[i], entrance_exit_flat[j])
            dist_mat[AB.size + i, AB.size + j] = d
            dist_mat[AB.size + j, AB.size + i] = d
    # add a constant penalty to ensure that rules are followed
    dist_mat += 1_000.0
    for i in range(entrance_exit_nodes.shape[0]):
        dist_mat[AB.size + 2*i, AB.size + 2*i + 1] = 0.0
        dist_mat[AB.size + 2*i + 1, AB.size + 2*i] = 0.0

    return dist_mat

@nb.njit(inline="always", cache=True, nogil=True)
def build_up_from_partial_tour(
        partial_tour: ListOfInt, 
        tour: ListOfInt, 
        entrance_exit_inds: ListOfEnterExit, 
        AB: ListOfInt, 
) -> np.ndarray:
    new_tour = np.empty_like(tour)
    idx = 0
    for i, node in enumerate(partial_tour):
        if node < AB.size:
            new_tour[idx] = AB[node]
            idx += 1
        else:
            segment_idx = (node - AB.size) // 2
            beginner = (partial_tour[(i + 1) % partial_tour.size] - AB.size) // 2 == segment_idx
            if beginner:
                enter_ind = entrance_exit_inds[segment_idx, ENTRANCE]
                exit_ind = entrance_exit_inds[segment_idx, EXIT]
                segment = tour[enter_ind:exit_ind+1] if enter_ind <= exit_ind else np.concatenate((tour[enter_ind:], tour[:exit_ind+1]))

                reverse_seg = (node - AB.size) % 2 == 1
                if reverse_seg:
                    segment = segment[::-1]
                
                new_tour[idx:idx+segment.size] = segment
                idx += segment.size

    if idx != tour.size:
        raise ValueError("Reconstructed tour size does not match original tour size")
    
    return new_tour

def _concorde_opt_euc(
    tour: ListOfInt,
    entrance_exit_inds: ListOfEnterExit, 
    AB: ListOfInt, 
    points: ListOfPoints
) -> ListOfInt:
    entrance_exit_nodes = np.empty_like(entrance_exit_inds)
    entrance_exit_nodes[:, ENTRANCE] = tour[entrance_exit_inds[:, ENTRANCE]]
    entrance_exit_nodes[:, EXIT] = tour[entrance_exit_inds[:, EXIT]]

    dist_mat = build_mini_problem(entrance_exit_nodes, AB, points)

    tsp_prob = build_concorde_solver(dist_mat)
    partial_tour, _ = solve_concorde_once(tsp_prob, 42)

    return build_up_from_partial_tour(partial_tour, tour, entrance_exit_inds, AB)

# MARK: - Master Repair Function


def repair_tour_euc(
    tour: ListOfInt,
    A: ListOfInt,
    B: ListOfInt,
    points: ListOfPoints,
    HIGH: int = 25
) -> ListOfInt:
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
    elif AB.size + len(entrance_exit_inds) <= HIGH:
        return _exhaustive_repair(tour, entrance_exit_inds, AB, points)
    else:
        return _concorde_opt_euc(tour, entrance_exit_inds, AB, points)
