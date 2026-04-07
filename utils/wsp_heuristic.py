import numpy as np
import numba as nb
import tsplib95

from .helpers import ListOfInt
from .wspd_euc import get_wspd
from .sort_by_key_inplace import sort_by_key_inplace

S_FACTOR = 1.5

@nb.njit(cache=True, inline="always", boundscheck=False, nogil=True)
def wsp_heuristic_good(a_list: ListOfInt, b_list: ListOfInt, pos_in_tour: ListOfInt) -> bool:
    r"""A heuristic to check if a path is good based on the WSPs

    Takes in the nodes in A and B in the tour (sorted in order of tour positions), and positions of each node in the tour.
    :math:`A_{pos} \subset [0, num_points-1]` are the positions of the nodes in A in the tour
    :math:`B_{pos} \subset [0, num_points-1]` are the positions of the nodes in B in the tour
    and :math:`A_{pos} \cap B_{pos} = \emptyset`.

    We then check the following conditions:
    1. If there are no exit pairs with endpoints in separate sets, then there must be exactly two edges connecting A and B
    2. If there are an even number of exit pairs with endpoints in separate sets, then there must be no edges connecting A and B 
    3. If there are an odd number of exit pairs with endpoints in separate sets, then there must be exactly one edge connecting A and B
    """
    num_points = len(pos_in_tour)
    na = len(a_list)
    nb = len(b_list)
    #assert len(na) > 0 and len(nb) > 0, "Sets must be non-empty"

    # keep track of how many edges directly connect A and B in the tour
    biconn_AB_count = 0
    biconn_BA_count = 0
    # keep track of how many exit pairs have endpoints in separate sets
    exit_AA_count = 0
    exit_BB_count = 0
    exit_AB_count = 0
    exit_BA_count = 0

    # check the edge cases
    if pos_in_tour[a_list[-1]] == num_points - 1 and pos_in_tour[b_list[0]] == 0:
        # last point in tour connects from A to B tour=(AxxxxB)
        biconn_AB_count += 1
    elif pos_in_tour[b_list[-1]] == num_points - 1 and pos_in_tour[a_list[0]] == 0:
        # last point in tour connects from B to A tour=(BxxxxA)
        biconn_BA_count += 1
    elif (pos_in_tour[a_list[-1]] == num_points - 1 and pos_in_tour[a_list[0]] == 0) or (pos_in_tour[b_list[-1]] == num_points - 1 and pos_in_tour[b_list[0]] == 0):
        pass # last point connects back to the same set tour=(AxxxxA or BxxxxB)
    elif pos_in_tour[a_list[-1]] > pos_in_tour[b_list[-1]]:
        # between A and B, which is the furthest back in the tour?
        if pos_in_tour[a_list[0]] < pos_in_tour[b_list[0]]:
            # the exit which loops around the end of the tour is an AA exit tour=(--AxxxxA--)
            exit_AA_count += 1
        else:
            # the exit which loops around the end of the tour is an AB exit tour=(--BxxxxA--)
            exit_AB_count += 1
    else:
        if pos_in_tour[b_list[0]] < pos_in_tour[a_list[0]]:
            # the exit which loops around the end of the tour is an BB exit tour=(--BxxxxB--)
            exit_BB_count += 1
        else:
            # the exit which loops around the end of the tour is an BB exit tour=(--AxxxxB--)
            exit_BA_count += 1

    # two pointer approach to count biconns and exits
    i = j = 0

    # tracking info
    next_a = pos_in_tour[a_list[0]]
    next_b = pos_in_tour[b_list[0]]
    exited_A = None

    while True:
        if next_a <= next_b:
            # handle if exited
            if exited_A is not None:
                if exited_A:
                    exit_AA_count += 1
                else:
                    exit_BA_count += 1
                exited_A = None

            # check if biconn or exit
            if next_a + 1 == next_b:
                biconn_AB_count += 1
            elif i + 1 >= na or next_a + 1 != pos_in_tour[a_list[i+1]]:
                exited_A = True

            # increment i and update next_a
            i += 1
            if i < na:
                next_a = pos_in_tour[a_list[i]]
            else:
                break
        else:
            # handle if exited
            if exited_A is not None:
                if exited_A:
                    exit_AB_count += 1
                else:
                    exit_BB_count += 1
                exited_A = None

            # check if biconn or exit
            if next_b + 1 == next_a:
                biconn_BA_count += 1
            elif j + 1 >= nb or next_b + 1 != pos_in_tour[b_list[j+1]]:
                exited_A = False
            
            # increment j and update next_b
            j += 1
            if j < nb:
                next_b = pos_in_tour[b_list[j]]
            else:
                break
    # at this point one or both sets are expended
    while i < na:
        if exited_A is not None:
            if exited_A:
                exit_AA_count += 1
            else:
                exit_BA_count += 1
            exited_A = None
        if i + 1 >= na or pos_in_tour[a_list[i]] + 1 != pos_in_tour[a_list[i+1]]:
            exited_A = True
        i += 1
    while j < nb:
        if exited_A is not None:
            if exited_A:
                exit_AB_count += 1
            else:
                exit_BB_count += 1
            exited_A = None
        if j + 1 >= nb or pos_in_tour[b_list[j]] + 1 != pos_in_tour[b_list[j+1]]:
            exited_A = False
        j += 1

    ## covers both single in outs and multi case
    cross_exits = exit_AB_count + exit_BA_count
    if cross_exits == 0:
        return biconn_AB_count == 1 and biconn_BA_count == 1
    elif cross_exits % 2 == 0:
        return biconn_AB_count == 0 and biconn_BA_count == 0
    else:
        return (biconn_AB_count == 1 and biconn_BA_count == 0) or (biconn_AB_count == 0 and biconn_BA_count == 1)
    

@nb.njit(cache=True, parallel=True, boundscheck=False, nogil=True)
def count_bad_wspd_parallel(pos_in_tour, pairs, node_ranges, indices):
    num_pairs = len(pairs)
    is_bad = np.zeros(num_pairs, dtype=np.bool_)
    
    for i in range(num_pairs): # nb.prange(num_pairs):
        a_node = pairs[i][0]
        b_node = pairs[i][1]

        a_start = node_ranges[a_node]['start']
        a_end = node_ranges[a_node]['end']
        b_start = node_ranges[b_node]['start']
        b_end = node_ranges[b_node]['end']
        
        # if both sets have more than 1 point, then we need to check the heuristic. Otherwise, it is automatically satisfied
        if (a_end - a_start) > 1 and (b_end - b_start) > 1:
            # 3. POSITION MAPPING & SORTING
            A = indices[a_start:a_end].copy()
            B = indices[b_start:b_end].copy()
            sort_by_key_inplace(A, pos_in_tour)
            sort_by_key_inplace(B, pos_in_tour)
            assert np.all(np.diff(pos_in_tour[A]) > 0)
            assert np.all(np.diff(pos_in_tour[B]) > 0)

            # 4. HEURISTIC CHECK
            if not wsp_heuristic_good(A, B, pos_in_tour):
                is_bad[i] = True

    return np.flatnonzero(is_bad)

def check_tour_with_wspd(wspd: tuple[np.ndarray, np.ndarray, np.ndarray], tour: np.ndarray):
    pairs, node_ranges, indices = wspd

    # vectorized node -> position map
    pos_in_tour = np.empty(len(tour), dtype=np.uint32)
    pos_in_tour[tour] = np.arange(len(tour), dtype=np.uint32)

    # 2. Execute parallel loop natively in Numba
    bad_pairs = count_bad_wspd_parallel(
        pos_in_tour,
        pairs,
        node_ranges,
        indices,
    )

    return bad_pairs

def check_problem_tour(prob: tsplib95.models.StandardProblem, tour: np.ndarray, s: float = S_FACTOR, verbose: bool = True, ):
    """Given a TSP and a non-wrapping tour, checks if the WSP heuristic holds"""
    points = np.array([prob.node_coords[i] for i in prob.get_nodes()], dtype=np.float64) # pyright: ignore[reportIndexIssue]

    # Build the WSPDs with different separation factors
    pairs, node_ranges, indices = get_wspd(points, s, np.int64)
    if verbose:
        print("built wspd1.5", flush=True)

    return check_tour_with_wspd((pairs, node_ranges, indices), tour), pairs, node_ranges, indices