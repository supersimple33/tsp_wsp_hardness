from typing import Literal

import numpy as np
import numba as nb

from .helpers import _euclidean, calc_tour_len_euc

BRUTE_FORCE_THRESHOLD = 10  # If there are fewer than this many mutable edges, just brute force all possibilities

type ListOfBool = np.ndarray[tuple[int], np.dtype[np.bool_]]
type ListOfInt = np.ndarray[tuple[int], np.dtype[np.integer]]
type ListOfIntPairs = np.ndarray[tuple[int, Literal[2]], np.dtype[np.integer]]
type ListOfPoints = np.ndarray[tuple[int, ...], np.dtype[np.floating]]

def _mutable_and_exit_edges_masks(
    tour: ListOfInt, A: ListOfInt, B: ListOfInt
) -> tuple[ListOfBool, ListOfBool, ListOfBool]:
    """Return boolean masks of which edges in the tour are mutable (fully internal to A union B) and exit edges (crossing the boundary of A union B)"""
    n = tour.size
    in_a = np.isin(tour, A, assume_unique=True)
    in_b = np.isin(tour, B, assume_unique=True)
    in_ab = in_a | in_b

    # An edge is mutable if both endpoints are in A union B.
    mutable_mask = np.empty(n, dtype=np.bool_)
    mutable_mask[:-1] = in_ab[:-1] & in_ab[1:]
    mutable_mask[-1] = in_ab[-1] & in_ab[0]

    # An edge is an entrance edge if it goes from outside A union B to inside, and an exit edge if it goes from inside to outside. We can detect both by XORing the in_ab status of adjacent nodes.
    entrance_mask = np.empty(n, dtype=np.bool_)
    entrance_mask[:-1] = ~in_ab[:-1] & in_ab[1:]
    entrance_mask[-1] = ~in_ab[-1] & in_ab[0]
    exit_mask = np.empty(n, dtype=np.bool_)
    exit_mask[:-1] = in_ab[:-1] & ~in_ab[1:]
    exit_mask[-1] = in_ab[-1] & ~in_ab[0]

    return mutable_mask, entrance_mask, exit_mask

@nb.njit(inline="always")
def _exit_entrance_pairs(
    tour: ListOfInt, entrance_indices: ListOfInt, exit_indices: ListOfInt
) -> ListOfIntPairs:
    """Return a list of (exit_index, entrance_index) pairs in the order they appear in the tour"""
    assert exit_indices.size == entrance_indices.size, "Number of exit edges must equal number of entrance edges"

    starts_exit = exit_indices[0] < entrance_indices[0]
    exit_entrance_pairs = np.empty((exit_indices.size, 2), dtype=tour.dtype)
    if starts_exit:
        exit_entrance_pairs[:, 0] = exit_indices
        exit_entrance_pairs[:, 1] = entrance_indices
    else:
        exit_entrance_pairs[:, 0] = exit_indices
        exit_entrance_pairs[:-1, 1] = entrance_indices[1:]
        exit_entrance_pairs[-1, 1] = entrance_indices[0]
    return exit_entrance_pairs

    


def _brute_force_repair(tour: ListOfInt, mutable_indices: ListOfInt, exit_entrance_pairs: ListOfIntPairs, points: ListOfPoints) -> ListOfInt:
    # iterate over all endpoint combinations then do a dp style search to find the best reconnection strategy
    pass

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

    mutable_mask, entrance_mask, exit_mask = _mutable_and_exit_edges_masks(tour, A, B)
    mutable_indices = np.nonzero(mutable_mask)[0]

    entrance_indices = np.nonzero(entrance_mask)[0]
    exit_indices = np.nonzero(exit_mask)[0]

    exit_entrance_pairs = _exit_entrance_pairs(tour, entrance_indices, exit_indices)

    if mutable_indices.size < 2:
        print("Cannot repair tour: not enough mutable edges")
        return tour.copy()
    elif len(mutable_indices) < 14:
        print("Few mutable edges, using brute force search")
        return _brute_force_repair(tour, mutable_indices, exit_entrance_pairs, points)
    else:
        print("Many mutable edges, using greedy repair")
        raise NotImplementedError("Greedy repair not implemented yet")
