from collections import deque
from itertools import combinations

import numpy as np

type DistMatrix[N: int] = np.ndarray[tuple[N, N], np.dtype[np.floating]]
type PointIndices = np.ndarray[tuple[int], np.dtype[np.integer]] # np.unsignedinteger
type WSPD = list[tuple[PointIndices, PointIndices]]

def gen_wspd[N: int](dist_matrix: DistMatrix[N], eps: float) -> WSPD:
    """Generates a WSPD for the given distance matrix and separation factor epsilon.
    Based on the algorithm described in "Well-Separated Pairs Decomposition Revisited" by Har-Peled et al.
    """
    assert 1 > eps > 0, "undefined behavior outside this range"
    n = dist_matrix.shape[0]

    rescaled_dist_mat = dist_matrix / (4/eps)
    np.log2(rescaled_dist_mat, out=rescaled_dist_mat) # this could be spedup by frexp
    np.floor(rescaled_dist_mat, out=rescaled_dist_mat)
    np.fill_diagonal(rescaled_dist_mat, np.nan)
    dist_indices = np.unique(rescaled_dist_mat) # sorting is necessary here

    wspd: WSPD = []
    seen_pairs: set[tuple[bytes, bytes]] = set() # used for deduplication

    for i in dist_indices:
        if np.isnan(i): # skip diagonals
            continue

        # compute the packing
        packing = []
        r = 2**(i-1)

        # track the distance to nearest center and index of that center for later
        min_dists = np.full(n, np.inf)
        nearest_site = np.empty(n, dtype=np.uintp)

        for j in range(n):
            if min_dists[j] > r:
                packing.append(j)
                
                dists = dist_matrix[j]

                closer_mask = dists < min_dists

                min_dists[closer_mask] = dists[closer_mask]
                nearest_site[closer_mask] = j

        if len(packing) < 2: # quick exit since 
            continue

        lower_bound = (2/eps) * (2**i)
        upper_bound = (16/eps) * (2**i)

        for j,k in combinations(packing, 2):
            if lower_bound <= dist_matrix[j, k] < upper_bound:
                A: PointIndices = np.nonzero(nearest_site == j)[0]
                B: PointIndices = np.nonzero(nearest_site == k)[0]

                bytes_A = A.tobytes()
                bytes_B = B.tobytes()
                pair_id = (bytes_A, bytes_B) if bytes_A < bytes_B else (bytes_B, bytes_A)
                if pair_id not in seen_pairs:
                    seen_pairs.add(pair_id)
                    wspd.append((A, B))

    return wspd

x = np.array([
    [0.0, 0.5, 5.0, 5.0],
    [0.5, 0.0, 5.0, 5.0],
    [5.0, 5.0, 0.0, 0.25],
    [5.0, 5.0, 0.25, 0.0]
])

    #[0.0, 1.5, 5.0],
    #[1.5, 0.0, 5.0],
    #[5.0, 5.0, 0.0]

    #[0.0, 4.0, 5.0], 
    #[4.0, 0.0, 1.5], 
    #[5.0, 1.5, 0.0]

wspd = gen_wspd(x, 0.9999)
print(wspd)