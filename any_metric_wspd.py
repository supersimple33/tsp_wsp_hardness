from collections.abc import Iterator

import numpy as np
from numba import njit

type DistMatrix[N: int] = np.ndarray[tuple[N, N], np.dtype[np.floating]]
type PointIndices = np.ndarray[tuple[int], np.dtype[np.integer]] # np.unsignedinteger

@njit
def _stream_raw_pairs[N: int](dist_matrix: DistMatrix[N], eps: float) -> Iterator[list[tuple[PointIndices, PointIndices]]]:
    """Optimized numba code for stating the WSPD generation"""
    assert 1 > eps > 0, "undefined behavior outside this range"
    n = dist_matrix.shape[0]

    rescaled_dist_mat = dist_matrix / (4/eps)
    rescaled_dist_mat = np.log2(rescaled_dist_mat) # out = would speed up except this is faster
    rescaled_dist_mat = np.floor(rescaled_dist_mat)
    np.fill_diagonal(rescaled_dist_mat, np.nan)
    dist_indices = np.unique(rescaled_dist_mat) # sorting is necessary here

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
        wspd_i: list[tuple[PointIndices, PointIndices]] = []

        for j in range(len(packing)):
            for k in range(j+1, len(packing)):
                x, y = packing[j], packing[k]
                if lower_bound <= dist_matrix[x, y] < upper_bound:
                    A: PointIndices = np.nonzero(nearest_site == x)[0]
                    B: PointIndices = np.nonzero(nearest_site == y)[0]

                    wspd_i.append((A, B))

        yield wspd_i

def gen_wspd[N: int](dist_matrix: DistMatrix[N], eps: float) -> list[tuple[PointIndices, PointIndices]]:
    """Generates a WSPD for the given distance matrix and separation factor epsilon.
    Based on the algorithm described in "Well-Separated Pairs Decomposition Revisited" by Har-Peled et al.
    """
    global_seen: set[tuple[bytes, bytes]] = set()
    wspd: list[tuple[PointIndices, PointIndices]] = []
    for wspd_i in _stream_raw_pairs(dist_matrix, eps):
        level_ids = []
        for A, B in wspd_i:
            A_bytes = A.tobytes()
            B_bytes = B.tobytes()
            level_ids.append((A_bytes, B_bytes) if A_bytes < B_bytes else (B_bytes, A_bytes))
        
        level_set = set(level_ids)
        new_ids = level_set - global_seen

        global_seen.update(new_ids)

        for i, id in enumerate(level_ids):
            if id in new_ids:
                wspd.append(wspd_i[i])

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