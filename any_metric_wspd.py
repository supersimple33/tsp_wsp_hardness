import numpy as np
from collections import deque

type DistMatrix[N: int] = np.ndarray[tuple[N, N], np.dtype[np.floating]]
type PointIndices = np.ndarray[tuple[int], np.dtype[np.integer]] # np.unsignedinteger
type WSPD = list[tuple[PointIndices, PointIndices]]

def gen_wspd[N: int](dist_matrix: DistMatrix[N], eps: float) -> WSPD:
    """Generates a WSPD for the given distance matrix and separation factor epsilon.
    Based on the algorithm described in "Well-Separated Pairs Decomposition Revisited" by Har-Peled et al.
    """
    assert 1 >= eps > 0, "undefined behavior outside this range"
    n = dist_matrix.shape[0]

    rescaled_dist_mat = dist_matrix / (4/eps)
    np.log2(rescaled_dist_mat, out=rescaled_dist_mat) # this could be spedup by frexp
    np.floor(rescaled_dist_mat, out=rescaled_dist_mat)
    np.fill_diagonal(rescaled_dist_mat, np.nan)
    dist_indices = np.unique(rescaled_dist_mat) # sorting is necessary here

    wspd: WSPD = []
    for i in dist_indices:
        if np.isnan(i): # skip diagonals
            continue

        # compute the packing
        available = np.ones(n, dtype=bool)
        packing = []
        r = 2**(i-1)
        for j in range(n):
            if available[j]:
                packing.append(j)
                available &= (dist_matrix[j] > r)

        if len(packing) < 2: # quick exit since 
            continue

        wspd_i: WSPD = []
        lower_bound = (2/eps) * (2**i)
        upper_bound = (16/eps) * (2**i)

        nearest_site = np.argmin(dist_matrix[:, packing], axis=1)

        for j in range(len(packing)):
            for k in range(j+1, len(packing)):
                if lower_bound <= dist_matrix[packing[j], packing[k]] < upper_bound:
                    A = np.nonzero(nearest_site == j)[0]
                    B = np.nonzero(nearest_site == k)[0]
                    wspd_i.append((A, B))

        wspd += wspd_i

    return wspd

x = np.array([
    [0.0, 0.0, 5.0, 5.0],
    [0.0, 0.0, 5.0, 5.0],
    [5.0, 5.0, 0.0, 0.25],
    [5.0, 5.0, 0.25, 0.0]
])

    #[0.0, 1.5, 5.0],
    #[1.5, 0.0, 5.0],
    #[5.0, 5.0, 0.0]

    #[0.0, 4.0, 5.0], 
    #[4.0, 0.0, 1.5], 
    #[5.0, 1.5, 0.0]

wspd = gen_wspd(x, 0.5)
print(wspd)