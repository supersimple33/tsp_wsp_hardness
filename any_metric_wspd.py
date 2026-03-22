from collections.abc import Iterator

import numpy as np
from numba import njit

type DistMatrix[N: int] = np.ndarray[tuple[N, N], np.dtype[np.floating]]
type PointIndices = np.ndarray[tuple[int], np.dtype[np.integer]] # np.unsignedinteger

@njit
def _stream_raw_pairs[N: int](dist_matrix: DistMatrix[N], eps: float) -> Iterator[list[tuple[PointIndices, PointIndices]]]:
    """Optimized numba code for stating the WSPD generation"""
    n = dist_matrix.shape[0]

    dist_indices = set()
    for r in range(n):
        for c in range(r+1, n):
            if dist_matrix[r, c] > 0:
                val = np.floor(np.log2(dist_matrix[r, c] / (4.0 / eps)))
                dist_indices.add(val)

    dist_indices = sorted(dist_indices)

    # pre-allocate trackers
    min_dists = np.empty(n, dtype=dist_matrix.dtype) # REVIEW: dtype choice
    nearest_site = np.empty(n, dtype=np.uintp) # REVIEW: dtype choice

    # pre-allocate cluster counting
    cluster_counts = np.empty(n, dtype=np.uintp)
    cluster_offsets = np.empty(n + 1, dtype=np.uintp)
    cluster_points = np.empty(n, dtype=np.uintp)
    current_offsets = np.empty(n, dtype=np.uintp)

    for i in dist_indices:
        # reset trackers
        min_dists.fill(np.inf)
        cluster_counts.fill(0)

        # compute the packing
        packing = []
        r = 2**(i-1)

        for j in range(n):
            if min_dists[j] > r:
                packing.append(j)
                # numba speedup of closer_mask = dist_matrix[j] < min_dists
                for k in range(n):
                    dist = dist_matrix[j, k]
                    if dist < min_dists[k]:
                        min_dists[k] = dist
                        nearest_site[k] = j

        if len(packing) < 2: # quick exit since there will be no pairs then
            continue

        # compute clusters
        for p in range(n):
            cluster_counts[nearest_site[p]] += 1
        
        cluster_offsets[0] = 0
        #np.cumsum(cluster_counts, out=cluster_offsets[1:]) # TODO: uncomment when supported
        for j in range(n):
            cluster_offsets[j+1] = cluster_offsets[j] + cluster_counts[j]
            current_offsets[j] = cluster_offsets[j]

        for p in range(n):
            site = nearest_site[p]
            cluster_points[current_offsets[site]] = p
            current_offsets[site] += 1

        lower_bound = (2/eps) * (2**i)
        upper_bound = (16/eps) * (2**i)
        wspd_i: list[tuple[PointIndices, PointIndices]] = []

        for j in range(len(packing)):
            for k in range(j+1, len(packing)):
                x, y = packing[j], packing[k]
                if lower_bound <= dist_matrix[x, y] < upper_bound:
                    start_x, end_x = cluster_offsets[x], cluster_offsets[x+1]
                    A = cluster_points[start_x:end_x].copy() # NOTE: copy needed since cluster_points is reused

                    start_y, end_y = cluster_offsets[y], cluster_offsets[y+1]
                    B = cluster_points[start_y:end_y].copy() # NOTE: copy needed since cluster

                    wspd_i.append((A, B))

        if len(wspd_i) > 0:
            yield wspd_i

def gen_wspd[N: int](dist_matrix: DistMatrix[N], eps: float) -> list[tuple[PointIndices, PointIndices]]:
    """Generates a WSPD for the given distance matrix and separation factor epsilon.
    Based on the algorithm described in "Well-Separated Pairs Decomposition Revisited"
    by Har-Peled et al.

    Undefined behavior if eps is not in the range (0, 1) or dist_matrix non metric
    Also questionable behavior if dist matrix in non-finite
    """
    assert 1 > eps > 0, "undefined behavior outside this range"
    
    global_seen: set[tuple[bytes, bytes]] = set()
    wspd: list[tuple[PointIndices, PointIndices]] = []
    for wspd_i in _stream_raw_pairs(dist_matrix, eps):
        level_ids = [] # REVIEW: is using level ids actually faster?
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