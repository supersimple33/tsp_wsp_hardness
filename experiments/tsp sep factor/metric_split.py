from __future__ import annotations

from typing import List, TypeVar

import numpy as np

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)

DistMatrix = np.ndarray[tuple[N, N], np.dtype[np.floating]]
NodeList = np.ndarray[tuple[M], np.dtype[np.integer]]
AdjMatrix = np.ndarray[tuple[N, N], np.dtype[np.bool_]]


def balanced_metric_split(D: DistMatrix, s: float, k: int) -> List[List[int]]:
    """
    BalancedMetricSplit(D, s, k)

    Inputs:
      - D: n x n metric distance matrix (symmetric, triangle inequality assumed)
      - s: separation factor > 0
      - k: desired number of clusters >= 2

    Output:
      - A list of k clusters (each a list of vertex indices 0..n-1) that:
          * arises as the connected components of threshold graph G_M with edges D_ij < s*M
          * each component has diameter <= M
          * pairwise separation holds: delta(C,C') >= s * max(diam(C), diam(C'))
        and among feasible candidates minimizes max cluster size.
      - Returns [] if no feasible partition is found.
    """
    n = len(D)
    if s <= 0:
        raise ValueError("s must be > 0")
    if k < 2 or k >= n:
        raise ValueError("k must be >= 2 and < n")

    n = len(D)
    if n == 0:
        return []
    if any(len(row) != n for row in D):
        raise ValueError("D must be square")
    # Basic symmetry check (tolerant)
    for i in range(n):
        if abs(D[i][i]) > 1e-12:
            raise ValueError("D must have zero diagonal (metric distance matrix)")
        for j in range(i + 1, n):
            if abs(D[i][j] - D[j][i]) > 1e-9:
                raise ValueError("D must be symmetric")

    # Candidate diameter thresholds: all distinct positive pairwise distances
    M_candidates = sorted(
        {D[i][j] for i in range(n) for j in range(i + 1, n) if D[i][j] > 0}
    )

    best_score = n
    best_components: List[List[int]] = [list(range(n))]

    # Pre-allocate buffers for speed
    adj: List[List[int]] = [[] for _ in range(n)]

    for M in M_candidates:
        thresh = s * M

        # Build threshold graph edges E_M = {(i,j): D_ij < s*M}
        for i in range(n):
            adj[i].clear()
        for i in range(n):
            Di = D[i]
            for j in range(i + 1, n):
                if Di[j] < thresh:
                    adj[i].append(j)
                    adj[j].append(i)

        comps = connected_components(adj)
        if not 1 < len(comps) <= k:
            continue

        # Diameter feasibility: ensure each component has diameter <= M
        comp_diams = []
        ok = True
        for comp in comps:
            d = diameter_of_set(D, comp)
            comp_diams.append(d)
            if d > M + 1e-12:
                ok = False
                break
        if not ok:
            continue

        # Separation feasibility: delta(C,C') >= s * max(diam(C), diam(C'))
        sep = True
        for a in range(len(comps)):
            Ca = comps[a]
            da = comp_diams[a]
            for b in range(a + 1, len(comps)):
                Cb = comps[b]
                db = comp_diams[b]
                delta = min_intercluster_distance(D, Ca, Cb)
                if delta < s * max(da, db) - 1e-12:
                    sep = False
                    break
            if not sep:
                break
        if not sep:
            continue

        # Balance objective: minimize size of largest cluster
        score = max(len(c) for c in comps)
        if score < best_score:
            best_score = score
            best_components = comps

    return best_components


def connected_components(adj: AdjMatrix) -> List[NodeList]:
    """
    Return connected components of an undirected graph
    given a boolean adjacency matrix.
    """
    n = adj.shape[0]
    unseen = np.ones(n, dtype=np.bool_)
    comps: List[NodeList] = []

    while np.any(unseen):
        # pick an unseen node to start a new component
        start = np.argmax(unseen)

        frontier = np.zeros(n, dtype=np.bool_)
        frontier[start] = True
        unseen[start] = False

        comp_nodes = []

        while np.any(frontier):
            current = np.argmax(frontier)
            frontier[current] = False
            comp_nodes.append(current)

            neighbors = adj[current]
            for neighbor in neighbors:
                if unseen[neighbor]:
                    unseen[neighbor] = False
                    frontier[neighbor] = True
        comps.append(np.array(comp_nodes, dtype=np.int32))
    return comps


def diameter_of_set(D: DistMatrix, nodes: NodeList) -> np.floating:
    """diam(C) = max_{x,y in C} D[x,y]. O(|C|^2) but vectorized in NumPy."""
    m = nodes.size
    if m <= 1:
        return D.dtype.type(0.0)
    submatrix = D[np.ix_(nodes, nodes)]
    return np.max(submatrix)


def min_intercluster_distance(D: DistMatrix, A: NodeList, B: NodeList) -> np.floating:
    """Delta(A,B) = min_{x in A, y in B} D[x][y]. O(|A||B|)."""
    if A.size == 0 or B.size == 0:
        return D.dtype.type(np.inf)

    submatrix = D[np.ix_(A, B)]
    return np.min(submatrix)


# -------------------------
# Small usage example
# -------------------------
if __name__ == "__main__":
    # Example: points on a line => metric via absolute difference
    pts = np.asarray([0.0, 0.1, 0.2, 10.0, 10.1, 20.0], dtype=np.float32)
    D = np.abs(pts[:, None] - pts[None, :])

    clusters = balanced_metric_split(D, s=0.5, k=2)
    print("Clusters (0-indexed):", clusters)
    # Pretty print with point values
    print("Clusters (by value):", [[pts[i] for i in c] for c in clusters])
