from __future__ import annotations

from typing import List, TypeVar

import numpy as np

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)

DistMatrix = np.ndarray[tuple[N, N], np.dtype[np.floating]]
NodeList = np.ndarray[tuple[M], np.dtype[np.integer]]
AdjMatrix = np.ndarray[tuple[N, N], np.dtype[np.bool_]]


def _validate_distance_matrix(
    D: DistMatrix, s: float, k: int
) -> tuple[DistMatrix, int]:
    D = np.asarray(D)
    n = D.shape[0]
    if s <= 0:
        raise ValueError("s must be > 0")
    if k < 2 or k >= n:
        raise ValueError("k must be >= 2 and < n")
    if n == 0:
        return D, n
    if D.ndim != 2 or D.shape[1] != n:
        raise ValueError("D must be square")
    if not np.allclose(np.diag(D), 0.0, atol=1e-12):
        raise ValueError("D must have zero diagonal (metric distance matrix)")
    if not np.allclose(D, D.T, atol=1e-9):
        raise ValueError("D must be symmetric")
    return D, n


def _candidate_diameters(D: DistMatrix) -> np.ndarray:
    n = D.shape[0]
    if n < 2:
        return np.array([], dtype=D.dtype)
    upper = D[np.triu_indices(n, 1)]
    return np.unique(upper[upper > 0])


def _separation_ok(
    D: DistMatrix, comps: List[NodeList], comp_diams: np.ndarray, s: float
) -> bool:
    for a, Ca in enumerate(comps):
        da = comp_diams[a]
        for b in range(a + 1, len(comps)):
            Cb = comps[b]
            db = comp_diams[b]
            delta = min_intercluster_distance(D, Ca, Cb)
            if delta < s * max(da, db) - 1e-12:
                return False
    return True


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
    D, n = _validate_distance_matrix(D, s, k)
    if n == 0:
        return []

    # Candidate diameter thresholds: all distinct positive pairwise distances
    M_candidates = _candidate_diameters(D)

    best_score = n
    best_components: List[NodeList] = [np.arange(n, dtype=np.int32)]

    for M in M_candidates:
        thresh = s * M

        # Build threshold graph adjacency matrix: D_ij < s*M
        adj = D < thresh
        np.fill_diagonal(adj, False)

        comps = connected_components(adj)
        if not 1 < len(comps) <= k:
            continue

        # Diameter feasibility: ensure each component has diameter <= M
        comp_diams = np.array([diameter_of_set(D, comp) for comp in comps])
        if np.any(comp_diams > M + 1e-12):
            continue

        # Separation feasibility: delta(C,C') >= s * max(diam(C), diam(C'))
        if not _separation_ok(D, comps, comp_diams, s):
            continue

        # Balance objective: minimize size of largest cluster
        score = max(c.size for c in comps)
        if score < best_score:
            best_score = score
            best_components = comps

    return [c.tolist() for c in best_components]


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
            newly_seen = neighbors & unseen
            if np.any(newly_seen):
                unseen[newly_seen] = False
                frontier[newly_seen] = True
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

    clusters = balanced_metric_split(D, s=0.5, k=3)
    print("Clusters (0-indexed):", clusters)
    # Pretty print with point values
    print("Clusters (by value):", [[pts[i] for i in c] for c in clusters])
