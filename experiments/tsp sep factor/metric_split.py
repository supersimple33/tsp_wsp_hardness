from __future__ import annotations

from typing import TypeVar

import numpy as np
from scipy.sparse.csgraph import connected_components

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)

DistMatrix = np.ndarray[tuple[N, N], np.dtype[np.floating]]
NodeList = np.ndarray[tuple[M], np.dtype[np.integer]]
AdjMatrix = np.ndarray[tuple[N, N], np.dtype[np.bool_]]


def _candidate_diameters(D: DistMatrix) -> np.ndarray:
    n = D.shape[0]
    if n < 2:
        return np.array([], dtype=D.dtype)
    upper = D[np.triu_indices(n, 1)]
    return np.unique(upper[upper > 0])


def _separation_ok(
    D: DistMatrix, comps: list[NodeList], comp_diams: np.ndarray, s: float, tol=1e-12
) -> bool:
    for a, Ca in enumerate(comps):
        da = comp_diams[a]
        for b in range(a + 1, len(comps)):
            Cb = comps[b]
            db = comp_diams[b]
            delta = min_intercluster_distance(D, Ca, Cb)
            if delta < s * max(da, db) - tol:
                return False
    return True


def balanced_metric_split(D: DistMatrix, s: float, k: int, tol=1e-12) -> list[NodeList]:
    """
    BalancedMetricSplit(D, s, k, tol=1e-12) -> list of clusters

    Inputs:
      - D: n x n metric distance matrix (symmetric, triangle inequality assumed)
      - s: separation factor > 0
      - k: desired number of clusters >= 2
      - tol: tolerance for diameter and separation checks

    Output:
      - A list of up to k clusters (each a list of vertex indices 0..n-1) that:
          * arises as the connected components of threshold graph G_M with edges D_ij < s*M
          * pairwise separation holds: delta(C,C') >= s * max(diam(C), diam(C'))
        and among feasible candidates minimizes max cluster size.
      - Returns [0...n-1] if no feasible partition is found.
    """
    n = D.shape[0]
    M_candidates = np.sort(_candidate_diameters(D))

    best_score = n
    best_components: list[NodeList] = [np.arange(n, dtype=np.int32)]

    for M in M_candidates:
        thresh = s * M
        adj = D < thresh + tol
        np.fill_diagonal(adj, False)

        n_components, labels = connected_components(
            csgraph=adj,
            directed=False,
            return_labels=True,
        )

        if n_components > k or n_components == n:
            # too many components
            continue
        elif n_components == 1:
            # graph is full at this threshold and cannot improve
            break

        score = np.bincount(labels, minlength=n_components).max()
        if best_score <= score:
            # No improvement
            continue

        comps: list[NodeList] = [
            np.flatnonzero(labels == g) for g in range(n_components)
        ]
        comp_diams = np.array([diameter_of_set(D, comp) for comp in comps])
        if not _separation_ok(D, comps, comp_diams, s, tol):
            # Separation condition failed
            continue

        best_score = score
        best_components = comps
        if best_score == (n + k - 1) // k:
            # Optimal possible score achieved
            break

    return best_components


def diameter_of_set(D: DistMatrix, nodes: NodeList) -> np.floating:
    """diam(C) = max_{x,y in C} D[x,y]. O(|C|^2) but vectorized in NumPy."""
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
