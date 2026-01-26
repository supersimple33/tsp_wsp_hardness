from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional
import math

import numpy as np


def balanced_metric_split(
    D: Sequence[Sequence[float]] | np.ndarray, s: float, k: int
) -> List[List[int]]:
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
    if s <= 0:
        raise ValueError("s must be > 0")
    if k < 2:
        raise ValueError("k must be >= 2")

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

    best_score = math.inf
    best_components: List[List[int]] = []

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
        if len(comps) != k:
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
        for a in range(k):
            Ca = comps[a]
            da = comp_diams[a]
            for b in range(a + 1, k):
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


def connected_components(adj: Sequence[Sequence[int]]) -> List[List[int]]:
    """Return components as lists of vertices (0..n-1) using DFS/BFS."""
    n = len(adj)
    seen = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp: List[int] = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(comp)
    return comps


def diameter_of_set(
    D: Sequence[Sequence[float]] | np.ndarray, nodes: Sequence[int]
) -> float:
    """diam(C) = max_{x,y in C} D[x][y]. O(|C|^2)."""
    m = len(nodes)
    if m <= 1:
        return 0.0
    diam = 0.0
    for i in range(m):
        xi = nodes[i]
        Dxi = D[xi]
        for j in range(i + 1, m):
            d = Dxi[nodes[j]]
            if d > diam:
                diam = d
    return diam


def min_intercluster_distance(
    D: Sequence[Sequence[float]] | np.ndarray, A: Sequence[int], B: Sequence[int]
) -> float:
    """Delta(A,B) = min_{x in A, y in B} D[x][y]. O(|A||B|)."""
    best = math.inf
    for x in A:
        Dx = D[x]
        for y in B:
            d = Dx[y]
            if d < best:
                best = d
                if best == 0.0:
                    return 0.0
    return best


# -------------------------
# Small usage example
# -------------------------
if __name__ == "__main__":
    # Example: points on a line => metric via absolute difference
    pts = [0.0, 0.1, 0.2, 10.0, 10.1, 20.0]
    n = len(pts)
    D = [[abs(pts[i] - pts[j]) for j in range(n)] for i in range(n)]

    clusters = balanced_metric_split(D, s=1.0, k=2)
    print("Clusters (0-indexed):", clusters)
    # Pretty print with point values
    print("Clusters (by value):", [[pts[i] for i in c] for c in clusters])
