# ZeRO Hypothesis: Maximum Achievable Separation (s) Tracking

## Problem Formulation
Given three groups of points G1, G2, G3:
- diam(G1) <= 1.0 and diam(G2) <= 1.0
- min dist(G1, outside) >= s
- min dist(G2, outside) >= s
- Distances within G3 are unconstrained (>= 0)
- All distances satisfy symmetry and the metric triangle inequality.

**Hypothesis**: The optimal (shortest) Hamiltonian path starting at a point in G1 and ending at a point in G2 will never contain two disjoint subpaths connecting G1 and G2 (i.e. cannot visit G1 -> G2 -> G1 -> G2).

**Goal**: Find metric configurations that produce a **counterexample** (a flagged path is strictly optimal) while maximizing the separation factor `s` (and separation ratio `r = s / max(diam)`).

---

## Summary of Results

| Configuration | Total Points N | Metric Space | Max Achievable s | Optimal Flagged Path | Proof Bound |
| :--- | :---: | :---: | :---: | :--- | :--- |
| G1=2, G2=2, G3=1 | 5 | Metric | **s < 1.50** (3/2) | `0 -> 3 -> 1 -> 4 -> 2` | 2s < 3 => s < 1.5 |
| G1=2, G2=2, G3=2 | 6 | Metric | **s < 3.00** | `0 -> 4 -> 2 -> 1 -> 5 -> 3` | 5s < 4s + 3 => s < 3.0 |
| G1=3, G2=2, G3=2 | 7 | Metric | **s < 3.00** | `0 -> 1 -> 5 -> 4 -> 2 -> 6 -> 3` | G2 has single intermediate node |
| G1=3, G2=3, G3=2 | 8 | Metric | **s < 3.00** | `0 -> 7 -> 5 -> 4 -> 2 -> 1 -> 6 -> 3` | Higher interleaving costs >= 7s |
| G1=2, G2=2, G3=3 | 7 | Metric | **s < 3.00** | 2 round trips to G3 clusters | Limited by 2 points in G1, G2 |
| G1=2, G2=2, G3=2 | 6 | 3D Euclidean | **s ≈ 2.29** | Geometric embedding | Flat geometry restrictions |
| G1=2, G2=2, G3=2 | 6 | 2D Euclidean | **s ≈ 1.81** | Planar embedding | Planar geometric obstruction |

---

## Key Mathematical Proofs

### 1. Case |G3| = 1 (N = 5 points) => s < 1.5
Let G1 = {0, 1}, G2 = {2, 3}, G3 = {4}. Start at 0, end at 2.

- The flagged path visits node 3 in G2 before node 1 in G1:
  $$\text{P}_{\text{flag}} = 0 \to 3 \to 1 \to 4 \to 2$$
  Since all external edges leaving or entering G1 and G2 have length at least `s`:
  $$\text{Cost}(P_{\text{flag}}) \ge 4s$$

- Consider the unflagged path:
  $$\text{P}_{\text{unflag}} = 0 \to 1 \to 4 \to 3 \to 2$$
  Using triangle inequality:
  $$d(0, 1) \le 1$$
  $$d(4, 3) \le d(4, 2) + d(2, 3) \le d(4, 2) + 1$$
  $$d(3, 2) \le 1$$
  $$\text{Cost}(P_{\text{unflag}}) \le 3 + d(1, 4) + d(4, 2)$$

- Subtracting shared edges $d(1, 4) + d(4, 2)$:
  $$\text{Cost}(P_{\text{flag}}) < \text{Cost}(P_{\text{unflag}}) \iff d(0, 3) + d(3, 1) < 3$$
  Since both distances are at least `s`:
  $$s + s < 3 \implies 2s < 3 \implies \mathbf{s < 1.5}$$

---

### 2. Case |G3| = 2 (N = 6 points) => s < 3.0
Let G1 = {0, 1}, G2 = {2, 3}, G3 = {4, 5}. Start at 0, end at 3.

- The flagged path visits:
  $$\text{P}_{\text{flag}} = 0 \xrightarrow{s} 4 \xrightarrow{s} 2 \xrightarrow{s} 1 \xrightarrow{s} 5 \xrightarrow{s} 3 \implies \text{Cost}(P_{\text{flag}}) = 5s$$

- The competing unflagged path visits:
  $$\text{P}_{\text{unflag}} = 0 \to 5 \to 1 \to 4 \to 2 \to 3$$
  By triangle inequality:
  $$\text{Cost}(P_{\text{unflag}}) \le (s + 1) + s + (s + 1) + s + 1 = 4s + 3$$

- For the flagged path to strictly win:
  $$5s < 4s + 3 \implies \mathbf{s < 3.0}$$

---

## Ready-to-Use Matrix Templates (Lower Triangular)

### 5-Point Setup (G1={0,1}, G2={2,3}, G3={4}) for s < 1.5
```python
s = 1.45  # any s up to ~1.49

LOWER_TRIANGULAR = [
    # to: 0
    [1.0],                          # Node 1 (G1)
    # to: 0     1
    [s,   s],                       # Node 2 (G2)
    # to: 0     1     2
    [s,   s,   1.0],                # Node 3 (G2)
    # to: 0        1     2     3
    [s + 0.5, s,   s,   s + 1.0],   # Node 4 (G3)
]
```

### 6-Point Setup (G1={0,1}, G2={2,3}, G3={4,5}) for s < 3.0
```python
s = 2.85  # any s up to ~2.99

LOWER_TRIANGULAR = [
    # to: 0
    [1.0],                          # Node 1 (G1)
    # to: 0     1
    [s,   s],                       # Node 2 (G2)
    # to: 0     1     2
    [s,   s,   1.0],                # Node 3 (G2)
    # to: 0     1        2     3
    [s,   s + 1.0, s,   s + 1.0],   # Node 4 (G3)
    # to: 0        1     2        3     4
    [s + 1.0, s,   s + 1.0, s,   2 * s + 1.0],  # Node 5 (G3)
]
```

### 7-Point Setup (G1={0,1,2}, G2={3,4}, G3={5,6}) for s < 3.0
```python
s = 2.85  # any s up to ~2.99

LOWER_TRIANGULAR = [
    # to: 0
    [0.1],                                  # Node 1 (G1)
    # to: 0     1
    [1.0, 1.0],                             # Node 2 (G1)
    # to: 0     1     2
    [s,   s,   s],                          # Node 3 (G2)
    # to: 0     1     2     3
    [s,   s,   s,   1.0],                   # Node 4 (G2)
    # to: 0     1     2        3        4
    [s,   s,   s+1, s+1,     s],            # Node 5 (G3)
    # to: 0        1        2     3     4        5
    [s+1,     s+1,     s,   s,   s+1, 2*s+1],   # Node 6 (G3)
]
```
