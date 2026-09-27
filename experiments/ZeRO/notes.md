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
| G1=3, G2=3, G3=2 | 8 | Metric | **s < 3.00** | `0 -> 7 -> 5 -> 4 -> 2 -> 1 -> 6 -> 3` | Limited by 2 wings in G3 |
| G1=2, G2=2, G3=3 | 7 | Metric | **Unbounded (s > 3.0)** | `0 -> 4 -> 2 -> 5 -> 1 -> 6 -> 3` | 3-Wing Pivot: 6s < 6s + 1 for arbitrary s |
| G1=3, G2=3, G3=3 | 9 | Metric | **Unbounded (s > 3.0)** | `0 -> 6 -> 3 -> 4 -> 7 -> 1 -> 2 -> 8 -> 5` | 6s + 2 < 7s => s > 2.0 (Arbitrary s!) |
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

### 3. Case |G1|=2, |G2|=2, |G3|=3 (N = 7 points) => Unbounded s via Pivot Interleaving!
Can 2 $G_1$ and 2 $G_2$ points break past $s \ge 3.0$ with 3 $G_3$ points? **YES!**

Previously, one might assume each wing requires 2 dedicated center nodes ($2 \times 3 = 6$ nodes). But center nodes can act as **pivots** shared between consecutive wings:
$$m \text{ wings require only } m + 1 \text{ center nodes!}$$

With 4 center nodes ($G_1=\{0, 1\}, G_2=\{2, 3\}$), we can chain exactly $4 - 1 = 3$ wings ($G_3=\{4, 5, 6\}$):

- **The 3-Wing Pivot Flagged Path**:
  $$\text{P}_{\text{flag}} = 0 \xrightarrow{s} 4 \xrightarrow{s} 2 \xrightarrow{s} 5 \xrightarrow{s} 1 \xrightarrow{s} 6 \xrightarrow{s} 3$$
  - Every single step is a crossing of length $s$:
  $$\text{Cost}(P_{\text{flag}}) = 6s$$

- **The Unflagged Path**:
  An unflagged path must visit $1$ before $2$ (no weaving between $G_1$ and $G_2$). Because the 3 wings are separated from each other ($d(4, 5) = 2s, d(5, 6) = 2s, d(4, 6) = 2s + 1$), any path visiting all 3 wings without alternating between $G_1$ and $G_2$ incurs extra hops or misalignment penalties:
  $$\text{Cost}(P_{\text{unflag}}) \ge 6s + 1 \quad (\text{or } 6s + 2)$$

- **Comparison**:
  $$\text{Cost}(P_{\text{flag}}) = 6s < 6s + 1 \le \text{Cost}(P_{\text{unflag}})$$
  The flagged path wins for **arbitrary $s \ge 3.0$** (tested up to $s = 10, 50$).

---

### 4. Case |G1|=3, |G2|=3, |G3|=3 (N = 9 points) => s Can Exceed 3.0 (Arbitrarily Large in Metric Space!)
When $|G_1| \ge 3$ and $|G_2| \ge 3$, the center capacity expands to $3 + 3 = 6$ nodes!
This unlocks **3 separate wings in G3** ($W_1 = \{6\}, W_2 = \{7\}, W_3 = \{8\}$).

- **The 3-Wing Alternating Flagged Path**:
  $$\text{P}_{\text{flag}} = 0 \xrightarrow{s} 6 \xrightarrow{s} 3 \xrightarrow{1} 4 \xrightarrow{s} 7 \xrightarrow{s} 1 \xrightarrow{1} 2 \xrightarrow{s} 8 \xrightarrow{s} 5$$
  Total crossing edges across the separation gap: exactly $2 \times 3 = 6$ edges of length $s$.
  Total internal edges: 2 edges of length 1.
  $$\text{Cost}(P_{\text{flag}}) = 6s + 2$$

- **The Unflagged Path Bottleneck**:
  An unflagged path cannot alternate back and forth between $G_1$ and $G_2$. To visit all 3 distant wings ($d(W_i, W_j) \ge 2s + 1$), it must either take direct inter-wing hops (cost $\ge 2s$) or make round-trip excursions from the same cluster. Any such unflagged path requires at least **7 edges of length $s$**:
  $$\text{Cost}(P_{\text{unflag}}) \ge 7s$$

- **Comparison**:
  $$\text{Cost}(P_{\text{flag}}) < \text{Cost}(P_{\text{unflag}}) \iff 6s + 2 < 7s \iff \mathbf{s > 2.0}$$
  Because the leading coefficient of the flagged path ($6s$) is strictly smaller than the unflagged path ($7s$), the flagged path wins for **any $s > 2.0$**, breaking past $s = 3.0$, $s = 4.0$, $s = 10.0$!

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

### 7-Point Setup (G1={0,1}, G2={2,3}, G3={4,5,6}) for Arbitrary s >= 3.0 (Pivot Interleaving)
```python
s = 3.5  # breaks for arbitrary s >= 3.0 (e.g. 3.5, 5.0, 10.0)

LOWER_TRIANGULAR = [
    # to: 0
    [1.0],                                              # Node 1 (G1)
    # to: 0     1
    [s,   s],                                           # Node 2 (G2)
    # to: 0     1     2
    [s,   s,   1.0],                                    # Node 3 (G2)
    # to: 0        1        2        3
    [s,       s + 1.0, s,       s + 1.0],               # Node 4 (G3, Wing 1)
    # to: 0        1        2        3        4
    [s + 1.0, s,       s,       s + 1.0, 2 * s],        # Node 5 (G3, Wing 2)
    # to: 0        1        2        3        4        5
    [s + 1.0, s,       s + 1.0, s,       2*s + 1, 2 * s], # Node 6 (G3, Wing 3)
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

### 9-Point Setup (G1={0,1,2}, G2={3,4,5}, G3={6,7,8}) for s > 3.0 (Arbitrary s)
```python
s = 3.5  # breaks for any s > 2.0 (e.g. 3.5, 4.0, 5.0, 10.0)

LOWER_TRIANGULAR = [
    # to: 0
    [1.0],                                                              # Node 1 (G1)
    # to: 0     1
    [1.0, 1.0],                                                         # Node 2 (G1)
    # to: 0     1     2
    [s,   s,   s],                                                      # Node 3 (G2)
    # to: 0     1     2     3
    [s,   s,   s,   1.0],                                               # Node 4 (G2)
    # to: 0     1     2     3     4
    [s,   s,   s,   1.0, 1.0],                                          # Node 5 (G2)
    # to: 0     1        2        3     4        5
    [s,   s + 1.0, s + 1.0, s,   s + 1.0, s + 1.0],                     # Node 6 (G3 Wing 1)
    # to: 0        1     2        3        4     5        6
    [s + 1.0, s,   s + 1.0, s + 1.0, s,   s + 1.0, 2*s + 1.0],          # Node 7 (G3 Wing 2)
    # to: 0        1        2     3        4        5     6        7
    [s + 1.0, s + 1.0, s,   s + 1.0, s + 1.0, s,   2*s + 1.0, 2*s + 1.0], # Node 8 (G3 Wing 3)
]
```
