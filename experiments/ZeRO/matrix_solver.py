import numpy as np

# ==============================================================================
# Fill in your 6x6 lower triangular matrix values here:
#   Row 1: distances from node 1 to [0]
#   Row 2: distances from node 2 to [0, 1]
#   Row 3: distances from node 3 to [0, 1, 2]
#   Row 4: distances from node 4 to [0, 1, 2, 3]
#   Row 5: distances from node 5 to [0, 1, 2, 3, 4]
# ==============================================================================

S = 1.8

S2 = S*10

LOWER_TRIANGULAR = [
    # to: 0
    [1.0],                          # Node 1 (G1)
    # to: 0     1
    [S, S],                     # Node 2 (G2)
    # to: 0     1     2
    [S, S, 1.0],                # Node 3 (G2)
    # to: 0     1     2     3
    [S2, S2+1, S2, S2+1],           # Node 4 (G3)
    # to: 0     1     2     3     4
    [S2, S2, S2, S2, 2*S2],      # Node 5 (G3)
]


def build_matrix(tri):
    """Builds symmetric 6x6 matrix from lower triangular list."""
    n = 6
    dist = np.zeros((n, n), dtype=float)
    for r_idx, row in enumerate(tri):
        i = r_idx + 1  # Node 1 to 5
        for j, val in enumerate(row):
            dist[i, j] = dist[j, i] = float(val)
    return dist


def held_karp_path(dist, start, end):
    """Exact shortest Hamiltonian path from start to end using Held-Karp DP."""
    n = len(dist)
    num_subsets = 1 << n
    dp = np.full((num_subsets, n), np.inf)
    parent = np.full((num_subsets, n), -1, dtype=int)
    dp[1 << start, start] = 0.0

    for mask in range(1, num_subsets):
        if not (mask & (1 << start)):
            continue
        for u in range(n):
            if not (mask & (1 << u)) or dp[mask, u] == np.inf:
                continue
            if u == end and mask != num_subsets - 1:
                continue
            for v in range(n):
                if not (mask & (1 << v)):
                    if v == end and (mask | (1 << v)) != num_subsets - 1:
                        continue
                    nxt_mask = mask | (1 << v)
                    cost = dp[mask, u] + dist[u, v]
                    if cost < dp[nxt_mask, v]:
                        dp[nxt_mask, v] = cost
                        parent[nxt_mask, v] = u

    # Reconstruct path
    curr, mask = end, num_subsets - 1
    path = []
    while curr != -1 and mask != 0:
        path.append(int(curr))
        p = parent[mask, curr]
        mask ^= (1 << curr)
        curr = p
    return list(reversed(path)), dp[num_subsets - 1, end]


def held_karp_tsp(dist):
    """Exact shortest closed TSP tour starting and ending at node 0."""
    n = len(dist)
    num_subsets = 1 << n
    dp = np.full((num_subsets, n), np.inf)
    parent = np.full((num_subsets, n), -1, dtype=int)
    dp[1, 0] = 0.0

    for mask in range(1, num_subsets):
        if not (mask & 1):
            continue
        for u in range(n):
            if not (mask & (1 << u)) or dp[mask, u] == np.inf:
                continue
            for v in range(n):
                if not (mask & (1 << v)):
                    nxt_mask = mask | (1 << v)
                    cost = dp[mask, u] + dist[u, v]
                    if cost < dp[nxt_mask, v]:
                        dp[nxt_mask, v] = cost
                        parent[nxt_mask, v] = u

    best_last = min(range(1, n), key=lambda u: dp[num_subsets - 1, u] + dist[u, 0])
    min_cost = dp[num_subsets - 1, best_last] + dist[best_last, 0]

    curr, mask = best_last, num_subsets - 1
    path = []
    while curr != -1 and mask != 0:
        path.append(int(curr))
        p = parent[mask, curr]
        mask ^= (1 << curr)
        curr = p
    return list(reversed(path)) + [0], min_cost


def is_flagged(path, s, e):
    """Checks for disjoint subpaths: s (G1) ... G2 ... G1 ... e (G2)."""
    other_g1 = 1 if s == 0 else 0
    other_g2 = 3 if e == 2 else 2
    return path.index(other_g2) < path.index(other_g1)


def group(node):
    return "G1" if node in (0, 1) else ("G2" if node in (2, 3) else "G3")


def check_triangle_inequality(dist):
    n = len(dist)
    violations = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k != i and k != j and dist[i, j] > dist[i, k] + dist[k, j] + 1e-9:
                    violations.append((i, j, k))
    return violations


def main():
    dist = build_matrix(LOWER_TRIANGULAR)
    
    print("--- 6x6 Distance Matrix ---")
    headers = [f"{i}({group(i)})" for i in range(6)]
    print(f"{'':>7} " + " ".join(f"{h:>8}" for h in headers))
    for i in range(6):
        print(f"{headers[i]:>7} " + " ".join(f"{dist[i, j]:8.2f}" for j in range(6)))

    violations = check_triangle_inequality(dist)
    if violations:
        print(f"\n⚠️  Triangle inequality violated ({len(violations)} times). Example: d({violations[0][0]}, {violations[0][1]}) > d({violations[0][0]}, {violations[0][2]}) + d({violations[0][2]}, {violations[0][1]})")
    else:
        print("\n✓ Triangle inequality holds.")

    print("\n--- ZeRO Hypothesis Paths (G1 -> G2) ---")
    for s, e in [(0, 2), (0, 3), (1, 2), (1, 3)]:
        path, cost = held_karp_path(dist, s, e)
        flagged = is_flagged(path, s, e)
        flag_str = "🚨 FLAGGED (disjoint G1/G2 subpaths)" if flagged else "✓ Normal"
        group_str = " -> ".join(group(x) for x in path)
        print(f"Pair ({s}, {e}): length = {cost:.3f} [{flag_str}]")
        print(f"  Path:   {path}")
        print(f"  Groups: {group_str}")

    tour, tour_cost = held_karp_tsp(dist)
    print("\n--- Closed TSP Tour ---")
    print(f"Tour:   {tour}")
    print(f"Length: {tour_cost:.3f}")


if __name__ == "__main__":
    main()
