import numpy as np

# ==============================================================================
# Parameterized 6x6 Lower Triangular Matrix (|G1|=2, |G2|=2, |G3|=2)
#   G1 = {0, 1}, G2 = {2, 3}, G3 = {4, 5}
#   Theoretical maximum is s < 3.0 in metric space
# ==============================================================================
s = 3.01

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


def get_preset_matrix(preset_name, s_val):
    """Returns lower triangular matrix for named presets."""
    if preset_name in ("5", "5pt"):
        # |G1|=2, |G2|=2, |G3|=1 (bound: s < 1.5)
        return [
            [1.0],
            [s_val, s_val],
            [s_val, s_val, 1.0],
            [s_val + 0.5, s_val, s_val, s_val + 1.0],
        ]
    elif preset_name in ("6", "6pt"):
        # |G1|=2, |G2|=2, |G3|=2 (bound: s < 3.0)
        return [
            [1.0],
            [s_val, s_val],
            [s_val, s_val, 1.0],
            [s_val, s_val + 1.0, s_val, s_val + 1.0],
            [s_val + 1.0, s_val, s_val + 1.0, s_val, 2 * s_val + 1.0],
        ]
    elif preset_name in ("7_g3", "7pt_g3"):
        # |G1|=2, |G2|=2, |G3|=3 (bound: s < 3.0)
        return [
            [1.0],
            [s_val, s_val],
            [s_val, s_val, 1.0],
            [s_val, s_val + 1.0, s_val, s_val + 1.0],
            [s_val + 1.0, s_val, s_val + 1.0, s_val, 2 * s_val + 1.0],
            [s_val, s_val + 1.0, s_val, s_val + 1.0, 0.0, 2 * s_val + 1.0],
        ]
    elif preset_name in ("7_g1", "7pt_g1"):
        # |G1|=3, |G2|=2, |G3|=2 (bound: s < 3.0)
        return [
            [0.1],
            [1.0, 1.0],
            [s_val, s_val, s_val],
            [s_val, s_val, s_val, 1.0],
            [s_val, s_val, s_val + 1.0, s_val + 1.0, s_val],
            [s_val + 1.0, s_val + 1.0, s_val, s_val, s_val + 1.0, 2 * s_val + 1.0],
        ]
    else:
        raise ValueError(f"Unknown preset: {preset_name}")


def build_matrix(tri):
    """Builds symmetric N x N matrix from lower triangular list."""
    n = len(tri) + 1
    dist = np.zeros((n, n), dtype=float)
    for r_idx, row in enumerate(tri):
        i = r_idx + 1
        for j, val in enumerate(row):
            dist[i, j] = dist[j, i] = float(val)
    return dist


def get_groups(n):
    """Returns group definitions based on matrix size."""
    if n == 8:
        return (0, 1, 2), (3, 4, 5), (6, 7)
    elif n == 7:
        return (0, 1, 2), (3, 4), (5, 6)
    elif n == 5:
        return (0, 1), (2, 3), (4,)
    else:
        return (0, 1), (2, 3), (4, 5)


def group(node, g1, g2, g3):
    return "G1" if node in g1 else ("G2" if node in g2 else "G3")


def is_flagged(path, s, e, g1, g2):
    """Checks for disjoint subpaths G1 -> G2 -> G1 -> G2."""
    other_g2 = [x for x in g2 if x != e]
    rem_g1 = [x for x in g1 if x != s]
    pos_g2_min = min(path.index(x) for x in other_g2)
    pos_g1_max = max(path.index(x) for x in rem_g1)
    return pos_g2_min < pos_g1_max


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


def check_triangle_inequality(dist):
    n = len(dist)
    violations = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k != i and k != j and dist[i, j] > dist[i, k] + dist[k, j] + 1e-9:
                    violations.append((i, j, k))
    return violations


def main(tri=None):
    if tri is None:
        tri = LOWER_TRIANGULAR
    dist = build_matrix(tri)
    n = len(dist)
    g1, g2, g3 = get_groups(n)
    
    print(f"--- {n}x{n} Distance Matrix (G1={g1}, G2={g2}, G3={g3}) ---")
    headers = [f"{i}({group(i, g1, g2, g3)})" for i in range(n)]
    print(f"{'':>7} " + " ".join(f"{h:>8}" for h in headers))
    for i in range(n):
        print(f"{headers[i]:>7} " + " ".join(f"{dist[i, j]:8.2f}" for j in range(n)))

    violations = check_triangle_inequality(dist)
    if violations:
        print(f"\n⚠️  Triangle inequality violated ({len(violations)} times). Example: d({violations[0][0]}, {violations[0][1]}) > d({violations[0][0]}, {violations[0][2]}) + d({violations[0][2]}, {violations[0][1]})")
    else:
        print("\n✓ Triangle inequality holds.")

    print("\n--- ZeRO Hypothesis Paths (G1 -> G2) ---")
    pairs = [(s_node, e_node) for s_node in g1 for e_node in g2]
    any_flagged = False
    for s_node, e_node in pairs:
        path, cost = held_karp_path(dist, s_node, e_node)
        flagged = is_flagged(path, s_node, e_node, g1, g2)
        if flagged:
            any_flagged = True
        flag_str = "🚨 FLAGGED (disjoint G1/G2 subpaths)" if flagged else "✓ Normal"
        group_str = " -> ".join(group(x, g1, g2, g3) for x in path)
        print(f"Pair ({s_node}, {e_node}): length = {cost:.3f} [{flag_str}]")
        print(f"  Path:   {path}")
        print(f"  Groups: {group_str}")

    if any_flagged:
        print("\n💥 RESULT: HYPOTHESIS BROKEN! Disjoint subpaths G1 -> G2 -> G1 -> G2 are optimal.")
    else:
        print("\n✓ RESULT: Hypothesis holds. No disjoint subpaths observed.")

    tour, tour_cost = held_karp_tsp(dist)
    print("\n--- Closed TSP Tour ---")
    print(f"Tour:   {tour}")
    print(f"Length: {tour_cost:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exact Held-Karp solver for ZeRO metric matrices.")
    parser.add_argument("--s", type=float, default=None, help="Separation factor s (default: value of s in script)")
    parser.add_argument("--preset", choices=["5", "6", "7_g3", "7_g1"], default=None, help="Preset matrix (5, 6, 7_g3, 7_g1)")
    args = parser.parse_args()
    
    if args.preset is not None or args.s is not None:
        p = args.preset if args.preset is not None else "6"
        s_val = args.s if args.s is not None else s
        tri = get_preset_matrix(p, s_val)
        print(f"Running Preset '{p}' with s = {s_val:.4f}\n")
        main(tri)
    else:
        main(LOWER_TRIANGULAR)
