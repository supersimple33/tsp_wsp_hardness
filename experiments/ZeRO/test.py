# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numba>=0.67.0",
#     "numpy>=2.5.3",
#     "tqdm>=4.70.0",
# ]
# ///

"""
Objective: Test the hypothesis that for 3 groups of points (G1, G2, G3)
the optimal hamiltonian path from a point in G1 to a point in G2 will never
have two disjoint subpaths that connect G1 and G2. (ie the following is 
impossible: G1 -> G2 ->  G1 -> G3 -> G2).
Given the distance of any two points within G1 is at most one and the distance
between any two points in G2 is at most one. And the distance between a point
within G1 and any outside point is at least :math:`s`, and the distance between a point
within G2 and any outside point is at least :math:`s`. Importantly points in G3 can be
arbitrarily close to each other.
"""

import math
import argparse
import time
import numpy as np
from numba import njit, prange
from tqdm import tqdm


# ==============================================================================
# 1. Thread-Safe 64-bit SplitMix PRNG (Zero heap allocation, lock-free)
# ==============================================================================

@njit(fastmath=True, inline='always')
def get_prng_state(base_seed, trial_idx):
    """Initializes a distinct, robust 64-bit state for each trial."""
    x = np.uint64(base_seed) + np.uint64(trial_idx) * np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))

@njit(fastmath=True, inline='always')
def next_u32(state):
    """Generates next 32-bit unsigned integer and updated state."""
    state = state + np.uint64(0x9E3779B97F4A7C15)
    z = state
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return state, np.uint32(z >> np.uint64(32))

@njit(fastmath=True, inline='always')
def next_f32(state):
    """Generates uniform float in [0, 1)."""
    state, u = next_u32(state)
    return state, np.float32(u) * np.float32(1.0 / 4294967296.0)

@njit(fastmath=True, inline='always')
def next_uniform(state, low, high):
    """Generates uniform float in [low, high)."""
    state, f = next_f32(state)
    return state, low + f * (high - low)


@njit(fastmath=True, inline='always')
def sample_unit_ball_point(points, p_idx, center_idx, k, state):
    """
    Samples a point uniformly in the k-dimensional unit ball centered at
    points[center_idx] (or origin if center_idx < 0) and writes directly into points[p_idx].
    """
    while True:
        norm_sq = np.float32(0.0)
        d = 0
        while d < k:
            state, u1 = next_f32(state)
            state, u2 = next_f32(state)
            if u1 < 1e-7:
                u1 = np.float32(1e-7)
            mag = np.float32(math.sqrt(-2.0 * math.log(u1)))
            theta = np.float32(2.0 * math.pi) * u2
            z0 = mag * np.float32(math.cos(theta))
            z1 = mag * np.float32(math.sin(theta))
            
            points[p_idx, d] = z0
            norm_sq += z0 * z0
            d += 1
            if d < k:
                points[p_idx, d] = z1
                norm_sq += z1 * z1
                d += 1
                
        if norm_sq > 1e-12:
            inv_norm = np.float32(1.0 / math.sqrt(norm_sq))
            state, u = next_f32(state)
            r = np.float32(math.pow(u, 1.0 / float(k)))
            factor = r * inv_norm
            for d_idx in range(k):
                c = np.float32(0.0) if center_idx < 0 else points[center_idx, d_idx]
                points[p_idx, d_idx] = c + points[p_idx, d_idx] * factor
            break
            
    return state


# ==============================================================================
# 2. Geometry Generation with Configurable Dimension k and G3 Size
# ==============================================================================

@njit(fastmath=True, inline='always')
def dist_sq_pts(points, i, j, k):
    """Computes squared Euclidean distance between points i and j in k dimensions."""
    d_sq = np.float32(0.0)
    for d in range(k):
        diff = points[i, d] - points[j, d]
        d_sq += diff * diff
    return d_sq


@njit(fastmath=True)
def generate_points_kd(points, n_g3, k, s, s_sq, scale, state):
    """
    Generates k-dimensional coordinates for:
      - G1 = {0, 1}: diam(G1) <= 1.0 (P0 at origin, P1 in unit ball)
      - G2 = {2, 3}: diam(G2) <= 1.0 (P2 >= s from G1, P3 in unit ball around P2 >= s from G1)
      - G3 = {4 .. 3+n_g3}: all points >= s from G1 and G2 (arbitrarily close to each other)
    """
    n_total = 4 + n_g3
    while True:
        # P0 at origin (G1)
        for d in range(k):
            points[0, d] = 0.0
        
        # P1 in unit ball around P0 (G1)
        state = sample_unit_ball_point(points, 1, 0, k, state)
        
        # P2: First point of G2 (must be >= s from P0 and P1)
        valid = False
        for _ in range(500):
            for d in range(k):
                state, val = next_uniform(state, -scale, scale)
                points[2, d] = val
            if dist_sq_pts(points, 2, 0, k) >= s_sq and dist_sq_pts(points, 2, 1, k) >= s_sq:
                valid = True
                break
        if not valid:
            continue
        
        # P3: Second point of G2 (in unit ball around P2, and >= s from G1)
        valid = False
        for _ in range(500):
            state = sample_unit_ball_point(points, 3, 2, k, state)
            if dist_sq_pts(points, 3, 0, k) >= s_sq and dist_sq_pts(points, 3, 1, k) >= s_sq:
                valid = True
                break
        if not valid:
            continue
        
        # P4 .. P(3+n_g3): G3 points (must be >= s from G1 and G2)
        failed = False
        for p_idx in range(4, n_total):
            placed = False
            for _ in range(500):
                for d in range(k):
                    state, val = next_uniform(state, -scale, scale)
                    points[p_idx, d] = val
                ok = True
                for j in range(4):
                    if dist_sq_pts(points, p_idx, j, k) < s_sq:
                        ok = False
                        break
                if ok:
                    placed = True
                    break
            if not placed:
                failed = True
                break
        if failed:
            continue
        
        return state

@njit(fastmath=True, inline='always')
def fill_dist_matrix_kd(points, dist, k):
    """Fills the pairwise Euclidean distance matrix for k-dimensional points."""
    n = points.shape[0]
    for i in range(n):
        dist[i, i] = 0.0
        for j in range(i + 1, n):
            d_sq = np.float32(0.0)
            for d in range(k):
                diff = points[i, d] - points[j, d]
                d_sq += diff * diff
            d = np.float32(math.sqrt(d_sq))
            dist[i, j] = d
            dist[j, i] = d


# ==============================================================================
# 3. Exact Optimal Hamiltonian Path Solver (Held-Karp Dynamic Programming)
# ==============================================================================

@njit(fastmath=True)
def held_karp_optimal_path(dist, start_node, end_node):
    """
    Finds the exact shortest Hamiltonian path from start_node to end_node
    using the Held-Karp dynamic programming algorithm over bitmasks.
    Returns: (path, cost)
    """
    n = dist.shape[0]
    num_subsets = 1 << n
    dp = np.full((num_subsets, n), 1e30, dtype=np.float32)
    parent = np.full((num_subsets, n), -1, dtype=np.int8)
    
    dp[1 << start_node, start_node] = 0.0
    full_mask = num_subsets - 1
    
    for mask in range(1, num_subsets):
        if not (mask & (1 << start_node)):
            continue
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            cost = dp[mask, u]
            if cost >= 1e29:
                continue
            if u == end_node and mask != full_mask:
                continue
            for v in range(n):
                if not (mask & (1 << v)):
                    if v == end_node and (mask | (1 << v)) != full_mask:
                        continue
                    new_mask = mask | (1 << v)
                    new_cost = cost + dist[u, v]
                    if new_cost < dp[new_mask, v]:
                        dp[new_mask, v] = new_cost
                        parent[new_mask, v] = u
                        
    # Reconstruct optimal path sequence
    curr = end_node
    mask = full_mask
    path = np.zeros(n, dtype=np.int8)
    for step in range(n - 1, -1, -1):
        path[step] = curr
        p = parent[mask, curr]
        mask = mask ^ (1 << curr)
        curr = p
        
    return path, dp[full_mask, end_node]

@njit(fastmath=True, inline='always')
def is_path_flagged(path, s_node, e_node):
    """
    Checks if the path contains two disjoint subpaths connecting G1 and G2.
    For G1={0, 1} and G2={2, 3}, with path starting at s_node in G1 and ending
    at e_node in G2:
      - The other G1 node is other_g1 = 1 if s_node == 0 else 0
      - The other G2 node is other_g2 = 3 if e_node == 2 else 2
    Two disjoint subpaths connecting G1 and G2 exist if and only if
    other_g2 appears BEFORE other_g1 in the path:
      s_node (G1) ... other_g2 (G2) ... other_g1 (G1) ... e_node (G2)
    """
    other_g1 = 1 if s_node == 0 else 0
    other_g2 = 3 if e_node == 2 else 2
    pos_g1 = -1
    pos_g2 = -1
    n = len(path)
    for i in range(n):
        if path[i] == other_g1:
            pos_g1 = i
        elif path[i] == other_g2:
            pos_g2 = i
    return pos_g2 < pos_g1

@njit(fastmath=True)
def evaluate_all_pairs(dist):
    """
    Evaluates optimal Hamiltonian paths for all 4 start/end pairs:
      (0, 2), (0, 3), (1, 2), (1, 3).
    Returns packed flags:
      bit 0 (1): global_flagged (the single overall shortest path is flagged)
      bit 1 (2): any_pair_flagged (at least one pair's optimal path is flagged)
    """
    pairs = ((0, 2), (0, 3), (1, 2), (1, 3))
    global_min_cost = np.float32(1e30)
    global_is_flagged = False
    any_pair_is_flagged = False
    
    for s_node, e_node in pairs:
        path, cost = held_karp_optimal_path(dist, s_node, e_node)
        flagged = is_path_flagged(path, s_node, e_node)
        
        if flagged:
            any_pair_is_flagged = True
            
        if cost < global_min_cost:
            global_min_cost = cost
            global_is_flagged = flagged
            
    flag_mask = np.int8(0)
    if global_is_flagged:
        flag_mask |= np.int8(1)
    if any_pair_is_flagged:
        flag_mask |= np.int8(2)
        
    return flag_mask


# ==============================================================================
# 4. Parallel Simulation Batch Engine
# ==============================================================================

@njit(parallel=True, fastmath=True)
def compute_flags_batch(batch_size, n_g3, k, s, base_seed, batch_offset):
    """Processes a batch of trials in parallel across CPU cores."""
    flags_out = np.zeros(batch_size, dtype=np.int8)
    n_total = 4 + n_g3
    s_sq = np.float32(s * s)
    scale = np.float32(4.0 * s + 2.0)
    
    for i in prange(batch_size):
        global_trial_idx = batch_offset + i
        state = get_prng_state(base_seed, global_trial_idx)
        
        pts = np.empty((n_total, k), dtype=np.float32)
        d_mat = np.empty((n_total, n_total), dtype=np.float32)
        
        state = generate_points_kd(pts, n_g3, k, s, s_sq, scale, state)
        fill_dist_matrix_kd(pts, d_mat, k)
        
        flags_out[i] = evaluate_all_pairs(d_mat)
        
    return flags_out


# ==============================================================================
# 5. Diagnostic Reporting & Independent Verification
# ==============================================================================

@njit(fastmath=True)
def regenerate_trial_instance(seed, global_idx, s, n_g3, k):
    """
    Reconstructs the exact instance within Numba to match the PRNG bit operations.
    """
    n_total = 4 + n_g3
    s_sq = np.float32(s * s)
    scale = np.float32(4.0 * s + 2.0)
    state = get_prng_state(seed, global_idx)
    pts = np.empty((n_total, k), dtype=np.float32)
    d_mat = np.empty((n_total, n_total), dtype=np.float32)
    generate_points_kd(pts, n_g3, k, s, s_sq, scale, state)
    fill_dist_matrix_kd(pts, d_mat, k)
    return pts, d_mat


def log_flagged_trial(seed, global_idx, s, n_g3, k):
    """
    Reconstructs and thoroughly reports on a counterexample trial.
    Validates all metric constraints and prints the optimal paths for all pairs.
    """
    n_total = 4 + n_g3
    pts, d_mat = regenerate_trial_instance(seed, global_idx, s, n_g3, k)
    
    # Verify metric constraints
    d_g1 = np.linalg.norm(pts[0] - pts[1])
    d_g2 = np.linalg.norm(pts[2] - pts[3])
    min_d_g1_out = min(d_mat[u, v] for u in (0, 1) for v in range(2, n_total))
    min_d_g2_out = min(d_mat[u, v] for u in (2, 3) for v in (0, 1) + tuple(range(4, n_total)))
    
    report = [
        "",
        "=" * 60,
        f"🚨 COUNTEREXAMPLE FOUND! Trial {global_idx + 1:,} (Seed: {seed})",
        "=" * 60,
        f"Separation parameter s = {s:.4f}, Dimension k = {k}, G3 size = {n_g3}, Total points = {n_total}",
        "",
        "--- Metric Constraint Checks ---",
        f"  diam(G1) = dist(P0, P1) = {d_g1:.4f} <= 1.0 : {'VALID' if d_g1 <= 1.0001 else 'INVALID'}",
        f"  diam(G2) = dist(P2, P3) = {d_g2:.4f} <= 1.0 : {'VALID' if d_g2 <= 1.0001 else 'INVALID'}",
        f"  min dist(G1, outside)   = {min_d_g1_out:.4f} >= s : {'VALID' if min_d_g1_out >= s - 1e-4 else 'INVALID'}",
        f"  min dist(G2, outside)   = {min_d_g2_out:.4f} >= s : {'VALID' if min_d_g2_out >= s - 1e-4 else 'INVALID'}",
        "",
        "--- Optimal Hamiltonian Paths for All 4 Endpoint Pairs ---"
    ]
    
    pairs = ((0, 2), (0, 3), (1, 2), (1, 3))
    best_flagged_cost = float('inf')
    best_flagged_path = None
    best_flagged_pair = None
    
    best_global_cost = float('inf')
    best_global_path = None
    best_global_pair = None
    
    for s_node, e_node in pairs:
        path, cost = held_karp_optimal_path(d_mat, s_node, e_node)
        flagged = is_path_flagged(path, s_node, e_node)
        path_list = [int(p) for p in path]
        group_seq = ["G1" if p in (0, 1) else ("G2" if p in (2, 3) else "G3") for p in path_list]
        
        status = "FLAGGED (G1 -> G2 -> G1 -> G2 subpaths)" if flagged else "Normal (unflagged)"
        report.append(f"  Pair ({s_node}, {e_node}): Length = {cost:.4f} [{status}]")
        report.append(f"    Path:   {path_list}")
        report.append(f"    Groups: {' -> '.join(group_seq)}")
        
        if cost < best_global_cost:
            best_global_cost = cost
            best_global_path = path_list
            best_global_pair = (s_node, e_node)
            
        if flagged and cost < best_flagged_cost:
            best_flagged_cost = cost
            best_flagged_path = path_list
            best_flagged_pair = (s_node, e_node)
            
    # Report the shortest path that flags (or fallback to global if none flagged)
    target_pair = best_flagged_pair if best_flagged_path is not None else best_global_pair
    target_cost = best_flagged_cost if best_flagged_path is not None else best_global_cost
    target_path = best_flagged_path if best_flagged_path is not None else best_global_path
    
    report.append("")
    report.append(f"--- Shortest Flagged Hamiltonian Path (Pair {target_pair}) ---")
    report.append(f"  Optimal Length: {target_cost:.4f}")
    target_group_seq = ["G1" if p in (0, 1) else ("G2" if p in (2, 3) else "G3") for p in target_path]
    report.append(f"  Group Sequence: {' -> '.join(target_group_seq)}")
    report.append("  Coordinates in path order:")
    for step_num, p_idx in enumerate(target_path):
        group = "G1" if p_idx in (0, 1) else ("G2" if p_idx in (2, 3) else "G3")
        coord_str = ", ".join(f"{pts[p_idx, d]:.4f}" for d in range(k))
        report.append(f"    Step {step_num + 1}: Index {p_idx} [{group}] -> ({coord_str})")
    report.append("=" * 60 + "\n")
    tqdm.write("\n".join(report))


# ==============================================================================
# 6. Main Simulation Runner & Parameter Sweep
# ==============================================================================

def run_simulation(trials=100_000, n_g3=4, k=2, s=1.5, seed=18, batch_size=20_000, check_mode="both"):
    """
    Executes the Monte Carlo test across the specified number of trials.
    
    check_mode options:
      - 'global': Flagged only if the single overall shortest path among all 4 pairs is flagged.
      - 'any_pair': Flagged if any of the 4 endpoint pairs has a flagged optimal path.
      - 'both': Track and report both conditions.
    """
    n_total = 4 + n_g3
    print(f"Running {trials:,} trials in {k}D Euclidean space:")
    print(f"  - Separation factor s = {s}")
    print(f"  - Dimension k = {k}")
    print(f"  - Groups: G1 (2 points), G2 (2 points), G3 ({n_g3} points) -> Total {n_total} points")
    print(f"  - Random seed = {seed}, Batch size = {batch_size:,}")
    print(f"  - Reporting mode = '{check_mode}'")
    print("Solving exact optimal Hamiltonian paths via Held-Karp DP...\n")
    
    total_global_flags = 0
    total_any_flags = 0
    first_flag_logged = False
    
    num_batches = (trials + batch_size - 1) // batch_size
    start_time = time.time()
    
    for batch_idx in tqdm(range(num_batches), desc="Simulating batches", unit="batch", dynamic_ncols=True):
        current_batch_size = min(batch_size, trials - batch_idx * batch_size)
        batch_offset = batch_idx * batch_size
        
        flags = compute_flags_batch(current_batch_size, n_g3, k, s, seed, batch_offset)
        
        global_flags = int(np.sum(flags & 1))
        any_flags = int(np.sum((flags & 2) >> 1))
        
        total_global_flags += global_flags
        total_any_flags += any_flags
        
        # Log details for the first observed violation
        if not first_flag_logged:
            condition = False
            if check_mode == "global" and global_flags > 0:
                first_local_idx = int(np.argmax(flags & 1))
                condition = True
            elif check_mode in ("any_pair", "both") and any_flags > 0:
                first_local_idx = int(np.argmax((flags & 2) >> 1))
                condition = True
            elif check_mode == "both" and global_flags > 0:
                first_local_idx = int(np.argmax(flags & 1))
                condition = True
                
            if condition:
                first_flag_logged = True
                log_flagged_trial(seed, batch_offset + first_local_idx, s, n_g3, k)
                
    elapsed = time.time() - start_time
    rate = trials / elapsed if elapsed > 0 else 0
    
    print("-" * 50)
    print("Simulation Complete.")
    print(f"Total trials: {trials:,} in {elapsed:.2f}s ({rate:,.0f} trials/sec)")
    print(f"Results:")
    print(f"  - Globally optimal path flagged:  {total_global_flags:,} times ({100 * total_global_flags / trials:.4f}%)")
    print(f"  - Any endpoint pair path flagged: {total_any_flags:,} times ({100 * total_any_flags / trials:.4f}%)")
    
    if total_any_flags == 0 and total_global_flags == 0:
        print("Conclusion: Hypothesis holds! No disjoint subpaths connecting G1 and G2 were observed.")
    else:
        print("Conclusion: Hypothesis DISPROVED for this configuration! Disjoint subpaths were observed.")
    print("-" * 50)
    
    return total_global_flags, total_any_flags


def run_sweep(s_values=None, trials_per_s=50_000, n_g3=4, k=2, seed=18, batch_size=20_000, check_mode="both"):
    """Sweeps multiple values of s to locate the empirical transition threshold."""
    if s_values is None:
        s_values = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5]
    print("=" * 60)
    print(f"Starting parameter sweep over s: {s_values}")
    print(f"Trials per s = {trials_per_s:,}, Dimension k = {k}, G3 size = {n_g3}, Seed = {seed}")
    print("=" * 60 + "\n")
    
    results = []
    for s_val in s_values:
        print(f"\n--- Testing s = {s_val:.2f} ---")
        g_flags, a_flags = run_simulation(
            trials=trials_per_s,
            n_g3=n_g3,
            k=k,
            s=s_val,
            seed=seed,
            batch_size=batch_size,
            check_mode=check_mode
        )
        results.append((s_val, g_flags, a_flags))
        
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY TABLE")
    print("=" * 60)
    print(f"{'s':>6} | {'Global Flags':>14} | {'Any Pair Flags':>16} | {'Status':>12}")
    print("-" * 60)
    for s_val, g_flags, a_flags in results:
        status = "VIOLATION" if a_flags > 0 else "HOLDS"
        print(f"{s_val:6.2f} | {g_flags:14,d} | {a_flags:16,d} | {status:>12}")
    print("=" * 60)


# ==============================================================================
# 7. CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the hypothesis that an optimal Hamiltonian path from G1 to G2 never contains two disjoint subpaths connecting G1 and G2."
    )
    parser.add_argument("-k", "--k", type=int, default=3, help="Number of spatial dimensions (default: 2)")
    parser.add_argument("--trials", type=int, default=100_000_000, help="Number of Monte Carlo trials (default: 100,000,000)")
    parser.add_argument("--s", type=float, default=2.75, help="Separation factor s (default: 2.75)")
    parser.add_argument("--n-g3", type=int, default=3, help="Number of points in G3 (default: 3)")
    parser.add_argument("--seed", type=int, default=111, help="PRNG base seed (default: 11)")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Numba parallel batch size (default: 100,000)")
    parser.add_argument("--mode", choices=["both", "global", "any_pair"], default="any_pair", help="Reporting mode (default: any_pair)")
    parser.add_argument(
        "--sweep", 
        nargs="*", 
        type=float, 
        default=None, 
        metavar="S",
        help="Run a sweep over separation factor s values. Can optionally specify custom values: --sweep 1.5 2.0 2.5 3.0 (defaults to [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5] if flag is passed with no arguments)"
    )
    parser.add_argument(
        "--s-range",
        nargs=3,
        type=float,
        default=None,
        metavar=("START", "STOP", "STEP"),
        help="Sweep s values generated via START, STOP, STEP (e.g. --s-range 1.5 3.0 0.25)"
    )
    
    args = parser.parse_args()
    
    if args.k < 1:
        parser.error("Number of spatial dimensions k must be >= 1.")
    
    if args.sweep is not None or args.s_range is not None:
        if args.s_range is not None:
            start, stop, step = args.s_range
            s_range = [round(float(x), 4) for x in np.arange(start, stop + step * 0.5, step)]
        elif len(args.sweep) > 0:
            s_range = args.sweep
        else:
            s_range = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5]
        run_sweep(s_values=s_range, trials_per_s=args.trials, n_g3=args.n_g3, k=args.k, seed=args.seed, batch_size=args.batch_size, check_mode=args.mode)
    else:
        run_simulation(
            trials=args.trials,
            n_g3=args.n_g3,
            k=args.k,
            s=args.s,
            seed=args.seed,
            batch_size=args.batch_size,
            check_mode=args.mode
        )
