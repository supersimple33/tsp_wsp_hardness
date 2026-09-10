"""
Objective: Test the hypothesis that for 4 groups of points (G1, G2, G3, G4)
the optimal hamiltonian path from a point in G1 to a point in G2 will never
have two disjoint subpaths that connect G1 and G2. (ie the following is 
impossible: G1 -> G3 -> G2 ->  G1 -> G4 -> G2).
Given the distance of any two points within G1 is at most one and the distance
between any two points in G2 is at most one. And the distance between a point
within G1 and any outside point is at least :math:`s`, and the distance between a point
within G2 and any outside point is at least :math:`s`.
"""

import numpy as np
from numba import njit, prange
from tqdm import tqdm
import math

# --- 1. Inline 64-bit PRNG (Reproducible without MT19937 overhead) ---

@njit(fastmath=True, inline='always')
def get_prng_state(base_seed, trial_idx):
    x = np.uint64(base_seed) + np.uint64(trial_idx) * np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))

@njit(fastmath=True, inline='always')
def next_u32(state):
    state = state + np.uint64(0x9E3779B97F4A7C15)
    z = state
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return state, np.uint32(z >> np.uint64(32))

@njit(fastmath=True, inline='always')
def next_f32(state):
    state, u = next_u32(state)
    return state, np.float32(u) * np.float32(1.0 / 4294967296.0)

@njit(fastmath=True, inline='always')
def next_uniform(state, low, high):
    state, f = next_f32(state)
    return state, low + f * (high - low)


# --- 2. Geometry Generation (Zero sqrt in rejection loops for k=2) ---

@njit(fastmath=True)
def generate_points_2d(points, s, s_sq, scale, state):
    while True:
        # P0 at origin
        points[0, 0] = 0.0
        points[0, 1] = 0.0
        
        # P1 in unit disk around P0 (polar sampling)
        state, u1 = next_f32(state)
        state, u2 = next_f32(state)
        r = np.float32(math.sqrt(u1))
        theta = np.float32(2.0 * math.pi) * u2
        points[1, 0] = r * np.float32(math.cos(theta))
        points[1, 1] = r * np.float32(math.sin(theta))
        
        # P2: >= S from P0 and P1
        valid = False
        for _ in range(500):
            state, px = next_uniform(state, -scale, scale)
            state, py = next_uniform(state, -scale, scale)
            d0_sq = px * px + py * py
            dx1 = px - points[1, 0]
            dy1 = py - points[1, 1]
            d1_sq = dx1 * dx1 + dy1 * dy1
            if d0_sq >= s_sq and d1_sq >= s_sq:
                points[2, 0] = px
                points[2, 1] = py
                valid = True
                break
        if not valid: continue
        
        # P3 in unit disk around P2, >= S from P0 and P1
        valid = False
        for _ in range(500):
            state, u1 = next_f32(state)
            state, u2 = next_f32(state)
            r = np.float32(math.sqrt(u1))
            theta = np.float32(2.0 * math.pi) * u2
            p3x = points[2, 0] + r * np.float32(math.cos(theta))
            p3y = points[2, 1] + r * np.float32(math.sin(theta))
            d0_sq = p3x * p3x + p3y * p3y
            dx1 = p3x - points[1, 0]
            dy1 = p3y - points[1, 1]
            d1_sq = dx1 * dx1 + dy1 * dy1
            if d0_sq >= s_sq and d1_sq >= s_sq:
                points[3, 0] = p3x
                points[3, 1] = p3y
                valid = True
                break
        if not valid: continue
        
        # P4..P7: >= S from G1 (0, 1) and G2 (2, 3)
        failed = False
        for p_idx in range(4, 8):
            placed = False
            for _ in range(500):
                state, px = next_uniform(state, -scale, scale)
                state, py = next_uniform(state, -scale, scale)
                ok = True
                for j in range(4):
                    dx = px - points[j, 0]
                    dy = py - points[j, 1]
                    if dx * dx + dy * dy < s_sq:
                        ok = False
                        break
                if ok:
                    points[p_idx, 0] = px
                    points[p_idx, 1] = py
                    placed = True
                    break
            if not placed:
                failed = True
                break
        if failed: continue
        
        return state

@njit(fastmath=True, inline='always')
def fill_dist_matrix_2d(points, dist):
    for i in range(8):
        dist[i, i] = 0.0
        for j in range(i + 1, 8):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            d = np.float32(math.sqrt(dx * dx + dy * dy))
            dist[i, j] = d
            dist[j, i] = d


# --- 3. Branch-and-Bound TSP Search (Replaces 2,880 iterations) ---

@njit(fastmath=True, inline='always')
def search_pair(s, t, other_g1, other_g2, dist, best_cost, best_is_flagged):
    # mid[0] is other_g1, mid[5] is other_g2.
    # Testing mid[0] first finds a clustered path instantly, tightening best_cost.
    mid = (other_g1, 4, 5, 6, 7, other_g2)
    
    for i1 in range(6):
        n1 = mid[i1]
        c1 = dist[s, n1]
        if c1 >= best_cost: continue
        
        for i2 in range(6):
            if i2 == i1: continue
            n2 = mid[i2]
            c2 = c1 + dist[n1, n2]
            if c2 >= best_cost: continue
            
            for i3 in range(6):
                if i3 == i1 or i3 == i2: continue
                n3 = mid[i3]
                c3 = c2 + dist[n2, n3]
                if c3 >= best_cost: continue
                
                for i4 in range(6):
                    if i4 == i1 or i4 == i2 or i4 == i3: continue
                    n4 = mid[i4]
                    c4 = c3 + dist[n3, n4]
                    if c4 >= best_cost: continue
                    
                    for i5 in range(6):
                        if i5 == i1 or i5 == i2 or i5 == i3 or i5 == i4: continue
                        n5 = mid[i5]
                        c5 = c4 + dist[n4, n5]
                        if c5 >= best_cost: continue
                        
                        i6 = 15 - (i1 + i2 + i3 + i4 + i5)
                        n6 = mid[i6]
                        c6 = c5 + dist[n5, n6] + dist[n6, t]
                        
                        if c6 < best_cost:
                            best_cost = c6
                            # Flagged iff other_g2 (index 5) appears before other_g1 (index 0)
                            if i1 == 5 or (i1 != 0 and (i2 == 5 or (i2 != 0 and (i3 == 5 or (i3 != 0 and (i4 == 5 or (i4 != 0 and i5 == 5))))))):
                                best_is_flagged = True
                            else:
                                best_is_flagged = False
                                
    return best_cost, best_is_flagged

@njit(fastmath=True, inline='always')
def solve_tsp_flagged(dist):
    best_cost = np.float32(1e30)
    best_is_flagged = False
    
    # 4 combinations of start in G1 (0, 1) and end in G2 (2, 3)
    best_cost, best_is_flagged = search_pair(0, 2, 1, 3, dist, best_cost, best_is_flagged)
    best_cost, best_is_flagged = search_pair(0, 3, 1, 2, dist, best_cost, best_is_flagged)
    best_cost, best_is_flagged = search_pair(1, 2, 0, 3, dist, best_cost, best_is_flagged)
    best_cost, best_is_flagged = search_pair(1, 3, 0, 2, dist, best_cost, best_is_flagged)
    
    return best_is_flagged


# --- 4. Parallel Engine ---

@njit(parallel=True, fastmath=True)
def compute_flags_batch(batch_size, s, base_seed, batch_offset):
    flags_out = np.zeros(batch_size, dtype=np.int8)
    s_sq = np.float32(s * s)
    scale = np.float32(4.0 * s + 2.0)
    
    for i in prange(batch_size):
        global_trial_idx = batch_offset + i
        state = get_prng_state(base_seed, global_trial_idx)
        
        # Local thread-stack allocation (zero heap slicing, zero false-sharing)
        pts = np.empty((8, 2), dtype=np.float32)
        d_mat = np.empty((8, 8), dtype=np.float32)
        
        state = generate_points_2d(pts, s, s_sq, scale, state)
        fill_dist_matrix_2d(pts, d_mat)
        
        if solve_tsp_flagged(d_mat):
            flags_out[i] = 1
            
    return flags_out


# --- 5. Diagnostic Reporting for Rare Flag Events ---

def log_flagged_trial(seed, global_idx, s):
    state = get_prng_state(seed, global_idx)
    s_sq = np.float32(s * s)
    scale = np.float32(4.0 * s + 2.0)
    pts = np.zeros((8, 2), dtype=np.float32)
    d_mat = np.zeros((8, 8), dtype=np.float32)
    generate_points_2d(pts, s, s_sq, scale, state)
    fill_dist_matrix_2d(pts, d_mat)
    
    import itertools
    best_cost = 1e30
    best_path = None
    for s_node in [0, 1]:
        for e_node in [2, 3]:
            mid = [i for i in range(8) if i != s_node and i != e_node]
            for p in itertools.permutations(mid):
                path = (s_node,) + p + (e_node,)
                cost = sum(d_mat[path[j], path[j+1]] for j in range(7))
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                    
    report = [
        f"\nFlag Found! Logging trial config:",
        f"Trial {global_idx + 1}: Path contains disjoint subpaths from G1 -> G2.",
        f"Path sequence: {list(best_path)}",
        "Coordinates in path order:"
    ]
    for step_num, p_idx in enumerate(best_path):
        group = "G1" if p_idx in [0, 1] else ("G2" if p_idx in [2, 3] else ("G3" if p_idx in [4, 5] else "G4"))
        report.append(f"  Step {step_num + 1}: Index {p_idx} [{group}] -> ({pts[p_idx, 0]:.4f}, {pts[p_idx, 1]:.4f})")
    report.append("-" * 40)
    tqdm.write("\n".join(report))


def run_simulation(trials, k=2, seed=17, s=3.0, batch_size=50_000):
    print(f"Running {trials:,} trials in {k}D (Seed: {seed}, S={s}, Batch size: {batch_size:,})...")
    print("Pruning 2,880 permutations via branch-and-bound search...\n")
    
    total_flags = 0
    flag_logged = False
    num_batches = (trials + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Simulating batches", unit="batch", dynamic_ncols=True):
        current_batch_size = min(batch_size, trials - batch_idx * batch_size)
        batch_offset = batch_idx * batch_size
        
        flags = compute_flags_batch(current_batch_size, s, seed, batch_offset)
        b_violations = int(np.sum(flags))
        
        if b_violations > 0:
            total_flags += b_violations
            if not flag_logged:
                flag_logged = True
                first_local_idx = int(np.argmax(flags))
                log_flagged_trial(seed, batch_offset + first_local_idx, s)
                
    print("\n" + "-" * 40)
    print("Simulation Complete.")
    if total_flags == 0:
        print("Result: No disjoint subpaths satisfying the condition were observed.")
    else:
        print(f"Result: Flagged condition occurred {total_flags:,} times.")

if __name__ == "__main__":
    run_simulation(trials=100_000_000, k=2, seed=18, s=2.5, batch_size=20_000)