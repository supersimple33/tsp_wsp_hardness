import numpy as np
from numba import njit, prange
from tqdm import tqdm
import math
import itertools

@njit
def set_seed(seed):
    np.random.seed(seed)

@njit(fastmath=True, inline='always')
def calc_dist(p1, p2):
    d = 0.0
    for i in range(len(p1)):
        diff = p1[i] - p2[i]
        d += diff * diff
    return math.sqrt(d)

@njit(fastmath=True)
def generate_points(k, s):
    # Indices mapping: 
    # 0, 1 -> G1 (2-point group, clustered within radius 1)
    # 2, 3 -> G2 (2-point group, clustered within radius 1)
    # 4, 5 -> G3 (2-point group, unconstrained internally, >= S from G1 & G2)
    # 6, 7 -> G4 (2-point group, unconstrained internally, >= S from G1 & G2)
    points = np.zeros((8, k), dtype=np.float32)
    temp_p = np.zeros(k, dtype=np.float32)
    
    scale = 4.0 * s + 2.0 
    
    while True:
        # 1. Place P0 (First point of G1) at origin
        for d in range(k):
            points[0, d] = 0.0
            
        # 2. Place P1 (Second point of G1) uniformly within radius 1 of P0
        norm_sq = 0.0
        for d in range(k):
            v_d = np.random.normal(0.0, 1.0)
            temp_p[d] = v_d
            norm_sq += v_d * v_d
        norm_v = math.sqrt(norm_sq)
        if norm_v == 0.0: norm_v = 1.0
        
        r = (np.random.uniform(0.0, 1.0)**(1.0/k)) * 1.0
        for d in range(k):
            points[1, d] = temp_p[d] * (r / norm_v)
            
        # 3. Place P2 (First point of G2), must be >= S from G1 (P0, P1)
        valid = False
        for _ in range(500):
            for d in range(k):
                temp_p[d] = np.random.uniform(-scale, scale)
            if calc_dist(temp_p, points[0]) >= s and calc_dist(temp_p, points[1]) >= s:
                for d in range(k):
                    points[2, d] = temp_p[d]
                valid = True
                break
        if not valid: continue
        
        # 4. Place P3 (Second point of G2) uniformly within radius 1 of P2, and >= S from G1
        valid = False
        for _ in range(500):
            norm_sq = 0.0
            for d in range(k):
                v_d = np.random.normal(0.0, 1.0)
                temp_p[d] = v_d
                norm_sq += v_d * v_d
            norm_v = math.sqrt(norm_sq)
            if norm_v == 0.0: norm_v = 1.0
            
            r = (np.random.uniform(0.0, 1.0)**(1.0/k)) * 1.0
            for d in range(k):
                temp_p[d] = points[2, d] + temp_p[d] * (r / norm_v)
            
            if calc_dist(temp_p, points[0]) >= s and calc_dist(temp_p, points[1]) >= s:
                for d in range(k):
                    points[3, d] = temp_p[d]
                valid = True
                break
        if not valid: continue
        
        # 5. Place Points 4, 5 (G3) and 6, 7 (G4)
        # Only required to be >= S from G1 (0, 1) and G2 (2, 3)
        failed = False
        for p_idx in range(4, 8):
            pt_placed = False
            for _ in range(500):
                for d in range(k):
                    temp_p[d] = np.random.uniform(-scale, scale)
                
                # Check separation only against points in G1 and G2
                ok = True
                for j in range(4):
                    if calc_dist(temp_p, points[j]) < s:
                        ok = False
                        break
                if ok:
                    for d in range(k):
                        points[p_idx, d] = temp_p[d]
                    pt_placed = True
                    break
            if not pt_placed:
                failed = True
                break
        if failed: continue
        
        return points

def get_permutations():
    """Precomputes all Hamiltonian path permutations for 8 nodes."""
    # Start: Any point in G1 (0, 1). End: Any point in G2 (2, 3).
    # Remaining 6 middle nodes permuted (2 * 2 * 6! = 2,880 paths).
    perms = []
    for start_node in [0, 1]:
        for end_node in [2, 3]:
            middle_nodes = [i for i in range(8) if i != start_node and i != end_node]
            for p in itertools.permutations(middle_nodes):
                perms.append((start_node,) + p + (end_node,))
                
    return np.array(perms, dtype=np.int32)

@njit(fastmath=True)
def solve_optimal_hamiltonian_path(points, perms):
    n = 8
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = calc_dist(points[i], points[j])
            dist[i, j] = d
            dist[j, i] = d
    
    best_cost = np.inf
    best_idx = -1
    
    num_perms = len(perms)
    for i in range(num_perms):
        cost = 0.0
        for j in range(n - 1):
            u = perms[i, j]
            v = perms[i, j + 1]
            cost += dist[u, v]
            if cost >= best_cost:
                break
                
        if cost < best_cost:
            best_cost = cost
            best_idx = i
            
    return perms[best_idx], best_cost

@njit(fastmath=True)
def is_flagged(path):
    """
    Checks if there are 2 disjoint subpaths from G1 to G2.
    Scans the 8-node path for the exact visit sequence G1 -> G2 -> G1 -> G2.
    """
    g_seq = np.zeros(4, dtype=np.int32)
    idx = 0
    for i in range(8):
        node = path[i]
        if node == 0 or node == 1:
            g_seq[idx] = 0
            idx += 1
        elif node == 2 or node == 3:
            g_seq[idx] = 1
            idx += 1
            
    if g_seq[0] == 0 and g_seq[1] == 1 and g_seq[2] == 0 and g_seq[3] == 1:
        return True
    return False

@njit(parallel=True, fastmath=True)
def compute_flags_batch(batch_size, k, s, base_seed, batch_offset, perms):
    flags_out = np.zeros(batch_size, dtype=np.int8)
    
    for i in prange(batch_size):
        global_trial_idx = batch_offset + i
        np.random.seed(base_seed + global_trial_idx * 19937)
        
        points = generate_points(k, s)
        path, cost = solve_optimal_hamiltonian_path(points, perms)
        
        if is_flagged(path):
            flags_out[i] = 1
            
    return flags_out

@njit(fastmath=True)
def run_simulation_batch(batch_size, k, s, base_seed, batch_offset, perms):
    flags_out = compute_flags_batch(batch_size, k, s, base_seed, batch_offset, perms)
    
    violations = 0
    best_local_idx = -1
    
    for i in range(batch_size):
        if flags_out[i] == 1:
            violations += 1
            if best_local_idx == -1:
                best_local_idx = i
                
    if best_local_idx != -1:
        worst_global_idx = batch_offset + best_local_idx
        np.random.seed(base_seed + worst_global_idx * 19937)
        worst_points = generate_points(k, s)
        worst_path, worst_cost = solve_optimal_hamiltonian_path(worst_points, perms)
        return violations, worst_points, worst_path, worst_global_idx + 1
        
    dummy_points = np.zeros((8, k), dtype=np.float32)
    dummy_path = np.zeros(8, dtype=np.int32)
    return violations, dummy_points, dummy_path, -1

def run_simulation(trials, k, seed, s, batch_size=2_000):
    print(f"Running {trials:,} trials in {k} dimensions (Seed: {seed}, S={s})...")
    print("Testing for disjoint subpaths from G1 to G2 across 8 total points.")
    print("Compiling Numba functions (the progress bar will pause briefly at 0%)...\n")
    
    total_flags = 0
    flag_logged = False
    
    perms = get_permutations()
    num_batches = (trials + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Simulating batches", unit="batch", dynamic_ncols=True):
        current_batch_size = min(batch_size, trials - batch_idx * batch_size)
        batch_offset = batch_idx * batch_size
        
        b_violations, b_points, b_path, b_trial_num = run_simulation_batch(
            current_batch_size, k, s, seed, batch_offset, perms
        )
        
        if b_violations > 0:
            total_flags += b_violations
            
            if not flag_logged:
                flag_logged = True
                
                report = [
                    f"\nFlag Found! Logging trial config:",
                    f"Trial {b_trial_num}: Path contains disjoint subpaths from G1 -> G2.",
                    f"Path sequence (by index): {list(b_path)}",
                    "Coordinates in path order:"
                ]
                
                for step_num, p_idx in enumerate(b_path):
                    if p_idx in [0, 1]: group = "G1 (2-pt)"
                    elif p_idx in [2, 3]: group = "G2 (2-pt)"
                    elif p_idx in [4, 5]: group = "G3 (2-pt)"
                    else: group = "G4 (2-pt)"
                    
                    coords = ", ".join([f"{c:.4f}" for c in b_points[p_idx]])
                    report.append(f"  Step {step_num + 1}: Index {p_idx} [{group}] -> ({coords})")
                report.append("-" * 40)
                
                tqdm.write("\n".join(report))
            
    print("\n" + "-" * 40)
    print("Simulation Complete.")
    if total_flags == 0:
        print("Result: No disjoint subpaths satisfying the condition were observed.")
    else:
        print(f"Result: Flagged condition occurred {total_flags:,} times.")

if __name__ == "__main__":
    run_simulation(trials=10_000_000_000, k=2, seed=17, s=3.0)