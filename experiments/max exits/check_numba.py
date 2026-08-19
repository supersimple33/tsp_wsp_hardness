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
def generate_points(k, n_inner, n_outer, s):
    n_total = n_inner + n_outer
    points = np.zeros((n_total, k), dtype=np.float32)
    
    while True:
        # 1. Generate inner cloud
        for i in range(n_inner):
            norm_sq = 0.0
            for d in range(k):
                v_d = np.random.normal(0.0, 1.0)
                points[i, d] = v_d
                norm_sq += v_d * v_d
            
            norm_v = math.sqrt(norm_sq)
            if norm_v == 0.0:
                points[i, 0] = 1.0
                norm_v = 1.0
            
            r = (np.random.uniform(0.0, 1.0)**(1.0/k)) * 2.0
            scale = r / norm_v
            for d in range(k):
                points[i, d] *= scale
                
        # 2. Calculate the maximum distance between any pair of inner points
        max_inner_dist = 0.0
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                dist = calc_dist(points[i], points[j])
                if dist > max_inner_dist:
                    max_inner_dist = dist
                    
        min_separation = s * max_inner_dist
        
        # --- EARLY GEOMETRIC PRUNING ---
        # The points are bound to a ball of radius 2.0.
        # The max distance from inner point p_i to anywhere in the ball is 2.0 + |p_i|.
        # If the required min_separation is strictly greater than this for ANY inner point, 
        # it is mathematically impossible to place outer points.
        impossible = False
        for i in range(n_inner):
            norm_sq = 0.0
            for d in range(k):
                norm_sq += points[i, d] * points[i, d]
            max_dist_to_boundary = 2.0 + math.sqrt(norm_sq)
            
            if min_separation >= max_dist_to_boundary:
                impossible = True
                break
                
        if impossible:
            continue # Instantly try a new inner configuration!
            
        # 3. Generate outer points
        count = n_inner
        attempts = 0
        # Radically reduced. Rolling a new inner cloud is much faster 
        # than brute-forcing a tiny valid volume.
        max_attempts = 200 
        
        temp_p = np.zeros(k, dtype=np.float32)
        while count < n_total and attempts < max_attempts:
            attempts += 1
            
            norm_sq = 0.0
            for d in range(k):
                v_d = np.random.normal(0.0, 1.0)
                temp_p[d] = v_d
                norm_sq += v_d * v_d
                
            norm_v = math.sqrt(norm_sq)
            if norm_v == 0.0:
                temp_p[0] = 1.0
                norm_v = 1.0
                
            r = (np.random.uniform(0.0, 1.0)**(1.0/k)) * 2.0
            scale = r / norm_v
            for d in range(k):
                temp_p[d] *= scale
                
            valid = True
            for j in range(n_inner):
                if calc_dist(temp_p, points[j]) < min_separation:
                    valid = False
                    break
                    
            if valid:
                for d in range(k):
                    points[count, d] = temp_p[d]
                count += 1
                
        if count == n_total:
            return points, n_inner

def get_permutations(n_total):
    """Precomputes all Hamiltonian path permutations."""
    start = 0
    end = n_total - 1
    middle_nodes = [i for i in range(n_total) if i != start and i != end]
    perms = list(itertools.permutations(middle_nodes))
    
    full_perms = np.zeros((len(perms), n_total), dtype=np.int32)
    for i, p in enumerate(perms):
        full_perms[i, 0] = start
        full_perms[i, 1:-1] = p
        full_perms[i, -1] = end
        
    return full_perms

@njit(fastmath=True)
def solve_optimal_hamiltonian_path(points, perms):
    n = len(points)
    
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
def count_inner_exits(path, n_inner):
    exits = 0
    for i in range(len(path) - 1):
        if path[i] < n_inner and path[i+1] >= n_inner:
            exits += 1
    return exits

@njit(parallel=True, fastmath=True)
def compute_exits_batch(batch_size, k, n_inner, n_outer, s, base_seed, batch_offset, perms):
    """Parallel loop that ONLY computes the integers. Massive memory savings."""
    exits_out = np.zeros(batch_size, dtype=np.int32)
    
    for i in prange(batch_size):
        global_trial_idx = batch_offset + i
        np.random.seed(base_seed + global_trial_idx * 19937)
        
        points, inner_count = generate_points(k, n_inner, n_outer, s)
        path, cost = solve_optimal_hamiltonian_path(points, perms)
        exits_out[i] = count_inner_exits(path, inner_count)
        
    return exits_out

@njit(fastmath=True)
def run_simulation_batch(batch_size, k, n_inner, n_outer, s, base_seed, batch_offset, perms, max_allowed_exits):
    """Coordinates the batch, returning only summary stats and the worst trial's data."""
    # 1. Run the highly parallel, low-memory loop
    exits_out = compute_exits_batch(batch_size, k, n_inner, n_outer, s, base_seed, batch_offset, perms)
    
    # 2. Find the worst offender
    batch_max_exits = -1
    best_local_idx = 0
    violations = 0
    
    for i in range(batch_size):
        e = exits_out[i]
        if e > max_allowed_exits:
            violations += 1
        if e > batch_max_exits:
            batch_max_exits = e
            best_local_idx = i
            
    # 3. Deterministically re-run the worst trial to get its specific points and path
    worst_global_idx = batch_offset + best_local_idx
    np.random.seed(base_seed + worst_global_idx * 19937)
    worst_points, inner_count = generate_points(k, n_inner, n_outer, s)
    worst_path, worst_cost = solve_optimal_hamiltonian_path(worst_points, perms)
    
    return batch_max_exits, violations, worst_points, worst_path, worst_global_idx + 1, inner_count


def run_simulation(trials, k, seed, max_exits, s, n_inner, n_outer, batch_size=5_000):
    print(f"Running {trials} trials in k={k} dimensions (Seed: {seed}, s={s})...")
    print(f"Testing if any path exits the inner cloud more than {max_exits} times.\n")
    print("Compiling Numba functions (the progress bar will pause briefly at 0%)...\n")
    
    max_exits_seen = 0
    total_violations = 0
    
    # Precompute all possible routing paths
    perms = get_permutations(n_inner + n_outer)
    num_batches = (trials + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Simulating batches", unit="batch", dynamic_ncols=True):
        current_batch_size = min(batch_size, trials - batch_idx * batch_size)
        batch_offset = batch_idx * batch_size
        
        b_max, b_violations, b_points, b_path, b_trial_num, n_in = run_simulation_batch(
            current_batch_size, k, n_inner, n_outer, s, seed, batch_offset, perms, max_exits
        )
        
        if b_max > max_exits_seen:
            max_exits_seen = b_max
            
        if b_violations > 0:
            total_violations += b_violations
            
            report = [
                f"\nBatch Violations Found! Logging worst trial in batch:",
                f"Trial {b_trial_num}: Exited {b_max} times.",
                f"Path sequence (by index): {list(b_path)}",
                "Coordinates in path order:"
            ]
            for step_num, p_idx in enumerate(b_path):
                location = "Inner" if p_idx < n_in else "Outer"
                coords = ", ".join([f"{c:.4f}" for c in b_points[p_idx]])
                report.append(f"  Step {step_num + 1}: Index {p_idx} [{location}] -> ({coords})")
            report.append("-" * 40)
            
            tqdm.write("\n".join(report))
            
    print("\n" + "-" * 40)
    print("Simulation Complete.")
    print(f"Maximum exits observed: {max_exits_seen}")
    if total_violations == 0:
        print(f"Result: Theorem holds! No paths exited the inner cloud more than {max_exits} times.")
    else:
        print(f"Result: {total_violations} paths violated the condition.")

if __name__ == "__main__":
    run_simulation(trials=1_000_000, k=2, seed=4, max_exits=3, s=1.5, n_inner=4, n_outer=4)
