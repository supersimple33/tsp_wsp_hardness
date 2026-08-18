import numpy as np
from numba import njit, prange
from tqdm import tqdm
import math

@njit
def set_seed(seed):
    """Numba maintains its own random state, so we seed it inside a jitted function."""
    np.random.seed(seed)

@njit(fastmath=True, inline='always')
def calc_dist(p1, p2):
    """A fast, allocation-free Euclidean distance calculator."""
    d = 0.0
    for i in range(len(p1)):
        diff = p1[i] - p2[i]
        d += diff * diff
    return math.sqrt(d)

@njit(fastmath=True)
def generate_points(k, n_inner=4, n_outer=4, s=1.0):
    n_total = n_inner + n_outer
    points = np.zeros((n_total, k), dtype=np.float32)
    
    while True:
        # 1. Generate inner cloud without creating temporary arrays
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
        
        # 3. Generate outer points
        count = n_inner
        attempts = 0
        max_attempts = 5000 
        
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

@njit(fastmath=True)
def solve_optimal_hamiltonian_path(points, start_idx, end_idx):
    n = len(points)
    
    # Precompute distance matrix avoiding array slicing
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = calc_dist(points[i], points[j])
    
    # Explicit float64 initialization prevents Numba dtype inference issues with np.inf
    dp = np.full((1 << n, n), np.inf, dtype=np.float64)
    parent = np.full((1 << n, n), -1, dtype=np.int32)
    
    dp[1 << start_idx, start_idx] = 0.0
    
    for mask in range(1, 1 << n):
        if not (mask & (1 << start_idx)):
            continue
            
        for u in range(n):
            if not (mask & (1 << u)):
                continue
                
            cost = dp[mask, u]
            if cost == np.inf:
                continue
                
            if u == end_idx and mask != (1 << n) - 1:
                continue
                
            for v in range(n):
                if not (mask & (1 << v)):
                    if v == end_idx and (mask | (1 << v)) != (1 << n) - 1:
                        continue
                        
                    new_mask = mask | (1 << v)
                    new_cost = cost + dist[u, v]
                    
                    if new_cost < dp[new_mask, v]:
                        dp[new_mask, v] = new_cost
                        parent[new_mask, v] = u
                        
    final_mask = (1 << n) - 1
    final_cost = dp[final_mask, end_idx]
    
    if final_cost == np.inf:
        return np.zeros(0, dtype=np.int32), np.inf
        
    path = np.zeros(n, dtype=np.int32)
    curr_mask = final_mask
    curr_u = end_idx
    step = n - 1
    
    while curr_u != -1:
        path[step] = curr_u
        p = parent[curr_mask, curr_u]
        curr_mask = curr_mask ^ (1 << curr_u)
        curr_u = p
        step -= 1
        
    return path, final_cost

@njit(fastmath=True)
def count_inner_exits(path, n_inner):
    exits = 0
    for i in range(len(path) - 1):
        if path[i] < n_inner and path[i+1] >= n_inner:
            exits += 1
    return exits

@njit(parallel=True, fastmath=True)
def run_simulation_batch(batch_size, k, n_inner, n_outer, s, base_seed, batch_offset):
    """Runs a batch of trials in parallel with deterministic per-trial seeding."""
    n_total = n_inner + n_outer
    
    points_out = np.zeros((batch_size, n_total, k), dtype=np.float32)
    paths_out = np.zeros((batch_size, n_total), dtype=np.int32)
    exits_out = np.zeros(batch_size, dtype=np.int32)
    
    for i in prange(batch_size):
        # 1. Calculate the absolute global index of this specific trial
        global_trial_idx = batch_offset + i
        
        # 2. Seed this specific thread for this specific iteration
        # Multiplying by a prime (e.g., 19937) helps ensure different trials 
        # don't overlap their Mersenne Twister sequences.
        np.random.seed(base_seed + global_trial_idx * 19937)
        
        # 3. Generate and solve
        points, inner_count = generate_points(k, n_inner, n_outer, s)
        path, cost = solve_optimal_hamiltonian_path(points, 0, n_total - 1)
        e = count_inner_exits(path, inner_count)
        
        points_out[i] = points
        paths_out[i] = path
        exits_out[i] = e
        
    return points_out, paths_out, exits_out, n_inner

def run_simulation(trials=1_000_000, k=2, seed=4, max_exits=3, batch_size=10_000):
    print(f"Running {trials} trials in k={k} dimensions (Seed: {seed})...")
    print("Testing if any path exits the inner cloud more than twice.\n")
    print("Compiling Numba functions (the progress bar will pause briefly at 0%)...\n")
    
    max_exits_seen = 0
    violations = 0
    
    n_inner = 4
    n_outer = 4
    s = 1.0
    
    num_batches = (trials + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Simulating batches", unit="batch", dynamic_ncols=True):
        current_batch_size = min(batch_size, trials - batch_idx * batch_size)
        batch_offset = batch_idx * batch_size
        
        # Pass the base_seed and the batch_offset so Numba knows exactly which trials it is running
        points, paths, exits, n_in = run_simulation_batch(
            current_batch_size, k, n_inner, n_outer, s, seed, batch_offset
        )
        
        batch_max = np.max(exits)
        if batch_max > max_exits_seen:
            max_exits_seen = batch_max
            
        if batch_max > max_exits:
            violation_indices = np.where(exits > max_exits)[0]
            for idx in violation_indices:
                violations += 1
                e = exits[idx]
                p = paths[idx]
                pts = points[idx]
                
                # The exact trial number is preserved flawlessly
                trial_num = batch_offset + idx + 1
                
                report = [
                    f"\nTrial {trial_num}: VIOLATION! Exited {e} times.",
                    f"Path sequence (by index): {list(p)}",
                    "Coordinates in path order:"
                ]
                for step_num, p_idx in enumerate(p):
                    location = "Inner" if p_idx < n_in else "Outer"
                    coords = ", ".join([f"{c:.4f}" for c in pts[p_idx]])
                    report.append(f"  Step {step_num + 1}: Index {p_idx} [{location}] -> ({coords})")
                report.append("-" * 40)
                
                tqdm.write("\n".join(report))
            
    print("\n" + "-" * 40)
    print("Simulation Complete.")
    print(f"Maximum exits observed: {max_exits_seen}")
    if violations == 0:
        print("Result: Theorem holds! No paths exited the inner cloud more than twice.")
    else:
        print(f"Result: {violations} paths violated the condition.")

if __name__ == "__main__":
    run_simulation(trials=1_000_000, k=2, seed=4, max_exits=3)
