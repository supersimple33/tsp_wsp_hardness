import numpy as np
from numba import njit
from tqdm import tqdm

@njit
def set_seed(seed):
    """Numba maintains its own random state, so we seed it inside a jitted function."""
    np.random.seed(seed)

@njit
def generate_points(k, n_inner=4, n_outer=4, s=1.0):
    """
    Generates an inner cloud of points and well-separated outer points.
    k: Number of dimensions.
    s: Separation multiplier relative to maximum inner distance.
    All points are generated within a ball of radius 2.
    """
    n_total = n_inner + n_outer
    points = np.zeros((n_total, k), dtype=np.float32)
    
    while True:
        # 1. Generate inner cloud (A1) within a ball of radius 2
        for i in range(n_inner):
            v = np.random.normal(0.0, 1.0, k)
            norm_v = np.linalg.norm(v)
            if norm_v == 0.0:  # Edge case protection
                v = np.zeros(k)
                v[0] = 1.0
                norm_v = 1.0
            v /= norm_v # Project to unit surface
            r = (np.random.uniform(0.0, 1.0)**(1.0/k)) * 2.0 # Scale by radius 2
            points[i] = v * r
            
        # 2. Calculate the maximum distance between any pair of inner points
        max_inner_dist = 0.0
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                d = np.linalg.norm(points[i] - points[j])
                if d > max_inner_dist:
                    max_inner_dist = d
                    
        # The required minimum separation distance to outer points
        min_separation = s * max_inner_dist
        
        # 3. Generate outer points within the same ball of radius 2
        count = n_inner
        attempts = 0
        max_attempts = 5000 # Prevent infinite loop if layout is too constrained
        
        while count < n_total and attempts < max_attempts:
            attempts += 1
            
            # Generate candidate outer point
            v = np.random.normal(0.0, 1.0, k)
            norm_v = np.linalg.norm(v)
            if norm_v == 0.0:
                v = np.zeros(k)
                v[0] = 1.0
                norm_v = 1.0
            v /= norm_v
            r = (np.random.uniform(0.0, 1.0)**(1.0/k)) * 2.0
            p = v * r
            
            # Check condition: must be at least `min_separation` away from ALL inner points
            valid = True
            for j in range(n_inner):
                if np.linalg.norm(p - points[j]) < min_separation:
                    valid = False
                    break
                    
            if valid:
                points[count] = p
                count += 1
                
        # If we successfully found enough outer points, break and return
        if count == n_total:
            return points, n_inner

@njit
def solve_optimal_hamiltonian_path(points, start_idx, end_idx):
    """
    Finds the exact shortest Hamiltonian path from start_idx to end_idx
    using the Held-Karp dynamic programming algorithm.
    """
    n = len(points)
    
    # Precompute distance matrix to avoid recalculating in the inner loops
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = np.linalg.norm(points[i] - points[j])
    
    # DP arrays
    dp = np.full((1 << n, n), np.inf)
    parent = np.full((1 << n, n), -1, dtype=np.int32)
    
    # Base case
    dp[1 << start_idx, start_idx] = 0.0
    
    # Iterate over all possible subsets of visited nodes
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
                        
    # Reconstruct the optimal path
    final_mask = (1 << n) - 1
    final_cost = dp[final_mask, end_idx]
    
    if final_cost == np.inf:
        return np.zeros(0, dtype=np.int32), np.inf
        
    path = np.zeros(n, dtype=np.int32)
    curr_mask = final_mask
    curr_u = end_idx
    step = n - 1
    
    # Backtrack through parent array
    while curr_u != -1:
        path[step] = curr_u
        p = parent[curr_mask, curr_u]
        curr_mask = curr_mask ^ (1 << curr_u)
        curr_u = p
        step -= 1
        
    return path, final_cost

@njit
def count_inner_exits(path, n_inner):
    """Counts how many times the path transitions from the inner cloud to the outside"""
    exits = 0
    for i in range(len(path) - 1):
        if path[i] < n_inner and path[i+1] >= n_inner:
            exits += 1
    return exits

def run_simulation(trials=100, k=2, seed=42, max_exits=2):
    # Set the Numba engine's random seed
    set_seed(seed)
    
    print(f"Running {trials} trials in k={k} dimensions (Seed: {seed})...")
    print("Testing if any path exits the inner cloud more than twice.\n")
    print("Compiling Numba functions (the progress bar will pause briefly at 0%)...\n")
    
    max_exits_seen = 0
    violations = 0
    
    # Use tqdm for the progress bar
    for i in tqdm(range(trials), desc="Simulating paths", unit="trial", dynamic_ncols=True):
        points, n_inner = generate_points(k=k)
        
        start_idx = 0
        end_idx = len(points) - 1
        
        path, cost = solve_optimal_hamiltonian_path(points, start_idx, end_idx)
        
        exits = count_inner_exits(path, n_inner)
        if exits > max_exits_seen:
            max_exits_seen = exits
        
        if exits > max_exits:
            violations += 1
            # Compile violation details into a single string
            report = [
                f"\nTrial {i+1}: VIOLATION! Exited {exits} times.",
                f"Path sequence (by index): {list(path)}",
                "Coordinates in path order:"
            ]
            for step_num, p_idx in enumerate(path):
                location = "Inner" if p_idx < n_inner else "Outer"
                coords = ", ".join([f"{c:.4f}" for c in points[p_idx]])
                report.append(f"  Step {step_num + 1}: Index {p_idx} [{location}] -> ({coords})")
            report.append("-" * 40)
            
            # Use tqdm.write instead of print to prevent breaking the progress bar visually
            tqdm.write("\n".join(report))
            
    print("\n" + "-" * 40)
    print("Simulation Complete.")
    print(f"Maximum exits observed in a single optimal path: {max_exits_seen}")
    if violations == 0:
        print("Result: Theorem holds! No paths exited the inner cloud more than twice.")
    else:
        print(f"Result: {violations} paths violated the condition.")

if __name__ == "__main__":
    run_simulation(trials=1_000_000, k=2, seed=4, max_exits=3)