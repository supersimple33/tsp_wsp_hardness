import numpy as np

def generate_points(k, n_inner=5, n_outer=5, s=1.0):
    """
    Generates an inner cloud of points and well-separated outer points.
    k: Number of dimensions.
    s: Minimum separation distance.
    """
    # 1. Generate inner cloud (A1)
    # To ensure max pairwise distance is <= 1.0, we generate points within a 
    # hypersphere of radius 0.5 centered at the origin.
    inner = []
    for _ in range(n_inner):
        # Random uniform point in k-dimensional ball of radius 0.5
        v = np.random.normal(0, 1, k)
        norm_v = np.linalg.norm(v)
        if norm_v == 0:  # Edge case protection
            v = np.zeros(k)
            v[0] = 1.0
            norm_v = 1.0
        v /= norm_v # Project to unit surface
        r = np.random.uniform(0, 1)**(1.0/k) * 0.5 # Scale by radius
        inner.append(v * r)
    
    points = list(inner)
    
    # 2. Generate outer points (A2 ... An)
    # We use rejection sampling to guarantee each outer point is at least 's' 
    # away from EVERY other point (both inner and other outer points)
    outer = []
    R_bound = max(5.0, s * n_outer) # Search space radius
    
    while len(outer) < n_outer:
        p = np.random.uniform(-R_bound, R_bound, k)
        
        # Check separation constraint
        valid = True
        for existing_p in points:
            if np.linalg.norm(p - existing_p) < s:
                valid = False
                break
                
        if valid:
            outer.append(p)
            points.append(p)
            
    return np.array(points), n_inner

def solve_optimal_hamiltonian_path(points, start_idx, end_idx):
    """
    Finds the exact shortest Hamiltonian path from start_idx to end_idx
    using the Held-Karp dynamic programming algorithm.
    """
    n = len(points)
    # Precompute distance matrix
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    
    # dp[(mask, u)] = (min_cost, parent_node)
    # mask represents the visited nodes (using bitwise flags)
    dp = { (1 << start_idx, start_idx): (0.0, None) }
    
    # Iterate over all possible subsets of visited nodes
    for mask in range(1, 1 << n):
        # We only care about paths that started at our fixed start_idx
        if not (mask & (1 << start_idx)):
            continue
            
        for u in range(n):
            if not (mask & (1 << u)):
                continue
                
            state = (mask, u)
            if state not in dp:
                continue
                
            cost, _ = dp[state]
            
            # If we haven't visited everything but we are at the end_idx, 
            # this is an invalid intermediate path (must end strictly at end_idx)
            if u == end_idx and mask != (1 << n) - 1:
                continue
                
            # Try transitioning to an unvisited node v
            for v in range(n):
                if not (mask & (1 << v)):
                    # Prevent visiting end_idx prematurely
                    if v == end_idx and (mask | (1 << v)) != (1 << n) - 1:
                        continue
                        
                    new_mask = mask | (1 << v)
                    new_cost = cost + dist[u][v]
                    new_state = (new_mask, v)
                    
                    if new_state not in dp or new_cost < dp[new_state][0]:
                        dp[new_state] = (new_cost, u)
                        
    # Reconstruct the optimal path
    final_state = ((1 << n) - 1, end_idx)
    
    # Safety check in case no path is found (shouldn't happen with complete graphs)
    if final_state not in dp:
        return [], float('inf')
        
    path = []
    curr = final_state
    
    while curr[1] is not None:
        u = curr[1]
        path.append(u)
        parent = dp[curr][1]
        if parent is None:
            break
        prev_mask = curr[0] ^ (1 << u)
        curr = (prev_mask, parent)
        
    return path[::-1], dp[final_state][0]

def count_inner_exits(path, n_inner):
    """Counts how many times the path transitions from the inner cloud to the outside"""
    exits = 0
    for i in range(len(path) - 1):
        if path[i] < n_inner and path[i+1] >= n_inner:
            exits += 1
    return exits

def run_simulation(trials=100, k=2, seed=42):
    # Set consistent seed for reproducibility
    np.random.seed(seed)
    
    print(f"Running {trials} trials in k={k} dimensions (Seed: {seed})...")
    print("Testing if any path exits the inner cloud more than twice.\n")
    
    max_exits_seen = 0
    violations = 0
    
    for i in range(trials):
        # Generate configurations (5 inner, 5 outer minimum)
        points, n_inner = generate_points(k=k)
        
        start_idx = 0                  # First point in inner cloud
        end_idx = len(points) - 1      # Last point in outer cloud
        
        # Solve exactly
        path, cost = solve_optimal_hamiltonian_path(points, start_idx, end_idx)
        
        # Analyze path transitions
        exits = count_inner_exits(path, n_inner)
        max_exits_seen = max(max_exits_seen, exits)
        
        # Report violations with exact coordinates
        if exits > 3:
            violations += 1
            print(f"Trial {i+1}: VIOLATION! Exited {exits} times.")
            print(f"Path sequence (by index): {path}")
            print("Coordinates in path order:")
            for step_num, p_idx in enumerate(path):
                location = "Inner" if p_idx < n_inner else "Outer"
                # Formatting the coordinates nicely to 4 decimal places
                coords = ", ".join([f"{c:.4f}" for c in points[p_idx]])
                print(f"  Step {step_num + 1}: Index {p_idx} [{location}] -> ({coords})")
            print("-" * 40)
            
    print("-" * 40)
    print("Simulation Complete.")
    print(f"Maximum exits observed in a single optimal path: {max_exits_seen}")
    if violations == 0:
        print("Result: Theorem holds! No paths exited the inner cloud more than twice.")
    else:
        print(f"Result: {violations} paths violated the condition.")

if __name__ == "__main__":
    run_simulation(trials=100_000, k=3, seed=44)
