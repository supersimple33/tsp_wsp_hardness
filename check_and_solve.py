import itertools

def parse_matrix(matrix_str):
    """Parses a comma-separated string into a 2D float list."""
    matrix = []
    for line in matrix_str.strip().split('\n'):
        matrix.append([float(x) for x in line.split(',')])
    return matrix

def check_triangle_inequality(matrix):
    """Verifies if the distance matrix satisfies the triangle inequality."""
    n = len(matrix)
    is_metric = True
    violations = []
    
    # Check d(i, j) <= d(i, k) + d(k, j) for all i, j, k
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i != j and j != k and i != k:
                    # Added a tiny epsilon (1e-6) to account for float precision issues
                    if matrix[i][j] > matrix[i][k] + matrix[k][j] + 1e-6:
                        is_metric = False
                        violations.append((i, j, k))
                        
    return is_metric, violations

def solve_tsp_path(matrix):
    """
    Finds the shortest path visiting all nodes exactly once, 
    starting at the first node and ending at the last node.
    """
    n = len(matrix)
    start_node = 0
    end_node = n - 1
    
    # We only permute the intermediate nodes
    middle_nodes = list(range(1, n - 1))
    
    best_path = None
    min_dist = float('inf')
    
    # Test all permutations of the intermediate nodes
    for perm in itertools.permutations(middle_nodes):
        path = [start_node] + list(perm) + [end_node]
        
        # Calculate the distance of the current path
        dist = 0
        for i in range(len(path) - 1):
            dist += matrix[path[i]][path[i+1]]
            
        if dist < min_dist:
            min_dist = dist
            best_path = path
            
    return best_path, min_dist

if __name__ == "__main__":
    matrix_str = """
0.00,82.57,52.00,153.00,206.25,270.51,214.90,135.96,209.00
82.57,0.00,83.82,151.79,163.20,192.09,135.00,55.01,132.20
52.00,83.82,0.00,101.00,160.61,246.79,208.01,137.10,215.37
153.00,151.79,101.00,0.00,94.34,230.18,231.26,187.42,259.82
206.25,163.20,160.61,94.34,0.00,148.37,179.81,170.60,223.28
270.51,192.09,246.79,230.18,148.37,0.00,92.42,153.84,145.38
214.90,135.00,208.01,231.26,179.81,92.42,0.00,80.23,54.01
135.96,55.01,137.10,187.42,170.60,153.84,80.23,0.00,78.45
209.00,132.20,215.37,259.82,223.28,145.38,54.01,78.45,0.00
"""

    dist_matrix = parse_matrix(matrix_str)
    
    # 1. Verify Triangle Inequality
    is_metric, violations = check_triangle_inequality(dist_matrix)
    print("--- Triangle Inequality Check ---")
    print(f"Satisfies Triangle Inequality: {is_metric}")
    if not is_metric:
        print(f"Found {len(violations)} violations.")
        v = violations
        print(f"Example violation: d({v}, {v}) > d({v}, {v}) + d({v}, {v})")

    # 2. Solve Shortest Path (First to Last)
    best_path, min_dist = solve_tsp_path(dist_matrix)
    
    print("\n--- Shortest Path Results ---")
    print(f"Shortest Path (Node Indices): {best_path}")
    print(f"Minimum Distance: {min_dist:.2f}")