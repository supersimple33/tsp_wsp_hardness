import itertools
import math

# The 8 points mapped by their index
points = {
    0: (-0.1951, 0.5742),
    1: (0.4465, 0.9571),
    2: (-0.4828, -0.0679),
    3: (0.5953, -0.4092),
    4: (1.7428, 0.3848),
    5: (-1.0539, 1.6879),
    6: (0.4866, -1.8278),
    7: (-1.8723, -0.5382)
}

def calculate_euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def solve_hamiltonian_path(points_dict, start_node, end_node):
    # Extract the nodes that will be permuted in the middle
    intermediate_nodes = [node for node in points_dict.keys() if node not in (start_node, end_node)]
    
    shortest_distance = float('inf')
    best_path = None
    
    # Generate all 720 permutations for the intermediate 6 nodes
    for perm in itertools.permutations(intermediate_nodes):
        # Construct the full path
        current_path = [start_node] + list(perm) + [end_node]
        current_distance = 0.0
        
        # Calculate the total distance for this specific sequence
        for i in range(len(current_path) - 1):
            u = current_path[i]
            v = current_path[i+1]
            current_distance += calculate_euclidean_distance(points_dict[u], points_dict[v])
            
            # Pruning optimization: if we exceed the shortest distance early, abandon this path
            if current_distance >= shortest_distance:
                break
        else:
            # If the loop finishes without breaking, we found a new shortest path
            shortest_distance = current_distance
            best_path = current_path
            
    return best_path, shortest_distance

if __name__ == "__main__":
    path, min_dist = solve_hamiltonian_path(points, 0, 7)
    
    print("Optimal Hamiltonian Path:")
    print(" -> ".join(map(str, path)))
    print(f"\nTotal Minimum Distance: {min_dist:.4f} units")