import math
import itertools

def test_hamiltonian_proof(s, M):
    """
    Tests the provided proof configuration for a given s and M.
    """
    # 1. Define the coordinates based on the proof
    I = {
        'i1': (-0.5, 0),
        'i2': (0, 0),
        'i3': (0.5, 0)
    }
    
    O = {
        'o1': (-M, s),
        'o2': (0, s),
        'o3': (M, s)
    }
    
    # Combine into a single dictionary
    points = {**I, **O}
    point_names = list(points.keys())
    
    # 2. Helper functions
    def path_length(path):
        """Calculates the total Euclidean distance of a given path sequence."""
        dist = 0
        for k in range(len(path) - 1):
            p1 = points[path[k]]
            p2 = points[path[k+1]]
            dist += math.dist(p1, p2)
        return dist

    def is_alternating(path):
        """Checks if a path strictly alternates between 'i' and 'o' nodes."""
        for k in range(len(path) - 1):
            # Check the first character of the point name ('i' or 'o')
            if path[k][0] == path[k+1][0]:
                return False
        return True

    # 3. Generate all possible Hamiltonian paths (permutations of the 6 vertices)
    all_paths = list(itertools.permutations(point_names))
    
    min_alt_len = float('inf')
    best_alt_path = None
    
    min_non_alt_len = float('inf')
    best_non_alt_path = None
    
    # 4. Evaluate all paths
    for path in all_paths:
        length = path_length(path)
        if is_alternating(path):
            if length < min_alt_len:
                min_alt_len = length
                best_alt_path = path
        else:
            if length < min_non_alt_len:
                min_non_alt_len = length
                best_non_alt_path = path

    # 5. Print out the results
    print(f"--- Testing Configuration: s = {s}, M = {M} ---")
    
    print("\n[Optimal ALTERNATING Path]")
    print(" -> ".join(best_alt_path))
    print(f"Length: {min_alt_len:.4f}")
    
    print("\n[Optimal NON-ALTERNATING Path]")
    print(" -> ".join(best_non_alt_path))
    print(f"Length: {min_non_alt_len:.4f}")
    
    print("\n[Conclusion]")
    if min_alt_len < min_non_alt_len:
        diff = min_non_alt_len - min_alt_len
        print(f"✅ PROOF HOLDS: The alternating path is shorter by {diff:.4f} units.")
    else:
        print("❌ PROOF FAILS for these parameters.")
    print("=" * 55)

if __name__ == "__main__":
    # Test 1: M is slightly larger than s
    test_hamiltonian_proof(s=2.0, M=100.0)
    
    # Test 2: M is massively larger than s (as M -> infinity)
    test_hamiltonian_proof(s=2.0, M=1000.0)
    
    # Test 3: Large s value
    test_hamiltonian_proof(s=10.0, M=5000.0)