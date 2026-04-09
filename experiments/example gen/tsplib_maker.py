import pandas as pd
import numpy as np
import tsplib95

def is_metric(dist_matrix: np.ndarray, tol=1e-9) -> bool:
    """Check if a distance matrix is metric (satisfies triangle inequality)."""
    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("Input must be a square matrix.")
    # check non-negativity
    if not np.all(dist_matrix >= 0):
        return False
    # check zero diagonal
    if not np.allclose(np.diag(dist_matrix), 0.0, atol=tol):
        return False
    # check symmetry
    if not np.allclose(dist_matrix, dist_matrix.T, atol=tol):
        return False
    # check triangle inequality
    inequality_holds = (
        dist_matrix[:, np.newaxis, :]
        <= dist_matrix[:, :, np.newaxis] + dist_matrix[np.newaxis, :, :] + tol
    )
    return np.all(inequality_holds)  # pyright: ignore[reportReturnType]

def csv_to_tsp():
    input_csv = "mcdonalds_sf_oakland_distances.csv"
    output_tsp = "mcdonalds_sf_oakland.tsp"
    
    # 1. Load the distance matrix from the CSV
    # index_col=0 ensures the store labels are read as the index, not a column of data
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv, index_col=0)
    
    # 2. Scale floats to integers to preserve precision for TSP solvers
    # Multiplies by 100, rounds to nearest whole number, and casts to int
    print("Scaling floats to integers (x100) to preserve precision...")
    matrix_int = (df * 100).round().astype(int)
    
    # Convert the Pandas DataFrame to a standard 2D Python list
    edge_weights = matrix_int.values.tolist()
    dist_matrix = np.array(edge_weights)
    assert is_metric(dist_matrix, tol=1e-4), "The distance matrix does not satisfy metric properties (triangle inequality)."
    
    # 3. Initialize a new TSPLIB problem
    print("Building TSPLIB data structure...")
    problem = tsplib95.models.StandardProblem()
    
    # 4. Set the required TSPLIB metadata
    problem.name = 'McDonalds_SF_Oakland'
    problem.comment = 'Symmetric distance matrix of SF/Oakland McDonalds (units in hundredths of a mile)'
    problem.type = 'TSP'
    problem.dimension = len(df)
    problem.edge_weight_type = 'EXPLICIT'
    problem.edge_weight_format = 'FULL_MATRIX'
    
    # 5. Attach the generated integer matrix
    problem.edge_weights = edge_weights
    
    # 6. Save the problem to a .tsp file
    problem.save(output_tsp)
    print(f"\nSuccess! Saved TSPLIB file to: {output_tsp}")

if __name__ == "__main__":
    csv_to_tsp()