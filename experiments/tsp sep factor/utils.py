import numpy as np


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
