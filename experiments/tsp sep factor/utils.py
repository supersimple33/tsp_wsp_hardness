import numpy as np


def is_metric(dist_matrix: np.ndarray, tol=1e-9) -> bool:
    """Check if a distance matrix is metric (satisfies triangle inequality)."""
    if any(
        dist_matrix.shape[0] != dist_matrix.shape[i]
        for i in range(1, len(dist_matrix.shape))
    ):
        raise ValueError("Distance matrix must be square")
    # check non-negativity
    if not np.all(dist_matrix >= 0):
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
