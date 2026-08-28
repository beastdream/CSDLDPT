"""Vector validation and Euclidean distance for 9D Color Moments."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


FEATURE_DIMENSION = 9


def validate_feature_vector(vector: ArrayLike) -> NDArray[np.float64]:
    """Convert a feature vector to finite float64 with exact shape ``(9,)``."""
    try:
        validated = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature vector must contain numeric values.") from exc

    if validated.shape != (FEATURE_DIMENSION,):
        raise ValueError(
            f"Feature vector must have shape (9,), received {validated.shape}."
        )
    if not np.all(np.isfinite(validated)):
        raise ValueError("Feature vector contains NaN or infinite values.")
    return validated


def euclidean_distance(vector_a: ArrayLike, vector_b: ArrayLike) -> float:
    """Return ``sqrt(sum((a - b)^2))`` for two validated 9D vectors."""
    a = validate_feature_vector(vector_a)
    b = validate_feature_vector(vector_b)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def euclidean_distances(
    query_vector: ArrayLike,
    database_matrix: ArrayLike,
) -> NDArray[np.float64]:
    """Return vectorized Euclidean distances from one query to an N x 9 matrix."""
    query = validate_feature_vector(query_vector)
    try:
        matrix = np.asarray(database_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Database matrix must contain numeric values.") from exc

    if matrix.ndim != 2 or matrix.shape[1:] != (FEATURE_DIMENSION,):
        raise ValueError(
            "Database matrix must have shape (N, 9); "
            f"received {matrix.shape}."
        )
    if matrix.shape[0] == 0:
        raise ValueError("Database matrix must contain at least one feature vector.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Database matrix contains NaN or infinite values.")

    distances = np.sqrt(np.sum((matrix - query) ** 2, axis=1))
    if distances.shape != (matrix.shape[0],) or not np.all(np.isfinite(distances)):
        raise ValueError("Euclidean distance output is invalid.")
    return distances.astype(np.float64, copy=False)
