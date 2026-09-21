"""Vector validation and distance metrics (Euclidean, Manhattan, Cosine)."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.config import validate_metric

FEATURE_DIMENSION = 9


def validate_feature_vector(
    vector: ArrayLike, expected_dim: int | None = None
) -> NDArray[np.float64]:
    """Convert a feature vector to a 1D finite float64 array.

    Args:
        vector: Input array-like vector.
        expected_dim: Expected feature vector length. If None, any non-empty 1D shape is accepted.

    Raises:
        ValueError: If vector is invalid, non-1D, empty, contains NaN/Inf, or has incorrect dimension.
    """
    try:
        validated = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature vector must contain numeric values.") from exc

    if validated.ndim != 1 or validated.size == 0:
        raise ValueError(
            f"Feature vector must be a non-empty 1D array; received shape {validated.shape}."
        )

    if expected_dim is not None and validated.shape != (expected_dim,):
        raise ValueError(
            f"Feature vector shape mismatch: expected ({expected_dim},), received {validated.shape}."
        )

    if not np.all(np.isfinite(validated)):
        raise ValueError("Feature vector contains NaN or infinite values.")

    return validated


def calculate_distance(
    vector_a: ArrayLike,
    vector_b: ArrayLike,
    metric: str = "euclidean",
) -> float:
    """Compute the distance between two 1D feature vectors.

    Supported metrics: "euclidean", "manhattan", "cosine".
    """
    metric_clean = validate_metric(metric)
    a = validate_feature_vector(vector_a)
    b = validate_feature_vector(vector_b, expected_dim=a.shape[0])

    if metric_clean == "euclidean":
        return float(np.sqrt(np.sum((a - b) ** 2)))

    if metric_clean == "manhattan":
        return float(np.sum(np.abs(a - b)))

    if metric_clean == "cosine":
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0
        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        similarity_clipped = max(-1.0, min(1.0, similarity))
        return float(1.0 - similarity_clipped)

    raise ValueError(f"Unsupported metric: {metric}")


def calculate_distances(
    query_vector: ArrayLike,
    database_matrix: ArrayLike,
    metric: str = "euclidean",
) -> NDArray[np.float64]:
    """Compute vectorized distances between a 1D query vector and an (N x D) matrix.

    Supported metrics: "euclidean", "manhattan", "cosine".
    """
    metric_clean = validate_metric(metric)
    query = validate_feature_vector(query_vector)

    try:
        matrix = np.asarray(database_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Database matrix must contain numeric values.") from exc

    if matrix.ndim != 2:
        raise ValueError(
            f"Database matrix must be a 2D array (N x D); received shape {matrix.shape}."
        )
    if matrix.shape[0] == 0:
        raise ValueError("Database matrix must contain at least one feature vector.")
    if matrix.shape[1] != query.shape[0]:
        raise ValueError(
            f"Feature dimension mismatch: query vector is {query.shape[0]}D, "
            f"database matrix is {matrix.shape[1]}D."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Database matrix contains NaN or infinite values.")

    if metric_clean == "euclidean":
        distances = np.sqrt(np.sum((matrix - query) ** 2, axis=1))

    elif metric_clean == "manhattan":
        distances = np.sum(np.abs(matrix - query), axis=1)

    elif metric_clean == "cosine":
        query_norm = float(np.linalg.norm(query))
        matrix_norms = np.linalg.norm(matrix, axis=1)
        dots = np.dot(matrix, query)

        denominator = matrix_norms * query_norm
        safe_denominator = np.where(denominator == 0.0, 1.0, denominator)

        similarities = np.where(denominator == 0.0, 0.0, dots / safe_denominator)
        similarities_clipped = np.clip(similarities, -1.0, 1.0)
        distances = 1.0 - similarities_clipped

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if distances.shape != (matrix.shape[0],) or not np.all(np.isfinite(distances)):
        raise ValueError("Distance computation produced invalid output.")

    return distances.astype(np.float64, copy=False)


def euclidean_distance(vector_a: ArrayLike, vector_b: ArrayLike) -> float:
    """Euclidean distance between two 9D vectors (maintained for baseline compatibility)."""
    return calculate_distance(vector_a, vector_b, metric="euclidean")


def euclidean_distances(
    query_vector: ArrayLike, database_matrix: ArrayLike
) -> NDArray[np.float64]:
    """Vectorized Euclidean distances for N x 9 matrix (maintained for baseline compatibility)."""
    return calculate_distances(query_vector, database_matrix, metric="euclidean")
