"""Vector validation and distance functions for Color Moments descriptors.

A single color space gives a 9D vector; concatenated color spaces give 18D or
27D vectors. ``dimension`` defaults to 9 to keep the baseline API unchanged.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FEATURE_DIMENSION = 9
DISTANCE_METRICS = ("euclidean", "manhattan", "cosine")


def validate_feature_vector(
    vector: ArrayLike,
    dimension: int | None = FEATURE_DIMENSION,
) -> NDArray[np.float64]:
    """Convert a feature vector to finite float64 1D array.

    ``dimension=None`` accepts any non-empty 1D vector.
    """
    try:
        validated = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature vector must contain numeric values.") from exc

    if dimension is None:
        if validated.ndim != 1 or validated.size == 0:
            raise ValueError(
                f"Feature vector must be a non-empty 1D array, received {validated.shape}."
            )
    elif validated.shape != (dimension,):
        raise ValueError(
            f"Feature vector must have shape ({dimension},), received {validated.shape}."
        )
    if not np.all(np.isfinite(validated)):
        raise ValueError("Feature vector contains NaN or infinite values.")
    return validated


def validate_feature_matrix(
    database_matrix: ArrayLike,
    dimension: int,
) -> NDArray[np.float64]:
    """Convert a database matrix to finite float64 with shape ``(N, dimension)``."""
    try:
        matrix = np.asarray(database_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Database matrix must contain numeric values.") from exc

    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError(
            f"Database matrix must have shape (N, {dimension}); "
            f"received {matrix.shape}."
        )
    if matrix.shape[0] == 0:
        raise ValueError("Database matrix must contain at least one feature vector.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Database matrix contains NaN or infinite values.")
    return matrix


def euclidean_distance(vector_a: ArrayLike, vector_b: ArrayLike) -> float:
    """Return ``sqrt(sum((a - b)^2))`` for two validated 9D vectors."""
    a = validate_feature_vector(vector_a)
    b = validate_feature_vector(vector_b)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _euclidean(query: NDArray[np.float64], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt(np.sum((matrix - query) ** 2, axis=1))


def _manhattan(query: NDArray[np.float64], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sum(np.abs(matrix - query), axis=1)


def _cosine(query: NDArray[np.float64], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cosine distance ``1 - cos(a, b)``; a zero vector has distance 1."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    dots = matrix @ query
    similarity = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


_DISTANCE_FUNCTIONS: dict[str, Callable[..., NDArray[np.float64]]] = {
    "euclidean": _euclidean,
    "manhattan": _manhattan,
    "cosine": _cosine,
}


def compute_distances(
    query_vector: ArrayLike,
    database_matrix: ArrayLike,
    metric: str = "euclidean",
) -> NDArray[np.float64]:
    """Return distances from one query to every row of an N x D matrix.

    ``metric`` is one of ``euclidean``, ``manhattan`` or ``cosine``. Smaller is
    always more similar.
    """
    metric_name = str(metric).strip().lower()
    if metric_name not in _DISTANCE_FUNCTIONS:
        raise ValueError(
            f"Unsupported metric {metric!r}; expected one of {', '.join(DISTANCE_METRICS)}."
        )
    query = validate_feature_vector(query_vector, dimension=None)
    matrix = validate_feature_matrix(database_matrix, query.shape[0])

    distances = _DISTANCE_FUNCTIONS[metric_name](query, matrix)
    if distances.shape != (matrix.shape[0],) or not np.all(np.isfinite(distances)):
        raise ValueError(f"{metric_name.capitalize()} distance output is invalid.")
    return distances.astype(np.float64, copy=False)


def euclidean_distances(
    query_vector: ArrayLike,
    database_matrix: ArrayLike,
) -> NDArray[np.float64]:
    """Return vectorized Euclidean distances from one query to an N x 9 matrix."""
    validate_feature_vector(query_vector)
    return compute_distances(query_vector, database_matrix, "euclidean")
