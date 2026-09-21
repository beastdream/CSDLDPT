"""Corpus-wide Z-Score normalization for feature vectors."""

import json
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

ImagePath: type = str | PathLike[str]


def fit_normalizer(features: ArrayLike) -> dict[str, NDArray[np.float64]]:
    """Compute corpus-wide mean and standard deviation from a 2D feature matrix (N x D).

    Handles std = 0 safely to prevent division-by-zero by replacing 0.0 std with 1.0.

    Raises:
        ValueError: If ``features`` matrix is invalid, non-2D, empty, or contains NaN/Inf.
    """
    try:
        matrix = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature matrix must contain numeric values.") from exc

    if matrix.ndim != 2:
        raise ValueError(
            f"Feature matrix must be a 2D array (N x D); received shape {matrix.shape}."
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Feature matrix must be non-empty.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Feature matrix contains NaN or infinite values.")

    mean = np.mean(matrix, axis=0, dtype=np.float64)
    std = np.std(matrix, axis=0, dtype=np.float64)

    # Prevent division-by-zero for zero-variance features
    safe_std = np.where(std == 0.0, 1.0, std)

    return {"mean": mean, "std": safe_std}


def transform(
    features: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
) -> NDArray[np.float64]:
    """Apply Z-score normalization ``(x - mean) / std`` to a 1D vector or 2D matrix.

    Raises:
        ValueError: If inputs have mismatched dimensions or contain NaN/Inf.
    """
    try:
        data = np.asarray(features, dtype=np.float64)
        mean_arr = np.asarray(mean, dtype=np.float64)
        std_arr = np.asarray(std, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Inputs must contain numeric values.") from exc

    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(mean_arr)) or not np.all(np.isfinite(std_arr)):
        raise ValueError("Inputs contain NaN or infinite values.")

    if mean_arr.ndim != 1 or std_arr.ndim != 1 or mean_arr.shape != std_arr.shape:
        raise ValueError("Mean and std must be 1D arrays of equal length.")

    feature_dim = mean_arr.shape[0]
    if data.ndim == 1:
        if data.shape[0] != feature_dim:
            raise ValueError(
                f"Feature vector length ({data.shape[0]}) does not match normalizer dimension ({feature_dim})."
            )
    elif data.ndim == 2:
        if data.shape[1] != feature_dim:
            raise ValueError(
                f"Feature matrix columns ({data.shape[1]}) do not match normalizer dimension ({feature_dim})."
            )
    else:
        raise ValueError(
            f"Features must be a 1D vector or 2D matrix; received shape {data.shape}."
        )

    safe_std = np.where(std_arr == 0.0, 1.0, std_arr)
    normalized = (data - mean_arr) / safe_std

    if not np.all(np.isfinite(normalized)):
        raise ValueError("Z-score normalization produced NaN or infinite values.")

    return normalized.astype(np.float64, copy=False)


def save_normalizer(
    params: dict[str, ArrayLike],
    filepath: ImagePath,
) -> Path:
    """Save mean and std parameters to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if "mean" not in params or "std" not in params:
        raise ValueError("Normalizer parameters must contain 'mean' and 'std' keys.")

    mean_list = np.asarray(params["mean"], dtype=np.float64).tolist()
    std_list = np.asarray(params["std"], dtype=np.float64).tolist()

    data = {
        "normalization": "zscore",
        "dimension": len(mean_list),
        "mean": mean_list,
        "std": std_list,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return path


def load_normalizer(filepath: ImagePath) -> dict[str, NDArray[np.float64]]:
    """Load mean and std parameters from a JSON normalizer file."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Normalizer parameter file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "mean" not in data or "std" not in data:
        raise ValueError(f"Invalid normalizer parameter file schema in {path}.")

    mean = np.asarray(data["mean"], dtype=np.float64)
    std = np.asarray(data["std"], dtype=np.float64)

    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
        raise ValueError(f"Corrupted normalizer vectors in {path}.")

    safe_std = np.where(std == 0.0, 1.0, std)
    return {"mean": mean, "std": safe_std}


class ZScoreNormalizer:
    """Object wrapper for Z-score normalization."""

    def __init__(self) -> None:
        self.mean: NDArray[np.float64] | None = None
        self.std: NDArray[np.float64] | None = None

    def fit(self, features: ArrayLike) -> "ZScoreNormalizer":
        """Compute mean and std from corpus features."""
        params = fit_normalizer(features)
        self.mean = params["mean"]
        self.std = params["std"]
        return self

    def transform(self, features: ArrayLike) -> NDArray[np.float64]:
        """Transform features using fitted mean and std."""
        if self.mean is None or self.std is None:
            raise ValueError("ZScoreNormalizer must be fitted before transform.")
        return transform(features, self.mean, self.std)

    def save(self, filepath: ImagePath) -> Path:
        """Save fitted normalizer parameters to disk."""
        if self.mean is None or self.std is None:
            raise ValueError("ZScoreNormalizer must be fitted before saving.")
        return save_normalizer({"mean": self.mean, "std": self.std}, filepath)

    def load(self, filepath: ImagePath) -> "ZScoreNormalizer":
        """Load normalizer parameters from disk."""
        params = load_normalizer(filepath)
        self.mean = params["mean"]
        self.std = params["std"]
        return self
