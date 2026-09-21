"""Unified Content-Based Image Retrieval engine supporting multi-color descriptors, distance metrics, Z-score normalization, and internal/external query modes."""

from functools import lru_cache
import os
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.color_moments import extract_descriptor
from src.config import (
    DEFAULT_DESCRIPTOR,
    DEFAULT_METRIC,
    DEFAULT_NORMALIZATION,
    get_descriptor_dimension,
    validate_descriptor,
    validate_metric,
    validate_normalization,
)
from src.normalization import fit_normalizer, transform
from src.repository import get_all_rgb_global_features
from src.similarity import calculate_distances, validate_feature_vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DESCRIPTOR_COUNT = 1000


def _resolve_project_path(path_value: str | PathLike[str]) -> Path:
    """Resolve an absolute path or a path relative to the project root."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _canonical_path(path_value: str | PathLike[str]) -> str:
    """Create a Windows-safe canonical path key for path equality checks."""
    return os.path.normcase(str(_resolve_project_path(path_value)))


def _display_path(path: Path) -> str:
    """Prefer a project-relative, forward-slash path string for public output."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


@lru_cache(maxsize=16)
def _get_database_descriptor_matrix(
    descriptor: str,
) -> tuple[list[dict[str, Any]], NDArray[np.float64]]:
    """Fetch database records and extract/cache feature matrix for a specific descriptor.

    Returns:
        Tuple of (records_list, database_matrix_N_x_D).
    """
    records = get_all_rgb_global_features()
    if not records:
        raise ValueError("Database returned no feature descriptors.")

    if len(records) != EXPECTED_DESCRIPTOR_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DESCRIPTOR_COUNT} database descriptors, received {len(records)}."
        )

    desc_clean = validate_descriptor(descriptor)
    expected_dim = get_descriptor_dimension(desc_clean)

    if desc_clean == "rgb":
        # Fast-path using existing 9D RGB features from MySQL repository
        matrix = np.vstack([record["features"] for record in records])
    else:
        # Extract fused descriptor for each dataset image
        features_list: list[NDArray[np.float64]] = []
        for record in records:
            img_path = _resolve_project_path(record["filepath"])
            if not img_path.is_file():
                raise FileNotFoundError(f"Database image file missing: {img_path}")
            feat = extract_descriptor(img_path, descriptor=desc_clean)
            features_list.append(feat)
        matrix = np.vstack(features_list)

    if matrix.shape != (len(records), expected_dim):
        raise ValueError(
            f"Database matrix shape error: expected ({len(records)}, {expected_dim}), "
            f"got {matrix.shape}."
        )

    return records, matrix.astype(np.float64, copy=False)


def retrieve(
    query_image: str | PathLike[str],
    descriptor: str = DEFAULT_DESCRIPTOR,
    metric: str = DEFAULT_METRIC,
    top_k: int = 10,
    normalization: str = DEFAULT_NORMALIZATION,
) -> dict[str, Any]:
    """Retrieve deterministic Top-K most similar images for a query image.

    Args:
        query_image: Path to the query image (internal dataset or external image).
        descriptor: Feature descriptor ("rgb", "hsv", "lab", "rgb_hsv", "rgb_lab", "hsv_lab", "rgb_hsv_lab").
        metric: Distance metric ("euclidean", "manhattan", "cosine").
        top_k: Number of nearest neighbors to retrieve (> 0).
        normalization: Normalization strategy ("none" or "zscore").

    Returns:
        Structured dictionary containing query metadata, configuration, and sorted Top-K results.

    Raises:
        FileNotFoundError: If query image does not exist.
        ValueError: If parameters or feature matrix shapes are invalid.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer greater than 0.")
    if top_k <= 0:
        raise ValueError("top_k must be an integer greater than 0.")

    desc_clean = validate_descriptor(descriptor)
    metric_clean = validate_metric(metric)
    norm_clean = validate_normalization(normalization)

    query_path = _resolve_project_path(query_image)
    if not query_path.is_file():
        raise FileNotFoundError(f"Query image does not exist or is not a file: {query_path}")

    expected_dim = get_descriptor_dimension(desc_clean)

    # 1. Extract raw query feature vector
    raw_query_features = extract_descriptor(query_path, descriptor=desc_clean)
    query_features = validate_feature_vector(raw_query_features, expected_dim=expected_dim)

    # 2. Fetch corpus database records and feature matrix
    records, database_matrix = _get_database_descriptor_matrix(desc_clean)

    # Check for internal vs external query image mode
    query_key = _canonical_path(query_path)
    query_record = next(
        (
            record
            for record in records
            if _canonical_path(record["filepath"]) == query_key
        ),
        None,
    )

    # 3. Apply normalization if requested
    if norm_clean == "zscore":
        params = fit_normalizer(database_matrix)
        norm_matrix = transform(database_matrix, params["mean"], params["std"])
        norm_query = transform(query_features, params["mean"], params["std"])
    else:
        norm_matrix = database_matrix
        norm_query = query_features

    # 4. Compute vectorized distances
    distances = calculate_distances(norm_query, norm_matrix, metric=metric_clean)

    # 5. Filter candidates (exclude self-query if internal query image)
    candidates = [
        (record, float(distance))
        for record, distance in zip(records, distances)
        if query_record is None or record["image_id"] != query_record["image_id"]
    ]

    if top_k > len(candidates):
        raise ValueError(
            f"top_k={top_k} exceeds the number of available candidates ({len(candidates)})."
        )

    # 6. Sort deterministically by distance ascending, breaking ties by image_id
    candidates.sort(key=lambda item: (item[1], item[0]["image_id"]))

    results = [
        {
            "rank": rank,
            "image_id": record["image_id"],
            "filename": record["filename"],
            "filepath": record["filepath"],
            "category": record["category"],
            "distance": distance,
        }
        for rank, (record, distance) in enumerate(candidates[:top_k], start=1)
    ]

    return {
        "query": {
            "filepath": _display_path(query_path),
            "image_id": query_record["image_id"] if query_record else None,
            "category": query_record["category"] if query_record else None,
            "features": query_features,
        },
        "descriptor": desc_clean,
        "method": f"{desc_clean.upper()} Color Moments",
        "metric": metric_clean,
        "normalization": norm_clean,
        "top_k": top_k,
        "results": results,
    }


def retrieve_similar_images(
    query_image_path: str | PathLike[str],
    top_k: int = 10,
) -> dict[str, Any]:
    """Retrieve Top-K neighbors using RGB Color Moments and Euclidean distance (baseline compatibility)."""
    return retrieve(
        query_image=query_image_path,
        descriptor="rgb",
        metric="euclidean",
        top_k=top_k,
        normalization="none",
    )
