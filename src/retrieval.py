"""Baseline content-based image retrieval using global RGB Color Moments."""

import os
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from src.color_moments import extract_color_moments
from src.repository import get_all_rgb_global_features
from src.similarity import euclidean_distances, validate_feature_vector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DESCRIPTOR_COUNT = 1000


def _resolve_project_path(path_value: str | PathLike[str]) -> Path:
    """Resolve an absolute path or a path relative to the project root."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _canonical_path(path_value: str | PathLike[str]) -> str:
    """Create a Windows-safe canonical path key for equality checks."""
    return os.path.normcase(str(_resolve_project_path(path_value)))


def _display_path(path: Path) -> str:
    """Prefer a project-relative, forward-slash path for public output."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def retrieve_similar_images(
    query_image_path: str | PathLike[str],
    top_k: int = 10,
) -> dict[str, Any]:
    """Retrieve deterministic Top-K neighbors using raw 9D Euclidean distance."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer greater than 0.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    query_path = _resolve_project_path(query_image_path)
    if not query_path.is_file():
        raise FileNotFoundError(f"Query image does not exist or is not a file: {query_path}")

    query_features = validate_feature_vector(extract_color_moments(query_path))
    records = get_all_rgb_global_features()
    if len(records) != EXPECTED_DESCRIPTOR_COUNT:
        raise ValueError(
            "Expected 1000 RGB/GLOBAL descriptors from MySQL, "
            f"received {len(records)}."
        )

    database_matrix = np.vstack([record["features"] for record in records])
    distances = euclidean_distances(query_features, database_matrix)

    query_key = _canonical_path(query_path)
    query_record = next(
        (
            record
            for record in records
            if _canonical_path(record["filepath"]) == query_key
        ),
        None,
    )

    candidates = [
        (record, float(distance))
        for record, distance in zip(records, distances)
        if query_record is None or record["image_id"] != query_record["image_id"]
    ]
    if top_k > len(candidates):
        raise ValueError(
            f"top_k={top_k} exceeds the number of available candidates "
            f"({len(candidates)})."
        )

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
        "method": "Global RGB Color Moments",
        "metric": "Euclidean",
        "top_k": top_k,
        "results": results,
    }
