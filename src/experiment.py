"""Leave-one-out retrieval evaluation for Color Moments configurations.

A configuration is a combination of color spaces (e.g. ``("RGB", "HSV")``), a
distance metric and a normalization. Every image is used once as the query and
ranked against the 999 other images.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.color_moments import normalize_color_space
from src.evaluation import evaluate_query
from src.repository import FeatureRecord
from src.similarity import compute_distances


KS = (5, 10, 20)
NORMALIZATIONS = ("none", "zscore")
METRIC_COLUMNS = (
    *(f"precision_at_{k}" for k in KS),
    *(f"recall_at_{k}" for k in KS),
    *(f"f1_at_{k}" for k in KS),
    "average_precision",
)


def method_name(color_spaces: Sequence[str]) -> str:
    """Return a display name such as ``RGB+HSV``."""
    return "+".join(normalize_color_space(space) for space in color_spaces)


def build_feature_matrix(
    records_by_space: Mapping[str, Sequence[FeatureRecord]],
    color_spaces: Sequence[str],
) -> tuple[NDArray[np.float64], list[FeatureRecord]]:
    """Concatenate the 9D vectors of ``color_spaces`` into an N x (9*S) matrix.

    All spaces must describe the same images in the same image_id order. The
    returned records are those of the first space (image metadata only).
    """
    if not color_spaces:
        raise ValueError("At least one color space is required.")
    spaces = [normalize_color_space(space) for space in color_spaces]
    base = list(records_by_space[spaces[0]])
    base_ids = [record["image_id"] for record in base]
    blocks = []
    for space in spaces:
        records = records_by_space[space]
        if [record["image_id"] for record in records] != base_ids:
            raise ValueError(f"{space} features do not cover the same images as {spaces[0]}.")
        blocks.append(np.vstack([record["features"] for record in records]))
    return np.hstack(blocks), base


def normalize_features(matrix: NDArray[np.float64], normalization: str) -> NDArray[np.float64]:
    """Apply per-dimension normalization over the whole database.

    ``zscore`` maps each column to mean 0 / std 1 so that channels with large
    ranges (e.g. RGB 0-255, Hue 0-360) do not dominate small ones (S, V 0-1).
    """
    if normalization == "none":
        return matrix
    if normalization == "zscore":
        std = matrix.std(axis=0)
        std[std == 0] = 1.0
        return (matrix - matrix.mean(axis=0)) / std
    raise ValueError(f"Unsupported normalization {normalization!r}; expected {NORMALIZATIONS}.")


def evaluate_configuration(
    matrix: NDArray[np.float64],
    records: Sequence[FeatureRecord],
    metric: str = "euclidean",
    ks: Sequence[int] = KS,
) -> pd.DataFrame:
    """Rank all other images for every query and return one metrics row per query.

    Ties are broken by image_id for deterministic output. Relevant images are
    those of the query's category, excluding the query itself.
    """
    if matrix.shape[0] != len(records):
        raise ValueError("Feature matrix rows do not match the number of records.")

    categories = np.asarray([record["category"] for record in records])
    image_ids = np.asarray([record["image_id"] for record in records], dtype=np.int64)
    category_sizes = pd.Series(categories).value_counts().to_dict()

    rows: list[dict[str, object]] = []
    for query_index, record in enumerate(records):
        distances = compute_distances(matrix[query_index], matrix, metric)
        candidates = np.flatnonzero(np.arange(len(records)) != query_index)
        order = candidates[np.lexsort((image_ids[candidates], distances[candidates]))]
        ranked = [{"category": category} for category in categories[order]]

        metrics = evaluate_query(
            record["category"],
            ranked,
            ks=ks,
            total_relevant=int(category_sizes[record["category"]]) - 1,
            include_ap=True,
        )
        rows.append(
            {
                "query_id": record["image_id"],
                "filename": record["filename"],
                "category": record["category"],
                **metrics,
            }
        )
    columns = [f"{name}_at_{k}" for name in ("precision", "recall", "f1") for k in ks]
    return pd.DataFrame(rows, columns=["query_id", "filename", "category", *columns, "average_precision"])


def top_k_confusion(
    matrix: NDArray[np.float64],
    records: Sequence[FeatureRecord],
    metric: str = "euclidean",
    k: int = 10,
) -> pd.DataFrame:
    """Return a category x category table: % of each category's Top-K results
    that belong to every category (the diagonal is Precision@K)."""
    categories = np.asarray([record["category"] for record in records])
    image_ids = np.asarray([record["image_id"] for record in records], dtype=np.int64)
    labels = sorted(set(categories))
    counts = pd.DataFrame(0, index=labels, columns=labels, dtype=np.int64)
    for query_index in range(len(records)):
        distances = compute_distances(matrix[query_index], matrix, metric)
        candidates = np.flatnonzero(np.arange(len(records)) != query_index)
        order = candidates[np.lexsort((image_ids[candidates], distances[candidates]))]
        for category in categories[order[:k]]:
            counts.loc[categories[query_index], category] += 1
    return counts.div(counts.sum(axis=1), axis=0) * 100


def summarize(query_metrics: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Return overall means and per-category means; AP mean is reported as ``map``."""
    metric_columns = [column for column in METRIC_COLUMNS if column in query_metrics]
    overall = {column: float(query_metrics[column].mean()) for column in metric_columns}
    overall["map"] = overall.pop("average_precision")
    overall["num_queries"] = len(query_metrics)

    by_category = (
        query_metrics.groupby("category", sort=True)[metric_columns]
        .mean()
        .rename(columns={"average_precision": "map"})
        .reset_index()
    )
    by_category.insert(1, "num_queries", query_metrics.groupby("category", sort=True).size().values)
    return overall, by_category
