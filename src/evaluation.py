"""Precision and recall metrics for ranked image-retrieval results."""

from collections.abc import Sequence
from typing import Any


def _validate_k(k: int, result_count: int) -> None:
    """Validate a metric cutoff against the available ranked results."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be an integer greater than 0.")
    if result_count < k:
        raise ValueError(
            f"At least {k} result categories are required; received {result_count}."
        )


def precision_at_k(
    query_category: str,
    result_categories: Sequence[str],
    k: int,
) -> float:
    """Return the fraction of the first K results matching the query category."""
    _validate_k(k, len(result_categories))
    relevant = sum(category == query_category for category in result_categories[:k])
    precision = relevant / k
    if not 0.0 <= precision <= 1.0:
        raise ValueError("Precision@K is outside the valid range [0, 1].")
    return float(precision)


def recall_at_k(
    query_category: str,
    result_categories: Sequence[str],
    k: int,
    total_relevant: int,
) -> float:
    """Return relevant results in the first K divided by all relevant candidates."""
    _validate_k(k, len(result_categories))
    if isinstance(total_relevant, bool) or not isinstance(total_relevant, int):
        raise ValueError("total_relevant must be an integer greater than 0.")
    if total_relevant <= 0:
        raise ValueError("total_relevant must be greater than 0.")

    relevant = sum(category == query_category for category in result_categories[:k])
    recall = relevant / total_relevant
    if not 0.0 <= recall <= 1.0:
        raise ValueError(
            "Recall@K is outside [0, 1]; check total_relevant and ranked results."
        )
    return float(recall)


def evaluate_query(
    query_category: str,
    results: Sequence[dict[str, Any]],
    ks: Sequence[int] = (5, 10, 20),
    total_relevant: int = 99,
) -> dict[str, float]:
    """Evaluate an already-ranked result list without sorting or retrieval."""
    try:
        result_categories = [str(result["category"]) for result in results]
    except (KeyError, TypeError) as exc:
        raise ValueError("Every ranked result must contain a category field.") from exc

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"precision_at_{k}"] = precision_at_k(
            query_category, result_categories, k
        )
        metrics[f"recall_at_{k}"] = recall_at_k(
            query_category, result_categories, k, total_relevant
        )
    return metrics
