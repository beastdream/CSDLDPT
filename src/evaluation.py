"""Precision, recall, F1, AP and mAP for ranked image-retrieval results."""

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


def f1_score(precision: float, recall: float) -> float:
    """Return the harmonic mean of precision and recall (0 when both are 0)."""
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def average_precision(
    query_category: str,
    result_categories: Sequence[str],
    total_relevant: int,
) -> float:
    """Return AP over a full ranking: mean of P@k at every relevant rank k.

    ``result_categories`` should be the complete ranking of all candidates
    (999 for WANG with self-exclusion) so that AP is not truncated. Relevant
    images missing from the ranking contribute 0, since the sum is divided by
    ``total_relevant``.
    """
    if isinstance(total_relevant, bool) or not isinstance(total_relevant, int):
        raise ValueError("total_relevant must be an integer greater than 0.")
    if total_relevant <= 0:
        raise ValueError("total_relevant must be greater than 0.")

    hits = 0
    precision_sum = 0.0
    for rank, category in enumerate(result_categories, start=1):
        if category == query_category:
            hits += 1
            precision_sum += hits / rank
    if hits > total_relevant:
        raise ValueError("Ranking contains more relevant items than total_relevant.")
    return float(precision_sum / total_relevant)


def mean_average_precision(average_precisions: Sequence[float]) -> float:
    """Return mAP, the mean of per-query Average Precision values."""
    if len(average_precisions) == 0:
        raise ValueError("At least one AP value is required to compute mAP.")
    return float(sum(average_precisions) / len(average_precisions))


def evaluate_query(
    query_category: str,
    results: Sequence[dict[str, Any]],
    ks: Sequence[int] = (5, 10, 20),
    total_relevant: int = 99,
    include_ap: bool = False,
) -> dict[str, float]:
    """Evaluate an already-ranked result list without sorting or retrieval.

    Returns ``precision_at_K``, ``recall_at_K`` and ``f1_at_K`` for every K.
    With ``include_ap=True`` it also returns ``average_precision``; pass the
    full ranking in that case.
    """
    try:
        result_categories = [str(result["category"]) for result in results]
    except (KeyError, TypeError) as exc:
        raise ValueError("Every ranked result must contain a category field.") from exc

    metrics: dict[str, float] = {}
    for k in ks:
        precision = precision_at_k(query_category, result_categories, k)
        recall = recall_at_k(query_category, result_categories, k, total_relevant)
        metrics[f"precision_at_{k}"] = precision
        metrics[f"recall_at_{k}"] = recall
        metrics[f"f1_at_{k}"] = f1_score(precision, recall)
    if include_ap:
        metrics["average_precision"] = average_precision(
            query_category, result_categories, total_relevant
        )
    return metrics
