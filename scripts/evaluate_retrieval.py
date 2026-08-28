"""Evaluate Phase 4 retrieval over all 1000 WANG/Corel-1K images."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_query
from src.repository import get_all_rgb_global_features
from src.similarity import euclidean_distances


OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"
QUERY_OUTPUT = OUTPUT_DIR / "query_metrics.csv"
CATEGORY_OUTPUT = OUTPUT_DIR / "category_metrics.csv"
OVERALL_OUTPUT = OUTPUT_DIR / "overall_metrics.csv"
KS = (5, 10, 20)
EXPECTED_QUERIES = 1000
EXPECTED_CATEGORIES = 10
EXPECTED_PER_CATEGORY = 100
TOTAL_RELEVANT = 99
METRIC_COLUMNS = tuple(
    [f"precision_at_{k}" for k in KS] + [f"recall_at_{k}" for k in KS]
)
QUERY_COLUMNS = ("query_id", "filename", "category", *METRIC_COLUMNS)
CATEGORY_COLUMNS = ("category", "num_queries", *METRIC_COLUMNS)
OVERALL_COLUMNS = ("num_queries", "num_categories", *METRIC_COLUMNS)


def validate_source_records(records: list[dict]) -> None:
    """Validate the fixed WANG evaluation population from MySQL."""
    if len(records) != EXPECTED_QUERIES:
        raise ValueError(f"Expected 1000 repository records, found {len(records)}.")

    image_ids = [record["image_id"] for record in records]
    if len(set(image_ids)) != EXPECTED_QUERIES:
        raise ValueError("Repository image_id values are not unique.")

    category_counts = Counter(record["category"] for record in records)
    if len(category_counts) != EXPECTED_CATEGORIES:
        raise ValueError(f"Expected 10 categories, found {len(category_counts)}.")
    invalid_counts = {
        category: count
        for category, count in category_counts.items()
        if count != EXPECTED_PER_CATEGORY
    }
    if invalid_counts:
        raise ValueError(f"Expected 100 images per category; found {invalid_counts}.")


def evaluate_all_queries(records: list[dict]) -> pd.DataFrame:
    """Rank all database descriptors and return per-query P@K and R@K."""
    feature_matrix = np.vstack([record["features"] for record in records])
    if feature_matrix.shape != (EXPECTED_QUERIES, 9):
        raise ValueError(
            f"Expected feature matrix shape (1000, 9), found {feature_matrix.shape}."
        )
    if not np.all(np.isfinite(feature_matrix)):
        raise ValueError("Feature matrix contains NaN or Inf.")

    image_ids = np.asarray([record["image_id"] for record in records], dtype=np.int64)
    query_rows: list[dict[str, object]] = []

    for query_index, query_record in enumerate(records):
        progress = query_index + 1
        if progress == 1 or progress % 100 == 0:
            print(f"Evaluating [{progress}/{EXPECTED_QUERIES}]")

        distances = euclidean_distances(
            query_record["features"], feature_matrix
        )
        candidate_mask = image_ids != query_record["image_id"]
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size != EXPECTED_QUERIES - 1:
            raise ValueError(
                f"Query {query_record['image_id']} has {candidate_indices.size} "
                "candidates after self-exclusion; expected 999."
            )

        candidate_distances = distances[candidate_indices]
        candidate_ids = image_ids[candidate_indices]
        ranking = np.lexsort((candidate_ids, candidate_distances))
        top_indices = candidate_indices[ranking[: max(KS)]]
        ranked_results = [
            {
                "image_id": records[index]["image_id"],
                "category": records[index]["category"],
                "distance": float(distances[index]),
            }
            for index in top_indices
        ]

        metrics = evaluate_query(
            query_record["category"],
            ranked_results,
            ks=KS,
            total_relevant=TOTAL_RELEVANT,
        )
        query_rows.append(
            {
                "query_id": query_record["image_id"],
                "filename": query_record["filename"],
                "category": query_record["category"],
                **metrics,
            }
        )

    return pd.DataFrame(query_rows, columns=QUERY_COLUMNS)


def aggregate_metrics(
    query_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute category and overall means without premature rounding."""
    category_metrics = (
        query_metrics.groupby("category", as_index=False, sort=True)
        .agg(
            num_queries=("query_id", "count"),
            **{column: (column, "mean") for column in METRIC_COLUMNS},
        )
        .loc[:, CATEGORY_COLUMNS]
    )
    overall_data: dict[str, object] = {
        "num_queries": len(query_metrics),
        "num_categories": query_metrics["category"].nunique(),
    }
    overall_data.update(
        {column: float(query_metrics[column].mean()) for column in METRIC_COLUMNS}
    )
    overall_metrics = pd.DataFrame([overall_data], columns=OVERALL_COLUMNS)
    return category_metrics, overall_metrics


def save_outputs(
    query_metrics: pd.DataFrame,
    category_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame,
) -> None:
    """Write the three required deterministic CSV artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    query_metrics.to_csv(QUERY_OUTPUT, index=False)
    category_metrics.to_csv(CATEGORY_OUTPUT, index=False)
    overall_metrics.to_csv(OVERALL_OUTPUT, index=False)


def print_summary(
    category_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame,
) -> None:
    """Print actual overall and per-category evaluation results."""
    overall = overall_metrics.iloc[0]
    print("\n" + "=" * 60)
    print("WANG/COREL-1K RETRIEVAL EVALUATION")
    print("=" * 60)
    print("Method            : Global RGB Color Moments")
    print("Feature dimension : 9")
    print("Metric            : Euclidean")
    print(f"Queries evaluated : {int(overall['num_queries'])}")
    print(f"Categories        : {int(overall['num_categories'])}")
    print(f"Relevant/query    : {TOTAL_RELEVANT}")
    print("\n" + "-" * 60)
    print("OVERALL RESULTS")
    print("-" * 60)
    for k in KS:
        print(f"Precision@{k:<2}      : {overall[f'precision_at_{k}']:.4f}")
    print()
    for k in KS:
        print(f"Recall@{k:<2}         : {overall[f'recall_at_{k}']:.4f}")

    print("\n" + "-" * 78)
    print(
        f"{'Category':<14}{'P@5':>8}{'P@10':>8}{'P@20':>8}"
        f"{'R@5':>8}{'R@10':>8}{'R@20':>8}"
    )
    print("-" * 78)
    for _, row in category_metrics.sort_values("category").iterrows():
        print(
            f"{row['category']:<14}"
            f"{row['precision_at_5']:>8.4f}"
            f"{row['precision_at_10']:>8.4f}"
            f"{row['precision_at_20']:>8.4f}"
            f"{row['recall_at_5']:>8.4f}"
            f"{row['recall_at_10']:>8.4f}"
            f"{row['recall_at_20']:>8.4f}"
        )

    print("\n" + "-" * 60)
    print("OUTPUT")
    print("-" * 60)
    print("Query metrics   : results/evaluation/query_metrics.csv")
    print("Category metrics: results/evaluation/category_metrics.csv")
    print("Overall metrics : results/evaluation/overall_metrics.csv")
    print("\nStatus: SUCCESS")
    print("=" * 60)


def main() -> int:
    """Run full Phase 5 evaluation from the MySQL source of truth."""
    try:
        records = get_all_rgb_global_features()
        validate_source_records(records)
        query_metrics = evaluate_all_queries(records)
        category_metrics, overall_metrics = aggregate_metrics(query_metrics)
        save_outputs(query_metrics, category_metrics, overall_metrics)
        print_summary(category_metrics, overall_metrics)
        return 0
    except Exception as exc:
        print(f"ERROR: Retrieval evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
