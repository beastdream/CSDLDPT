"""Evaluate one retrieval configuration over all 1000 WANG/Corel-1K images.

Every image is a leave-one-out query ranked against the other 999 images.
Reports Precision@K, Recall@K, F1@K (K = 5, 10, 20), AP per query and mAP.

Usage:
    python scripts/evaluate_retrieval.py                        # baseline: RGB, Euclidean, MySQL
    python scripts/evaluate_retrieval.py --color-spaces RGB HSV LAB --metric cosine --normalization zscore
    python scripts/evaluate_retrieval.py --source csv           # no MySQL needed
"""

import argparse
from collections import Counter
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.color_moments import COLOR_SPACE_CHANNELS
from src.experiment import (
    KS,
    NORMALIZATIONS,
    build_feature_matrix,
    evaluate_configuration,
    method_name,
    normalize_features,
    summarize,
)
from src.repository import FeatureRecord, load_features
from src.similarity import DISTANCE_METRICS


OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"
QUERY_OUTPUT = OUTPUT_DIR / "query_metrics.csv"
CATEGORY_OUTPUT = OUTPUT_DIR / "category_metrics.csv"
OVERALL_OUTPUT = OUTPUT_DIR / "overall_metrics.csv"
EXPECTED_QUERIES = 1000
EXPECTED_CATEGORIES = 10
EXPECTED_PER_CATEGORY = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--color-spaces", nargs="+", type=str.upper,
        choices=tuple(COLOR_SPACE_CHANNELS), default=["RGB"],
    )
    parser.add_argument("--metric", choices=DISTANCE_METRICS, default="euclidean")
    parser.add_argument("--normalization", choices=NORMALIZATIONS, default="none")
    parser.add_argument("--source", choices=("mysql", "csv"), default="mysql")
    return parser.parse_args()


def validate_source_records(records: list[FeatureRecord]) -> None:
    """Validate the fixed WANG evaluation population."""
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


def print_summary(
    method: str,
    args: argparse.Namespace,
    feature_dim: int,
    category_metrics: pd.DataFrame,
    overall: dict[str, float],
) -> None:
    """Print actual overall and per-category evaluation results."""
    print("\n" + "=" * 60)
    print("WANG/COREL-1K RETRIEVAL EVALUATION")
    print("=" * 60)
    print(f"Method            : Global {method} Color Moments")
    print(f"Feature dimension : {feature_dim}")
    print(f"Metric            : {args.metric.capitalize()}")
    print(f"Normalization     : {args.normalization}")
    print(f"Feature source    : {args.source}")
    print(f"Queries evaluated : {overall['num_queries']}")
    print(f"Relevant/query    : {EXPECTED_PER_CATEGORY - 1}")
    print("\n" + "-" * 60)
    print("OVERALL RESULTS")
    print("-" * 60)
    for name, prefix in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")):
        for k in KS:
            print(f"{name + '@' + str(k):<18}: {overall[f'{prefix}_at_{k}']:.4f}")
        print()
    print(f"{'mAP':<18}: {overall['map']:.4f}")

    print("\n" + "-" * 78)
    print(
        f"{'Category':<12}{'P@5':>8}{'P@10':>8}{'P@20':>8}"
        f"{'R@20':>8}{'F1@20':>8}{'mAP':>8}"
    )
    print("-" * 78)
    for _, row in category_metrics.sort_values("map", ascending=False).iterrows():
        print(
            f"{row['category']:<12}"
            f"{row['precision_at_5']:>8.4f}"
            f"{row['precision_at_10']:>8.4f}"
            f"{row['precision_at_20']:>8.4f}"
            f"{row['recall_at_20']:>8.4f}"
            f"{row['f1_at_20']:>8.4f}"
            f"{row['map']:>8.4f}"
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
    """Run the evaluation and write query, category and overall CSVs."""
    args = parse_args()
    try:
        records_by_space = {
            space: load_features(space, args.source) for space in args.color_spaces
        }
        for records in records_by_space.values():
            validate_source_records(records)

        matrix, records = build_feature_matrix(records_by_space, args.color_spaces)
        matrix = normalize_features(matrix, args.normalization)
        query_metrics = evaluate_configuration(matrix, records, args.metric, KS)
        overall, category_metrics = summarize(query_metrics)

        method = method_name(args.color_spaces)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        query_metrics.to_csv(QUERY_OUTPUT, index=False)
        category_metrics.to_csv(CATEGORY_OUTPUT, index=False)
        pd.DataFrame(
            [{
                "method": method,
                "feature_dim": matrix.shape[1],
                "distance_metric": args.metric,
                "normalization": args.normalization,
                "num_categories": category_metrics["category"].nunique(),
                **overall,
            }]
        ).to_csv(OVERALL_OUTPUT, index=False)
        print_summary(method, args, matrix.shape[1], category_metrics, overall)
        return 0
    except Exception as exc:
        print(f"ERROR: Retrieval evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
