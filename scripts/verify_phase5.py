"""Verify Phase 5 evaluation artifacts and metric consistency."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.repository import get_all_rgb_global_features


OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"
QUERY_PATH = OUTPUT_DIR / "query_metrics.csv"
CATEGORY_PATH = OUTPUT_DIR / "category_metrics.csv"
OVERALL_PATH = OUTPUT_DIR / "overall_metrics.csv"
KS = (5, 10, 20)
AT_K_COLUMNS = tuple(
    [f"precision_at_{k}" for k in KS]
    + [f"recall_at_{k}" for k in KS]
    + [f"f1_at_{k}" for k in KS]
)
# Per query the AP column is average_precision; aggregated it is reported as map.
QUERY_METRIC_COLUMNS = (*AT_K_COLUMNS, "average_precision")
METRIC_COLUMNS = (*AT_K_COLUMNS, "map")
QUERY_COLUMNS = ("query_id", "filename", "category", *QUERY_METRIC_COLUMNS)
CATEGORY_COLUMNS = ("category", "num_queries", *METRIC_COLUMNS)
OVERALL_COLUMNS = {
    "method", "feature_dim", "distance_metric", "normalization",
    "num_queries", "num_categories", *METRIC_COLUMNS,
}
CHECK_NAMES = (
    "Repository images",
    "Repository categories",
    "Images per category",
    "Evaluation files",
    "Query metric rows",
    "Category metric rows",
    "Unique complete queries",
    "Queries per category",
    "Metric ranges",
    "Precision/Recall relation",
    "F1 relation",
    "Category aggregation",
    "Overall aggregation",
    "Evaluation totals",
)


def _metric_frame_is_valid(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    """Return whether every metric value is finite and lies in [0, 1]."""
    values = frame.loc[:, columns].to_numpy(dtype=np.float64)
    return bool(np.all(np.isfinite(values)) and np.all((values >= 0) & (values <= 1)))


def main() -> int:
    """Validate source records, CSV schemas, aggregates, and metric identities."""
    checks = {name: False for name in CHECK_NAMES}
    error: Exception | None = None

    try:
        records = get_all_rgb_global_features()
        source_counts = Counter(record["category"] for record in records)
        checks["Repository images"] = len(records) == 1000
        checks["Repository categories"] = len(source_counts) == 10
        checks["Images per category"] = all(
            count == 100 for count in source_counts.values()
        )

        checks["Evaluation files"] = all(
            path.is_file() for path in (QUERY_PATH, CATEGORY_PATH, OVERALL_PATH)
        )
        if not checks["Evaluation files"]:
            raise FileNotFoundError(
                "Evaluation CSV files are missing. Run scripts/evaluate_retrieval.py."
            )

        query = pd.read_csv(QUERY_PATH)
        category = pd.read_csv(CATEGORY_PATH)
        overall = pd.read_csv(OVERALL_PATH)
        if tuple(query.columns) != QUERY_COLUMNS:
            raise ValueError("query_metrics.csv has an invalid schema.")
        if tuple(category.columns) != CATEGORY_COLUMNS:
            raise ValueError("category_metrics.csv has an invalid schema.")
        if set(overall.columns) != OVERALL_COLUMNS:
            raise ValueError("overall_metrics.csv has an invalid schema.")

        checks["Query metric rows"] = len(query) == 1000
        checks["Category metric rows"] = len(category) == 10
        checks["Unique complete queries"] = (
            query["query_id"].nunique() == 1000 and not query.isna().any().any()
        )
        query_counts = query["category"].value_counts().to_dict()
        checks["Queries per category"] = (
            query_counts == dict(source_counts)
            and all(count == 100 for count in query_counts.values())
        )
        checks["Metric ranges"] = (
            _metric_frame_is_valid(query, QUERY_METRIC_COLUMNS)
            and _metric_frame_is_valid(category, METRIC_COLUMNS)
            and _metric_frame_is_valid(overall, METRIC_COLUMNS)
        )

        relation_ok = True
        for k in KS:
            expected_recall = query[f"precision_at_{k}"] * k / 99
            relation_ok &= bool(
                np.allclose(query[f"recall_at_{k}"], expected_recall, atol=1e-12)
            )
        checks["Precision/Recall relation"] = relation_ok

        f1_ok = True
        for k in KS:
            precision = query[f"precision_at_{k}"]
            recall = query[f"recall_at_{k}"]
            denominator = (precision + recall).replace(0, np.nan)
            expected_f1 = (2 * precision * recall / denominator).fillna(0.0)
            f1_ok &= bool(np.allclose(query[f"f1_at_{k}"], expected_f1, atol=1e-12))
        checks["F1 relation"] = f1_ok

        query = query.rename(columns={"average_precision": "map"})
        expected_category = (
            query.groupby("category", as_index=False, sort=True)
            .agg(
                num_queries=("query_id", "count"),
                **{column: (column, "mean") for column in METRIC_COLUMNS},
            )
            .loc[:, CATEGORY_COLUMNS]
        )
        actual_category = category.sort_values("category").reset_index(drop=True)
        checks["Category aggregation"] = (
            expected_category[["category", "num_queries"]].equals(
                actual_category[["category", "num_queries"]]
            )
            and np.allclose(
                expected_category.loc[:, METRIC_COLUMNS],
                actual_category.loc[:, METRIC_COLUMNS],
                atol=1e-12,
            )
        )

        expected_overall = query.loc[:, METRIC_COLUMNS].mean().to_numpy()
        checks["Overall aggregation"] = (
            len(overall) == 1
            and np.allclose(
                overall.loc[0, list(METRIC_COLUMNS)].to_numpy(dtype=np.float64),
                expected_overall,
                atol=1e-12,
            )
        )
        checks["Evaluation totals"] = (
            len(overall) == 1
            and int(overall.loc[0, "num_queries"]) == 1000
            and int(overall.loc[0, "num_categories"]) == 10
        )
    except Exception as exc:
        error = exc

    complete = all(checks.values())
    print("=" * 60)
    print("PHASE 5 VERIFICATION")
    print("=" * 60)
    for name in CHECK_NAMES:
        print(f"{name:<29}: {'PASS' if checks[name] else 'FAIL'}")
    if error is not None:
        print(f"\nError: {error}")
    print(f"\nPHASE 5 STATUS: {'COMPLETE' if complete else 'INCOMPLETE'}")
    print("=" * 60)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
