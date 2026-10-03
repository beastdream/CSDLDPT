"""Compare Color Moments configurations on WANG/Corel-1K and export CSVs.

Every combination of color spaces (RGB, HSV, LAB and their unions) is
evaluated with every distance metric and normalization, using all 1000 images
as leave-one-out queries. The baseline is the current system: RGB, Euclidean,
no normalization.

Usage:
    python scripts/run_experiments.py                 # features from data/features/*.csv
    python scripts/run_experiments.py --source mysql  # features from MySQL
    python scripts/run_experiments.py --save-db       # also store results in MySQL

Outputs (results/experiments/):
    overall_results.csv        one row per configuration, with gain vs baseline
    category_results.csv       one row per configuration x category
    category_map_pivot.csv     mAP per category x method (best metric/normalization)
    query_results_best.csv     per-query metrics of the baseline and the best configuration
    confusion_top10_*.csv      % of Top-10 results per category pair (baseline and best)
"""

import argparse
from itertools import combinations
from pathlib import Path
import sys
import time

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
    top_k_confusion,
)
from src.repository import load_features, save_experiment_results
from src.similarity import DISTANCE_METRICS


OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments"
BASELINE = ("RGB", "euclidean", "none")
SPACES = tuple(COLOR_SPACE_CHANNELS)
METHODS = tuple(
    combo for size in range(1, len(SPACES) + 1) for combo in combinations(SPACES, size)
)
CONFIG_COLUMNS = ("method", "feature_dim", "distance_metric", "normalization")
METRIC_COLUMNS = (
    *(f"precision_at_{k}" for k in KS),
    *(f"recall_at_{k}" for k in KS),
    *(f"f1_at_{k}" for k in KS),
    "map",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("csv", "mysql"), default="csv")
    parser.add_argument(
        "--metrics", nargs="+", choices=DISTANCE_METRICS, default=list(DISTANCE_METRICS)
    )
    parser.add_argument(
        "--normalizations", nargs="+", choices=NORMALIZATIONS, default=list(NORMALIZATIONS)
    )
    parser.add_argument("--save-db", action="store_true", help="Store results in MySQL.")
    parser.add_argument("--experiment-name", default="color_moments_wang")
    return parser.parse_args()


def run_all(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate every configuration; return overall rows, category rows, query frames."""
    records_by_space = {space: load_features(space, args.source) for space in SPACES}
    overall_rows: list[dict] = []
    category_frames: list[pd.DataFrame] = []
    query_frames: dict[tuple[str, str, str], pd.DataFrame] = {}

    configs = [
        (spaces, metric, normalization)
        for spaces in METHODS
        for normalization in args.normalizations
        for metric in args.metrics
    ]
    for index, (spaces, metric, normalization) in enumerate(configs, start=1):
        method = method_name(spaces)
        started = time.perf_counter()
        matrix, records = build_feature_matrix(records_by_space, spaces)
        matrix = normalize_features(matrix, normalization)
        query_metrics = evaluate_configuration(matrix, records, metric, KS)
        overall, by_category = summarize(query_metrics)

        config = {
            "method": method,
            "feature_dim": matrix.shape[1],
            "distance_metric": metric,
            "normalization": normalization,
        }
        overall_rows.append({**config, **overall})
        category_frames.append(by_category.assign(**config))
        query_frames[(method, metric, normalization)] = query_metrics
        print(
            f"[{index:>2}/{len(configs)}] {method:<12} {metric:<10} {normalization:<7} "
            f"P@10={overall['precision_at_10']:.4f}  mAP={overall['map']:.4f}  "
            f"({time.perf_counter() - started:.1f}s)"
        )

    overall_df = pd.DataFrame(overall_rows)
    category_df = pd.concat(category_frames, ignore_index=True)
    return overall_df, category_df, query_frames


def add_baseline_gain(overall: pd.DataFrame, category: pd.DataFrame) -> None:
    """Add absolute and relative mAP / P@10 gains versus the baseline in place."""
    method, metric, normalization = BASELINE
    base = overall[
        (overall["method"] == method)
        & (overall["distance_metric"] == metric)
        & (overall["normalization"] == normalization)
    ].iloc[0]
    for column in ("map", "precision_at_10"):
        overall[f"{column}_gain"] = overall[column] - base[column]
        overall[f"{column}_gain_pct"] = 100 * overall[f"{column}_gain"] / base[column]

    base_category = category[
        (category["method"] == method)
        & (category["distance_metric"] == metric)
        & (category["normalization"] == normalization)
    ].set_index("category")["map"]
    category["map_gain"] = category["map"] - category["category"].map(base_category)


def save_outputs(
    overall: pd.DataFrame,
    category: pd.DataFrame,
    query_frames: dict,
) -> tuple[pd.Series, pd.Series]:
    """Write all CSVs and return (baseline row, best row)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall = overall.sort_values(["map", "precision_at_10"], ascending=False)
    overall.to_csv(OUTPUT_DIR / "overall_results.csv", index=False)

    category = category.loc[
        :, [*CONFIG_COLUMNS, "category", "num_queries", *METRIC_COLUMNS, "map_gain"]
    ].sort_values([*CONFIG_COLUMNS, "category"])
    category.to_csv(OUTPUT_DIR / "category_results.csv", index=False)

    # For each method, keep its best metric/normalization, then pivot category x method.
    best_per_method = overall.drop_duplicates("method")
    pivot_source = category.merge(
        best_per_method[["method", "distance_metric", "normalization"]],
        on=["method", "distance_metric", "normalization"],
    )
    pivot = pivot_source.pivot(index="category", columns="method", values="map")
    pivot = pivot[[method_name(spaces) for spaces in METHODS]]
    baseline_label = f"{BASELINE[0]} baseline ({BASELINE[1]}, {BASELINE[2]})"
    pivot.insert(
        0,
        baseline_label,
        category[
            (category["method"] == BASELINE[0])
            & (category["distance_metric"] == BASELINE[1])
            & (category["normalization"] == BASELINE[2])
        ].set_index("category")["map"],
    )
    pivot.to_csv(OUTPUT_DIR / "category_map_pivot.csv")

    best = overall.iloc[0]
    baseline = overall[
        (overall["method"] == BASELINE[0])
        & (overall["distance_metric"] == BASELINE[1])
        & (overall["normalization"] == BASELINE[2])
    ].iloc[0]
    best_key = (best["method"], best["distance_metric"], best["normalization"])
    query_best = pd.concat(
        [
            query_frames[BASELINE].assign(config="baseline"),
            query_frames[best_key].assign(config="best"),
        ],
        ignore_index=True,
    )
    for column, value in zip(("method", "distance_metric", "normalization"), BASELINE):
        query_best.loc[query_best["config"] == "baseline", column] = value
    for column, value in zip(("method", "distance_metric", "normalization"), best_key):
        query_best.loc[query_best["config"] == "best", column] = value
    query_best.to_csv(OUTPUT_DIR / "query_results_best.csv", index=False)
    return baseline, best


def save_confusions(args: argparse.Namespace, best: pd.Series) -> None:
    """Write Top-10 category confusion tables for the baseline and best configs."""
    records_by_space = {space: load_features(space, args.source) for space in SPACES}
    best_key = (best["method"], best["distance_metric"], best["normalization"])
    for label, (method, metric, normalization) in (("baseline", BASELINE), ("best", best_key)):
        matrix, records = build_feature_matrix(records_by_space, method.split("+"))
        matrix = normalize_features(matrix, normalization)
        confusion = top_k_confusion(matrix, records, metric, k=10)
        confusion.round(1).to_csv(OUTPUT_DIR / f"confusion_top10_{label}.csv")


def print_report(overall: pd.DataFrame, baseline: pd.Series, best: pd.Series) -> None:
    ranked = overall.sort_values(["map", "precision_at_10"], ascending=False)
    print("\n" + "=" * 96)
    print("TOP CONFIGURATIONS (sorted by mAP)")
    print("=" * 96)
    print(
        f"{'Method':<13}{'Dim':>4} {'Metric':<10}{'Norm':<7}"
        f"{'P@5':>7}{'P@10':>7}{'P@20':>7}{'R@20':>7}{'F1@20':>7}{'mAP':>7}{'dmAP%':>8}"
    )
    for _, row in ranked.head(15).iterrows():
        print(
            f"{row['method']:<13}{row['feature_dim']:>4} {row['distance_metric']:<10}"
            f"{row['normalization']:<7}{row['precision_at_5']:>7.4f}"
            f"{row['precision_at_10']:>7.4f}{row['precision_at_20']:>7.4f}"
            f"{row['recall_at_20']:>7.4f}{row['f1_at_20']:>7.4f}{row['map']:>7.4f}"
            f"{row['map_gain_pct']:>+8.1f}"
        )

    print("\n" + "-" * 96)
    print(
        f"Baseline : RGB / euclidean / none  P@10={baseline['precision_at_10']:.4f}  "
        f"mAP={baseline['map']:.4f}"
    )
    print(
        f"Best     : {best['method']} / {best['distance_metric']} / {best['normalization']}  "
        f"P@10={best['precision_at_10']:.4f}  mAP={best['map']:.4f}"
    )
    print(
        f"Gain     : mAP {best['map_gain']:+.4f} ({best['map_gain_pct']:+.1f}%), "
        f"P@10 {best['precision_at_10_gain']:+.4f} ({best['precision_at_10_gain_pct']:+.1f}%)"
    )
    print(f"Outputs  : {OUTPUT_DIR.relative_to(PROJECT_ROOT).as_posix()}/")


def main() -> int:
    args = parse_args()
    if "euclidean" not in args.metrics or "none" not in args.normalizations:
        print("ERROR: The baseline needs --metrics euclidean and --normalizations none.")
        return 1
    try:
        overall, category, query_frames = run_all(args)
        add_baseline_gain(overall, category)
        baseline, best = save_outputs(overall, category, query_frames)
        save_confusions(args, best)
        print_report(overall, baseline, best)
        if args.save_db:
            saved = save_experiment_results(
                args.experiment_name,
                overall.to_dict("records"),
                category.to_dict("records"),
            )
            print(f"MySQL    : saved {saved} runs to experiment_runs")
        return 0
    except Exception as exc:
        print(f"ERROR: Experiment failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
