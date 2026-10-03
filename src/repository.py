"""Read Color Moments descriptors (RGB/HSV/LAB) from MySQL or feature CSVs.

Also stores experiment results in MySQL.
"""

import csv
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from src.color_moments import feature_names, normalize_color_space
from src.database import connection_scope


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DB_FEATURE_COLUMNS = tuple(
    f"c{channel}_{moment}" for channel in (1, 2, 3) for moment in ("mean", "std", "skew")
)


class FeatureRecord(TypedDict):
    """One image and its ordered 9D Color Moments descriptor."""

    image_id: int
    filename: str
    filepath: str
    category: str
    features: NDArray[np.float64]


GLOBAL_FEATURE_QUERY = f"""
SELECT
    i.image_id,
    i.filename,
    i.filepath,
    c.category_name,
    {", ".join(f"f.{column}" for column in DB_FEATURE_COLUMNS)}
FROM images AS i
JOIN categories AS c
    ON i.category_id = c.category_id
JOIN color_moment_features AS f
    ON i.image_id = f.image_id
WHERE f.color_space = %s
  AND f.feature_type = %s
ORDER BY i.image_id
"""


def feature_csv_path(color_space: str) -> Path:
    """Return ``data/features/color_moments_<space>.csv`` for a color space."""
    return FEATURES_DIR / f"color_moments_{normalize_color_space(color_space).lower()}.csv"


def _to_record(
    image_id: Any,
    filename: Any,
    filepath: Any,
    category: Any,
    values: Any,
    label: str,
) -> FeatureRecord:
    """Build a validated record with a finite ``(9,)`` float64 vector."""
    features = np.asarray(values, dtype=np.float64)
    if features.shape != (9,):
        raise ValueError(
            f"Image ID {image_id} has feature shape {features.shape}; expected (9,)."
        )
    if not np.all(np.isfinite(features)):
        raise ValueError(f"Image ID {image_id} has {label} features containing NaN or Inf.")
    return {
        "image_id": int(image_id),
        "filename": str(filename),
        "filepath": str(filepath),
        "category": str(category),
        "features": features,
    }


def get_global_features(color_space: str = "RGB") -> list[FeatureRecord]:
    """Return every GLOBAL descriptor of ``color_space`` from MySQL by image ID."""
    space = normalize_color_space(color_space)
    with connection_scope() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(GLOBAL_FEATURE_QUERY, (space, "GLOBAL"))
            rows = cursor.fetchall()
        finally:
            cursor.close()

    if not rows:
        raise ValueError(f"MySQL returned no {space}/GLOBAL Color Moments descriptors.")
    return [
        _to_record(row[0], row[1], row[2], row[3], row[4:13], f"{space}/GLOBAL")
        for row in rows
    ]


def get_all_rgb_global_features() -> list[FeatureRecord]:
    """Return every RGB/GLOBAL descriptor from MySQL ordered by image ID."""
    return get_global_features("RGB")


def load_features_from_csv(color_space: str = "RGB") -> list[FeatureRecord]:
    """Read descriptors from the extracted feature CSV, ordered by image ID."""
    space = normalize_color_space(color_space)
    path = feature_csv_path(space)
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature CSV not found: {path}. Run "
            f"scripts/extract_features.py --color-space {space} first."
        )
    columns = feature_names(space)
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"Feature CSV is empty: {path}")

    records = [
        _to_record(
            row["image_id"],
            row["filename"],
            row["filepath"],
            row["category"],
            [float(row[column]) for column in columns],
            f"{space}/GLOBAL",
        )
        for row in rows
    ]
    return sorted(records, key=lambda record: record["image_id"])


def load_features(color_space: str, source: str = "csv") -> list[FeatureRecord]:
    """Load descriptors from ``source`` = ``csv`` or ``mysql``."""
    if source == "csv":
        return load_features_from_csv(color_space)
    if source == "mysql":
        return get_global_features(color_space)
    raise ValueError(f"Unsupported feature source {source!r}; expected csv or mysql.")


EXPERIMENT_METRIC_COLUMNS = (
    "precision_at_5", "precision_at_10", "precision_at_20",
    "recall_at_5", "recall_at_10", "recall_at_20",
    "f1_at_5", "f1_at_10", "f1_at_20",
    "map_score",
)

RUN_UPSERT = f"""
INSERT INTO experiment_runs (
    experiment_name, method, distance_metric, normalization, feature_dim, num_queries,
    {", ".join(EXPERIMENT_METRIC_COLUMNS)}
)
VALUES ({", ".join(["%s"] * (6 + len(EXPERIMENT_METRIC_COLUMNS)))})
ON DUPLICATE KEY UPDATE
    run_id = LAST_INSERT_ID(run_id),
    feature_dim = VALUES(feature_dim),
    num_queries = VALUES(num_queries),
    {", ".join(f"{column} = VALUES({column})" for column in EXPERIMENT_METRIC_COLUMNS)},
    created_at = CURRENT_TIMESTAMP
"""

CATEGORY_RESULT_UPSERT = f"""
INSERT INTO experiment_category_results (
    run_id, category_id, num_queries, {", ".join(EXPERIMENT_METRIC_COLUMNS)}
)
VALUES ({", ".join(["%s"] * (3 + len(EXPERIMENT_METRIC_COLUMNS)))})
ON DUPLICATE KEY UPDATE
    num_queries = VALUES(num_queries),
    {", ".join(f"{column} = VALUES({column})" for column in EXPERIMENT_METRIC_COLUMNS)}
"""


def save_experiment_results(
    experiment_name: str,
    overall_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
) -> int:
    """Upsert overall and per-category experiment rows in one transaction.

    Rows are the dictionaries written to the experiment CSVs; each needs
    ``method``, ``distance_metric``, ``normalization`` and the metric columns
    (``map`` is stored as ``map_score``). Returns the number of runs saved.
    """
    def metric_values(row: dict[str, Any]) -> list[float]:
        return [float(row["map" if column == "map_score" else column])
                for column in EXPERIMENT_METRIC_COLUMNS]

    def config_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return row["method"], row["distance_metric"], row["normalization"]

    with connection_scope() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT category_name, category_id FROM categories")
            category_ids = dict(cursor.fetchall())
            connection.commit()
            connection.start_transaction()

            run_ids: dict[tuple[str, str, str], int] = {}
            for row in overall_rows:
                cursor.execute(
                    RUN_UPSERT,
                    (
                        experiment_name, row["method"], row["distance_metric"],
                        row["normalization"], int(row["feature_dim"]),
                        int(row["num_queries"]), *metric_values(row),
                    ),
                )
                run_ids[config_key(row)] = cursor.lastrowid

            for row in category_rows:
                cursor.execute(
                    CATEGORY_RESULT_UPSERT,
                    (
                        run_ids[config_key(row)], category_ids[row["category"]],
                        int(row["num_queries"]), *metric_values(row),
                    ),
                )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            cursor.close()
    return len(run_ids)
