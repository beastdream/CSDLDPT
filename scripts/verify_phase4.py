"""End-to-end verification for the Phase 4 retrieval baseline."""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.repository import get_all_rgb_global_features
from src.retrieval import retrieve_similar_images


CHECK_NAMES = (
    "MySQL descriptors",
    "Feature dimension",
    "Finite features",
    "Query extraction",
    "Euclidean retrieval",
    "Self-query exclusion",
    "Top-K",
    "Ranking order",
    "Result file paths",
)


def resolve_database_path(filepath: str) -> Path:
    """Resolve a stored absolute path or project-relative filepath."""
    path = Path(filepath)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main() -> int:
    """Run all mandatory Phase 4 checks and print a single status report."""
    checks = {name: False for name in CHECK_NAMES}
    error: Exception | None = None

    try:
        records = get_all_rgb_global_features()
        checks["MySQL descriptors"] = len(records) == 1000
        checks["Feature dimension"] = all(
            record["features"].shape == (9,) for record in records
        )
        checks["Finite features"] = all(
            np.all(np.isfinite(record["features"])) for record in records
        )

        if not all(checks[name] for name in CHECK_NAMES[:3]):
            raise ValueError("MySQL descriptor validation failed.")

        query_record = records[0]
        retrieval = retrieve_similar_images(query_record["filepath"], top_k=5)
        query = retrieval["query"]
        results = retrieval["results"]

        checks["Query extraction"] = (
            query["features"].shape == (9,)
            and np.all(np.isfinite(query["features"]))
        )
        distances = np.asarray([item["distance"] for item in results], dtype=np.float64)
        checks["Euclidean retrieval"] = (
            distances.shape == (5,)
            and np.all(np.isfinite(distances))
            and np.all(distances >= 0)
        )
        checks["Self-query exclusion"] = all(
            item["image_id"] != query_record["image_id"] for item in results
        )
        result_ids = [item["image_id"] for item in results]
        checks["Top-K"] = len(results) == 5 and len(set(result_ids)) == 5
        checks["Ranking order"] = results == sorted(
            results, key=lambda item: (item["distance"], item["image_id"])
        )
        checks["Result file paths"] = all(
            resolve_database_path(item["filepath"]).is_file() for item in results
        )
    except Exception as exc:
        error = exc

    complete = all(checks.values())
    print("=" * 60)
    print("PHASE 4 VERIFICATION")
    print("=" * 60)
    for name in CHECK_NAMES:
        print(f"{name:<25}: {'PASS' if checks[name] else 'FAIL'}")
    if error is not None:
        print(f"\nError: {error}")
    print(f"\nPHASE 4 STATUS          : {'COMPLETE' if complete else 'INCOMPLETE'}")
    print("=" * 60)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
