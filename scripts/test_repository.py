"""Validate the MySQL RGB/global feature repository."""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.repository import get_all_rgb_global_features


def main() -> int:
    """Check record count, vector integrity, and metadata uniqueness."""
    try:
        records = get_all_rgb_global_features()
        image_ids = [record["image_id"] for record in records]
        filepaths = [record["filepath"] for record in records]
        invalid_shapes = sum(record["features"].shape != (9,) for record in records)
        nonfinite = sum(
            int(np.count_nonzero(~np.isfinite(record["features"])))
            for record in records
        )
        passed = (
            len(records) == 1000
            and invalid_shapes == 0
            and nonfinite == 0
            and len(set(image_ids)) == 1000
            and len(set(filepaths)) == 1000
        )
    except Exception as exc:
        print(f"ERROR: Repository test failed: {exc}")
        return 1

    print("=" * 40)
    print("RGB/GLOBAL FEATURE REPOSITORY TEST")
    print("=" * 40)
    print(f"Records           : {len(records)}")
    print("Feature dimension : 9" if invalid_shapes == 0 else "Feature dimension : INVALID")
    print(f"Unique image IDs  : {len(set(image_ids))}")
    print(f"Unique filepaths  : {len(set(filepaths))}")
    print(f"NaN/Inf           : {nonfinite}")
    print(f"Status            : {'PASS' if passed else 'FAIL'}")
    print("=" * 40)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
