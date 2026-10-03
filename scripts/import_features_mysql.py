"""Import precomputed global Color Moments (RGB/HSV/LAB) from CSV into MySQL.

Usage:
    python scripts/import_features_mysql.py                    # RGB
    python scripts/import_features_mysql.py --color-space ALL  # RGB, HSV, LAB
"""

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.color_moments import COLOR_SPACE_CHANNELS, feature_names
from src.database import connection_scope
from src.repository import feature_csv_path


EXPECTED_TOTAL = 1000
EXPECTED_PER_CATEGORY = 100
FEATURE_TYPE = "GLOBAL"
METADATA_COLUMNS = ("image_id", "filename", "filepath", "category")

IMAGE_UPSERT = """
INSERT INTO images (
    image_id, filename, filepath, category_id, width, height, file_extension
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    filename = VALUES(filename),
    filepath = VALUES(filepath),
    category_id = VALUES(category_id),
    width = VALUES(width),
    height = VALUES(height),
    file_extension = VALUES(file_extension)
"""

# CSV channel columns (e.g. h_mean) map by position to generic DB columns c1..c3.
FEATURE_UPSERT = """
INSERT INTO color_moment_features (
    image_id, color_space, feature_type,
    c1_mean, c1_std, c1_skew,
    c2_mean, c2_std, c2_skew,
    c3_mean, c3_std, c3_skew
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    c1_mean = VALUES(c1_mean),
    c1_std = VALUES(c1_std),
    c1_skew = VALUES(c1_skew),
    c2_mean = VALUES(c2_mean),
    c2_std = VALUES(c2_std),
    c2_skew = VALUES(c2_skew),
    c3_mean = VALUES(c3_mean),
    c3_std = VALUES(c3_std),
    c3_skew = VALUES(c3_skew)
"""


def load_csv_rows(color_space: str) -> list[dict[str, str]]:
    """Load and validate the precomputed feature CSV without recomputation."""
    csv_path = feature_csv_path(color_space)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    feature_columns = feature_names(color_space)
    required_columns = (*METADATA_COLUMNS, *feature_columns)
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if tuple(reader.fieldnames or ()) != required_columns:
            raise ValueError(
                "CSV must contain exactly these columns in order: "
                + ", ".join(required_columns)
            )
        rows = list(reader)

    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"Expected 1000 CSV rows, found {len(rows)}.")

    filepaths = [row["filepath"] for row in rows]
    if len(filepaths) != len(set(filepaths)):
        raise ValueError("CSV contains duplicate filepath values.")

    for row_number, row in enumerate(rows, start=2):
        try:
            int(row["image_id"])
        except ValueError as exc:
            raise ValueError(f"CSV row {row_number} has a non-numeric image_id.") from exc
        for column in feature_columns:
            try:
                value = float(row[column])
            except ValueError as exc:
                raise ValueError(
                    f"CSV row {row_number}, column {column} is not numeric."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"CSV row {row_number}, column {column} contains NaN or Inf."
                )
    return rows


def read_image_size(filepath: str) -> tuple[int | None, int | None]:
    """Read width and height when the project-relative image exists."""
    image_path = (PROJECT_ROOT / filepath).resolve()
    if not image_path.is_relative_to(PROJECT_ROOT) or not image_path.is_file():
        return None, None

    try:
        encoded = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except OSError:
        return None, None
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return width, height


def fetch_categories(cursor: Any) -> dict[str, int]:
    """Return the exact category-name to ID mapping stored in MySQL."""
    cursor.execute("SELECT category_id, category_name FROM categories")
    return {name: category_id for category_id, name in cursor.fetchall()}


def validate_database(
    cursor: Any,
    color_space: str,
) -> tuple[bool, int, int, Counter[str], list[str]]:
    """Validate imported totals, duplicate paths, and per-category counts."""
    warnings: list[str] = []
    cursor.execute("SELECT COUNT(*) FROM images")
    image_count = cursor.fetchone()[0]
    cursor.execute(
        "SELECT COUNT(*) FROM color_moment_features "
        "WHERE color_space = %s AND feature_type = %s",
        (color_space, FEATURE_TYPE),
    )
    feature_count = cursor.fetchone()[0]
    cursor.execute(
        "SELECT filepath, COUNT(*) FROM images GROUP BY filepath HAVING COUNT(*) > 1"
    )
    duplicate_paths = cursor.fetchall()
    cursor.execute(
        "SELECT c.category_name, COUNT(i.image_id) "
        "FROM categories c LEFT JOIN images i ON i.category_id = c.category_id "
        "GROUP BY c.category_id, c.category_name"
    )
    category_counts = Counter(dict(cursor.fetchall()))

    if image_count != EXPECTED_TOTAL:
        warnings.append(f"Expected 1000 images, found {image_count}.")
    if feature_count != EXPECTED_TOTAL:
        warnings.append(
            f"Expected 1000 {color_space}/GLOBAL features, found {feature_count}."
        )
    if duplicate_paths:
        warnings.append(f"Found {len(duplicate_paths)} duplicate filepath values.")
    for category, count in category_counts.items():
        if count != EXPECTED_PER_CATEGORY:
            warnings.append(f"Expected 100 images for {category}, found {count}.")

    return not warnings, image_count, feature_count, category_counts, warnings


def print_summary(
    database: str,
    images: int,
    features: int,
    status: str,
    color_space: str,
) -> None:
    """Print the required MySQL import summary."""
    print("\n" + "=" * 40)
    print("MYSQL IMPORT SUMMARY")
    print("=" * 40)
    print(f"Database       : {database}")
    print(f"Images         : {images}")
    print(f"Features       : {features}")
    print(f"Color space    : {color_space}")
    print(f"Feature type   : {FEATURE_TYPE}")
    print("Feature dim    : 9")
    print(f"Status         : {status}")
    print("=" * 40)


def main() -> int:
    """Import the requested color spaces, one atomic transaction each."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--color-space",
        default="RGB",
        type=str.upper,
        choices=(*COLOR_SPACE_CHANNELS, "ALL"),
        help="Color space to import (default: RGB).",
    )
    args = parser.parse_args()
    spaces = COLOR_SPACE_CHANNELS if args.color_space == "ALL" else (args.color_space,)
    return max(import_color_space(space) for space in spaces)


def import_color_space(color_space: str) -> int:
    """Import one color space's CSV rows atomically and validate the database."""
    feature_columns = feature_names(color_space)
    try:
        rows = load_csv_rows(color_space)
        csv_categories = {row["category"] for row in rows}

        with connection_scope() as connection:
            cursor = connection.cursor()
            try:
                database_categories = fetch_categories(cursor)
                if csv_categories != set(database_categories):
                    print(f"CSV categories: {sorted(csv_categories)}")
                    print(f"Database categories: {sorted(database_categories)}")
                    print("ERROR: Category sets do not match. Import stopped before transaction.")
                    return 1

                # End the read transaction opened by the category query before
                # starting the all-or-nothing import transaction.
                connection.commit()
                connection.start_transaction()
                for index, row in enumerate(rows, start=1):
                    if index == 1 or index % 100 == 0:
                        print(f"Importing {color_space} [{index}/{len(rows)}]: {row['filepath']}")
                    width, height = read_image_size(row["filepath"])
                    extension = Path(row["filename"]).suffix.lower()
                    image_id = int(row["image_id"])
                    cursor.execute(
                        IMAGE_UPSERT,
                        (
                            image_id,
                            row["filename"],
                            row["filepath"],
                            database_categories[row["category"]],
                            width,
                            height,
                            extension,
                        ),
                    )
                    features = tuple(float(row[name]) for name in feature_columns)
                    cursor.execute(
                        FEATURE_UPSERT,
                        (image_id, color_space, FEATURE_TYPE, *features),
                    )

                valid, images, features, counts, warnings = validate_database(
                    cursor, color_space
                )
                if not valid:
                    connection.rollback()
                    for warning in warnings:
                        print(f"WARNING: {warning}")
                    print_summary(
                        connection.database, images, features,
                        "FAILED (ROLLED BACK)", color_space,
                    )
                    return 1

                connection.commit()
                print("\nImages by category:")
                for category in sorted(counts):
                    print(f"{category:<12} : {counts[category]}")
                print_summary(connection.database, images, features, "SUCCESS", color_space)
                return 0
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise
            finally:
                cursor.close()
    except Exception as exc:
        print(f"ERROR: MySQL {color_space} feature import failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
