"""Extract global RGB Color Moments from the processed WANG dataset."""

from collections import Counter
import csv
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.color_moments import extract_color_moments


DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "wang"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "color_moments_rgb.csv"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_TOTAL = 1000
EXPECTED_PER_CATEGORY = 100
CATEGORIES = (
    "africa",
    "beach",
    "buildings",
    "buses",
    "dinosaurs",
    "elephants",
    "flowers",
    "horses",
    "mountains",
    "food",
)
FEATURE_COLUMNS = (
    "r_mean",
    "r_std",
    "r_skew",
    "g_mean",
    "g_std",
    "g_skew",
    "b_mean",
    "b_std",
    "b_skew",
)
CSV_COLUMNS = (
    "image_id",
    "filename",
    "filepath",
    "category",
    *FEATURE_COLUMNS,
)


def filename_sort_key(path: Path) -> tuple[int, int | str, str]:
    """Sort numeric stems numerically, followed by other names alphabetically."""
    try:
        return 0, int(path.stem), path.name.casefold()
    except ValueError:
        return 1, path.name.casefold(), path.name


def discover_images() -> list[tuple[str, Path]]:
    """Find supported images in a deterministic category and filename order."""
    images: list[tuple[str, Path]] = []
    for category in CATEGORIES:
        category_dir = DATASET_DIR / category
        if not category_dir.is_dir():
            print(f"WARNING: Missing category directory: {relative_path(category_dir)}")
            continue

        category_images = sorted(
            (
                path
                for path in category_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=filename_sort_key,
        )
        images.extend((category, path) for path in category_images)
    return images


def relative_path(path: Path) -> str:
    """Return a project-relative path with portable forward slashes."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def image_id_from_filename(path: Path) -> int | str:
    """Return the numeric filename stem, or an empty value when non-numeric."""
    try:
        return int(path.stem)
    except ValueError:
        return ""


def extract_rows(
    images: list[tuple[str, Path]],
) -> tuple[list[dict[str, object]], list[tuple[str, str]]]:
    """Extract one validated 9D vector per image while collecting failures."""
    rows: list[dict[str, object]] = []
    failed: list[tuple[str, str]] = []
    total = len(images)

    for index, (category, image_path) in enumerate(images, start=1):
        filepath = relative_path(image_path)
        print(f"Processing [{index}/{total}]: {filepath}")
        try:
            vector = extract_color_moments(image_path)
            if vector.shape != (9,):
                raise ValueError(f"Expected feature shape (9,), got {vector.shape}.")
            if not np.all(np.isfinite(vector)):
                raise ValueError("Feature vector contains NaN or Inf.")

            row: dict[str, object] = {
                "image_id": image_id_from_filename(image_path),
                "filename": image_path.name,
                "filepath": filepath,
                "category": category,
            }
            row.update(zip(FEATURE_COLUMNS, (float(value) for value in vector)))
            rows.append(row)
        except Exception as exc:
            error = str(exc)
            failed.append((filepath, error))
            print(f"WARNING: Failed to extract {filepath}: {error}")

    return rows, failed


def write_csv(rows: list[dict[str, object]]) -> None:
    """Write extracted rows with the exact required 13-column schema."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def validate_csv() -> tuple[bool, Counter[str], list[str]]:
    """Validate the saved CSV schema, counts, values, and filepath uniqueness."""
    errors: list[str] = []
    category_counts: Counter[str] = Counter()

    with OUTPUT_PATH.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            errors.append(
                f"CSV columns do not match the required 13-column schema: "
                f"{reader.fieldnames}."
            )
        rows = list(reader)

    if len(rows) != EXPECTED_TOTAL:
        errors.append(f"Expected {EXPECTED_TOTAL} CSV rows, found {len(rows)}.")

    filepaths = [row.get("filepath", "") for row in rows]
    if len(filepaths) != len(set(filepaths)):
        errors.append("CSV contains duplicate filepath values.")

    for row_number, row in enumerate(rows, start=2):
        category = row.get("category", "")
        category_counts[category] += 1
        for column in FEATURE_COLUMNS:
            try:
                value = float(row.get(column, ""))
            except (TypeError, ValueError):
                errors.append(f"Row {row_number}, {column} is not numeric.")
                continue
            if not np.isfinite(value):
                errors.append(f"Row {row_number}, {column} contains NaN or Inf.")

    for category in CATEGORIES:
        count = category_counts[category]
        if count != EXPECTED_PER_CATEGORY:
            errors.append(
                f"Expected {EXPECTED_PER_CATEGORY} rows for {category}, found {count}."
            )

    unexpected = sorted(set(category_counts) - set(CATEGORIES))
    if unexpected:
        errors.append(f"CSV contains unexpected categories: {', '.join(unexpected)}.")

    return not errors, category_counts, errors


def print_summary(
    successful: int,
    failed: list[tuple[str, str]],
    category_counts: Counter[str],
) -> None:
    """Print the required extraction summary."""
    print("\n" + "=" * 40)
    print("RGB COLOR MOMENTS EXTRACTION SUMMARY\n")
    print("Dataset           : WANG/Corel-1K")
    print(f"Total expected    : {EXPECTED_TOTAL}")
    print(f"Successful        : {successful}")
    print(f"Failed            : {len(failed)}")
    print("Feature dimension : 9")
    print("Color space       : RGB")
    print(f"Output            : {relative_path(OUTPUT_PATH)}")
    print("\nImages by category:")
    for category in CATEGORIES:
        print(f"{category:<12} : {category_counts[category]}")
    print(
        "\nFeature order: R_mean, R_std, R_skew, G_mean, G_std, G_skew, "
        "B_mean, B_std, B_skew"
    )


def main() -> int:
    """Extract, save, and validate raw RGB Color Moments for WANG/Corel-1K."""
    if not DATASET_DIR.is_dir():
        print(f"ERROR: Processed WANG dataset not found: {relative_path(DATASET_DIR)}")
        return 1

    images = discover_images()
    rows, failed = extract_rows(images)
    write_csv(rows)
    csv_valid, category_counts, validation_errors = validate_csv()
    print_summary(len(rows), failed, category_counts)

    if len(rows) != EXPECTED_TOTAL:
        print(
            f"\nWARNING: Expected 1000 images but successfully extracted "
            f"{len(rows)} images."
        )

    if failed:
        print("\nFailed images:")
        for filepath, error in failed:
            print(f"- {filepath}: {error}")
    elif len(rows) != EXPECTED_TOTAL:
        print("\nFailed images: none recorded; the dataset may be incomplete or contain unsupported files.")

    if validation_errors:
        print("\nValidation warnings:")
        for error in validation_errors:
            print(f"WARNING: {error}")

    extraction_valid = (
        len(rows) == EXPECTED_TOTAL
        and not failed
        and csv_valid
    )
    if extraction_valid:
        print("\nRGB Color Moments extraction completed successfully.")
        return 0

    print("\nRGB Color Moments extraction did not pass validation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
