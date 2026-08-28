"""Smoke-test global RGB Color Moments on a few processed WANG images."""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.color_moments import extract_color_moments


DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "wang"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PREFERRED_CATEGORIES = ("africa", "beach", "dinosaurs", "flowers", "food")
FEATURE_NAMES = (
    "R_mean",
    "R_std",
    "R_skew",
    "G_mean",
    "G_std",
    "G_skew",
    "B_mean",
    "B_std",
    "B_skew",
)


def first_image(category_dir: Path) -> Path | None:
    """Return the first supported image in filename order for one category."""
    if not category_dir.is_dir():
        return None

    images = sorted(
        (
            path
            for path in category_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    return images[0] if images else None


def select_test_images(minimum: int = 5) -> list[tuple[str, Path]]:
    """Select one existing image from each of at least five distinct classes."""
    available_categories = sorted(
        (path.name for path in DATASET_DIR.iterdir() if path.is_dir()),
        key=str.casefold,
    )
    category_order = list(PREFERRED_CATEGORIES)
    category_order.extend(
        category
        for category in available_categories
        if category not in PREFERRED_CATEGORIES
    )

    selected: list[tuple[str, Path]] = []
    for category in category_order:
        image_path = first_image(DATASET_DIR / category)
        if image_path is not None:
            selected.append((category, image_path))
        if len(selected) == minimum:
            break
    return selected


def assert_valid_vector(vector: np.ndarray, image_path: Path) -> None:
    """Assert all required invariants for one Color Moments vector."""
    image_label = image_path.relative_to(PROJECT_ROOT).as_posix()
    assert vector.shape == (9,), (
        f"{image_label}: expected vector shape (9,), got {vector.shape}."
    )
    assert np.all(np.isfinite(vector)), (
        f"{image_label}: feature vector contains NaN or Inf."
    )

    for index, channel in zip((0, 3, 6), ("R", "G", "B")):
        assert 0 <= vector[index] <= 255, (
            f"{image_label}: {channel}_mean must be in [0, 255], "
            f"got {vector[index]}."
        )

    for index, channel in zip((1, 4, 7), ("R", "G", "B")):
        assert vector[index] >= 0, (
            f"{image_label}: {channel}_std must be >= 0, got {vector[index]}."
        )


def print_result(category: str, image_path: Path, vector: np.ndarray) -> None:
    """Print one test result without modifying the original feature values."""
    image_label = image_path.relative_to(PROJECT_ROOT).as_posix()
    print("=" * 50)
    print(f"Image: {image_label}")
    print(f"Category: {category}\n")
    print(f"Feature vector shape: {vector.shape}\n")

    for index, (name, value) in enumerate(zip(FEATURE_NAMES, vector)):
        print(f"{name:<7} : {value:.4f}")
        if index in (2, 5):
            print()

    formatted_vector = ", ".join(f"{value:.4f}" for value in vector)
    print(f"\nVector: [{formatted_vector}]")


def main() -> int:
    """Run Color Moments assertions on five images from distinct categories."""
    if not DATASET_DIR.is_dir():
        print(
            "Processed WANG dataset not found. "
            "Run python scripts/prepare_wang.py first."
        )
        return 1

    selected_images = select_test_images()
    if len(selected_images) < 5:
        print(
            "ERROR: Fewer than 5 processed categories contain supported images; "
            f"found {len(selected_images)}."
        )
        return 1

    for category, image_path in selected_images:
        try:
            vector = extract_color_moments(image_path)
            assert_valid_vector(vector, image_path)
            print_result(category, image_path, vector)
        except Exception as exc:
            image_label = image_path.relative_to(PROJECT_ROOT).as_posix()
            print(f"ERROR: Image {image_label} failed: {exc}")
            return 1

    print("\nAll Color Moments tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
