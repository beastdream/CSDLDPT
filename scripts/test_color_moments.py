"""Smoke-test multi color space and fusion descriptors on processed WANG images."""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.color_moments import extract_color_moments, extract_descriptor
from src.config import DESCRIPTOR_DIMENSIONS


DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "wang"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PREFERRED_CATEGORIES = ("africa", "beach", "dinosaurs", "flowers", "food")


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


def assert_valid_descriptor(
    vector: np.ndarray, descriptor: str, image_path: Path
) -> None:
    """Assert invariants for a Color Moments descriptor vector."""
    expected_dim = DESCRIPTOR_DIMENSIONS[descriptor]
    image_label = image_path.relative_to(PROJECT_ROOT).as_posix()
    assert vector.shape == (expected_dim,), (
        f"{image_label} ({descriptor}): expected shape ({expected_dim},), got {vector.shape}."
    )
    assert np.all(np.isfinite(vector)), (
        f"{image_label} ({descriptor}): vector contains NaN or Inf."
    )


def main() -> int:
    """Run multi color space assertions on five images from distinct categories."""
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

    print("=" * 60)
    print("MULTI COLOR SPACE & FEATURE FUSION SMOKE TEST")
    print("=" * 60)

    # 1. Single color space extraction test
    for category, image_path in selected_images:
        for cs in ("RGB", "HSV", "LAB"):
            try:
                vec = extract_color_moments(image_path, color_space=cs)
                assert vec.shape == (9,)
                assert np.all(np.isfinite(vec))
            except Exception as exc:
                image_label = image_path.relative_to(PROJECT_ROOT).as_posix()
                print(f"ERROR: Image {image_label} color_space={cs} failed: {exc}")
                return 1

    print("Single color space extractions (RGB, HSV, LAB 9D): PASS")

    # 2. Descriptor fusion extractions test
    for desc, expected_dim in DESCRIPTOR_DIMENSIONS.items():
        for category, image_path in selected_images:
            try:
                vec = extract_descriptor(image_path, descriptor=desc)
                assert_valid_descriptor(vec, desc, image_path)
            except Exception as exc:
                image_label = image_path.relative_to(PROJECT_ROOT).as_posix()
                print(f"ERROR: Image {image_label} descriptor={desc} failed: {exc}")
                return 1
        print(f"Descriptor '{desc}' ({expected_dim}D): PASS")

    print("\nAll Multi Color Space & Fusion tests passed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
