"""Prepare the WANG/Corel-1K dataset without modifying source images."""

from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wang"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "wang"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_PER_CATEGORY = 100
EXPECTED_TOTAL = 1000

RAW_TO_PROCESSED = {
    "africans": "africa",
    "beaches": "beach",
    "buildings": "buildings",
    "buses": "buses",
    "dinosaurs": "dinosaurs",
    "elephants": "elephants",
    "flowers": "flowers",
    "food": "food",
    "horses": "horses",
    "mountains": "mountains",
}


def image_files(directory: Path) -> list[Path]:
    """Return supported image files directly inside a directory."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def copy_dataset() -> None:
    """Copy images from each raw class folder to its normalized category."""
    for raw_folder, category in RAW_TO_PROCESSED.items():
        source_dir = RAW_DIR / raw_folder
        if not source_dir.is_dir():
            print(f"WARNING: Missing raw folder: {source_dir}")
            continue

        destination_dir = PROCESSED_DIR / category
        destination_dir.mkdir(parents=True, exist_ok=True)

        for source_path in image_files(source_dir):
            shutil.copy2(source_path, destination_dir / source_path.name)


def validate_processed() -> bool:
    """Print processed image counts and return whether validation succeeded."""
    counts: dict[str, int] = {}

    print("\nProcessed dataset summary:")
    for category in RAW_TO_PROCESSED.values():
        count = len(image_files(PROCESSED_DIR / category))
        counts[category] = count
        print(f"{category:<12} : {count}")
        if count != EXPECTED_PER_CATEGORY:
            print(
                f"WARNING: Expected {EXPECTED_PER_CATEGORY} images in "
                f"{category}, but found {count}."
            )

    total = sum(counts.values())
    print(f"{'Total':<12} : {total}")

    if total != EXPECTED_TOTAL:
        print(f"WARNING: Expected 1000 images but found {total}.")

    valid = (
        total == EXPECTED_TOTAL
        and all(count == EXPECTED_PER_CATEGORY for count in counts.values())
    )
    if valid:
        print("Dataset preparation completed successfully.")
    else:
        print("Dataset preparation incomplete. Success was not reported.")
    return valid


def main() -> None:
    if not RAW_DIR.is_dir():
        print(
            "Dataset not found. Please place WANG/Corel-1K images in "
            "data/raw/wang/"
        )
        return

    copy_dataset()
    validate_processed()


if __name__ == "__main__":
    main()
