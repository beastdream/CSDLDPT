"""End-to-end smoke tests for the unified Retrieval Engine."""

from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.repository import get_all_rgb_global_features
from src.retrieval import retrieve


def main() -> int:
    """Run mandatory smoke tests for the retrieval engine."""
    print("=" * 60)
    print("RETRIEVAL ENGINE SMOKE TEST")
    print("=" * 60)

    # 1. Fetch a sample internal query record from database
    records = get_all_rgb_global_features()
    if not records:
        print("ERROR: MySQL repository returned no features.")
        return 1

    sample_record = records[0]
    internal_query_path = sample_record["filepath"]

    # Create a temporary external query image file
    with tempfile.TemporaryDirectory() as tmp_dir:
        external_query_path = Path(tmp_dir) / "external_test_query.jpg"
        synthetic_bgr = np.full((100, 100, 3), (40, 120, 200), dtype=np.uint8)
        cv2.imwrite(str(external_query_path), synthetic_bgr)

        # Test Case 1: RGB + Euclidean + Top 5
        print("\n[1/4] Running RGB + Euclidean + Top 5...")
        res1 = retrieve(
            query_image=internal_query_path,
            descriptor="rgb",
            metric="euclidean",
            top_k=5,
            normalization="none",
        )
        assert len(res1["results"]) == 5, f"Expected 5 results, got {len(res1['results'])}"
        assert res1["query"]["image_id"] == sample_record["image_id"]
        assert all(item["image_id"] != sample_record["image_id"] for item in res1["results"])
        print("  -> PASS (Self query excluded, 5 results returned)")

        # Test Case 2: LAB + Euclidean + Z-Score + Top 10
        print("\n[2/4] Running LAB + Euclidean + Z-score + Top 10...")
        res2 = retrieve(
            query_image=internal_query_path,
            descriptor="lab",
            metric="euclidean",
            top_k=10,
            normalization="zscore",
        )
        assert len(res2["results"]) == 10, f"Expected 10 results, got {len(res2['results'])}"
        assert res2["descriptor"] == "lab"
        assert res2["normalization"] == "zscore"
        print("  -> PASS (10 results returned, Z-score normalized)")

        # Test Case 3: RGB+HSV+LAB + Euclidean + Z-Score + Top 10
        print("\n[3/4] Running RGB+HSV+LAB + Euclidean + Z-score + Top 10...")
        res3 = retrieve(
            query_image=internal_query_path,
            descriptor="rgb_hsv_lab",
            metric="euclidean",
            top_k=10,
            normalization="zscore",
        )
        assert len(res3["results"]) == 10
        assert res3["query"]["features"].shape == (27,)
        assert res3["descriptor"] == "rgb_hsv_lab"
        print("  -> PASS (27D features, Z-score normalized)")

        # Test Case 4: External Query Image
        print("\n[4/4] Running External Query Image (RGB_HSV_LAB + Cosine + Top 10)...")
        res4 = retrieve(
            query_image=external_query_path,
            descriptor="rgb_hsv_lab",
            metric="cosine",
            top_k=10,
            normalization="zscore",
        )
        assert len(res4["results"]) == 10
        assert res4["query"]["image_id"] is None
        assert res4["query"]["category"] is None
        print("  -> PASS (External query processed, 10 candidates returned)")

    print("\n" + "=" * 60)
    print("ALL RETRIEVAL ENGINE SMOKE TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
