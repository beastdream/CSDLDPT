"""Unit tests for the unified retrieval engine."""

import tempfile
import unittest

import cv2
import numpy as np

from src.repository import get_all_rgb_global_features
from src.retrieval import retrieve, retrieve_similar_images


class RetrievalEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        records = get_all_rgb_global_features()
        if not records:
            raise AssertionError("Repository returned no query candidate.")
        cls.query_record = records[0]

        # Create a temporary synthetic external BGR image file
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.external_img_path = f"{cls.temp_dir.name}/external_query.jpg"
        synthetic_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cv2.imwrite(cls.external_img_path, synthetic_img)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_internal_query_exclusion(self) -> None:
        res = retrieve(
            query_image=self.query_record["filepath"],
            descriptor="rgb",
            metric="euclidean",
            top_k=5,
        )
        self.assertEqual(len(res["results"]), 5)
        self.assertEqual(res["query"]["image_id"], self.query_record["image_id"])

        result_ids = [item["image_id"] for item in res["results"]]
        self.assertNotIn(self.query_record["image_id"], result_ids)

    def test_external_query_mode(self) -> None:
        res = retrieve(
            query_image=self.external_img_path,
            descriptor="rgb_hsv_lab",
            metric="euclidean",
            top_k=10,
            normalization="zscore",
        )
        self.assertEqual(len(res["results"]), 10)
        self.assertIsNone(res["query"]["image_id"])
        self.assertIsNone(res["query"]["category"])
        self.assertEqual(res["query"]["features"].shape, (27,))

    def test_all_distance_metrics(self) -> None:
        for metric in ("euclidean", "manhattan", "cosine"):
            res = retrieve(
                query_image=self.query_record["filepath"],
                descriptor="rgb",
                metric=metric,
                top_k=5,
            )
            self.assertEqual(len(res["results"]), 5)
            self.assertEqual(res["metric"], metric)
            distances = [item["distance"] for item in res["results"]]
            self.assertTrue(all(np.isfinite(d) for d in distances))
            # Results sorted by distance ascending
            self.assertEqual(
                res["results"],
                sorted(res["results"], key=lambda x: (x["distance"], x["image_id"])),
            )

    def test_normalization_integration(self) -> None:
        res_none = retrieve(
            query_image=self.query_record["filepath"],
            descriptor="rgb_hsv_lab",
            normalization="none",
            top_k=5,
        )
        res_zscore = retrieve(
            query_image=self.query_record["filepath"],
            descriptor="rgb_hsv_lab",
            normalization="zscore",
            top_k=5,
        )

        self.assertEqual(len(res_none["results"]), 5)
        self.assertEqual(len(res_zscore["results"]), 5)
        self.assertEqual(res_none["normalization"], "none")
        self.assertEqual(res_zscore["normalization"], "zscore")

    def test_invalid_parameters_raise_exceptions(self) -> None:
        # Invalid file
        with self.assertRaises(FileNotFoundError):
            retrieve("non_existent_image_12345.jpg")

        # Invalid top_k
        with self.assertRaises(ValueError):
            retrieve(self.query_record["filepath"], top_k=0)

        with self.assertRaises(ValueError):
            retrieve(self.query_record["filepath"], top_k=-5)

        # Invalid descriptor
        with self.assertRaises(ValueError):
            retrieve(self.query_record["filepath"], descriptor="invalid_descriptor")

        # Invalid metric
        with self.assertRaises(ValueError):
            retrieve(self.query_record["filepath"], metric="invalid_metric")

        # Invalid normalization
        with self.assertRaises(ValueError):
            retrieve(self.query_record["filepath"], normalization="invalid_norm")

    def test_baseline_compatibility_function(self) -> None:
        res = retrieve_similar_images(self.query_record["filepath"], top_k=5)
        self.assertEqual(len(res["results"]), 5)
        self.assertEqual(res["query"]["features"].shape, (9,))
        self.assertNotIn(
            self.query_record["image_id"],
            [item["image_id"] for item in res["results"]],
        )


if __name__ == "__main__":
    unittest.main()
