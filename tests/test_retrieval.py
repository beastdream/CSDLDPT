"""MySQL integration tests for baseline image retrieval."""

import unittest

import numpy as np

from src.repository import get_all_rgb_global_features
from src.retrieval import retrieve_similar_images


class RetrievalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        records = get_all_rgb_global_features()
        if not records:
            raise AssertionError("Repository returned no query candidate.")
        cls.query_record = records[0]
        cls.retrieval = retrieve_similar_images(cls.query_record["filepath"], top_k=5)

    def test_top_k_and_query_vector(self) -> None:
        self.assertEqual(len(self.retrieval["results"]), 5)
        self.assertEqual(self.retrieval["query"]["features"].shape, (9,))

    def test_self_query_is_excluded(self) -> None:
        result_ids = [item["image_id"] for item in self.retrieval["results"]]
        self.assertNotIn(self.query_record["image_id"], result_ids)

    def test_result_ids_are_unique(self) -> None:
        result_ids = [item["image_id"] for item in self.retrieval["results"]]
        self.assertEqual(len(result_ids), len(set(result_ids)))

    def test_distances_are_valid_and_sorted(self) -> None:
        results = self.retrieval["results"]
        distances = np.asarray([item["distance"] for item in results])
        self.assertTrue(np.all(np.isfinite(distances)))
        self.assertTrue(np.all(distances >= 0))
        ordered = sorted(results, key=lambda item: (item["distance"], item["image_id"]))
        self.assertEqual(results, ordered)

    def test_result_schema(self) -> None:
        required = {"rank", "image_id", "filename", "filepath", "category", "distance"}
        for result in self.retrieval["results"]:
            self.assertTrue(required.issubset(result))

    def test_zero_top_k_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            retrieve_similar_images(self.query_record["filepath"], top_k=0)


if __name__ == "__main__":
    unittest.main()
