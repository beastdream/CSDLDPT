"""Unit tests for feature combination, normalization and leave-one-out evaluation."""

import unittest

import numpy as np

from src.experiment import (
    build_feature_matrix,
    evaluate_configuration,
    method_name,
    normalize_features,
    summarize,
    top_k_confusion,
)


def make_records(features, categories):
    return [
        {
            "image_id": index,
            "filename": f"{index}.jpg",
            "filepath": f"data/{index}.jpg",
            "category": category,
            "features": np.asarray(vector, dtype=np.float64),
        }
        for index, (vector, category) in enumerate(zip(features, categories))
    ]


class ExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        # Two well separated clusters of three images each.
        self.categories = ["a", "a", "a", "b", "b", "b"]
        self.rgb = make_records(
            [[0] * 9, [1] * 9, [2] * 9, [50] * 9, [51] * 9, [52] * 9], self.categories
        )
        self.hsv = make_records([[i] * 9 for i in range(6)], self.categories)

    def test_method_name(self) -> None:
        self.assertEqual(method_name(["rgb", "hsv", "lab"]), "RGB+HSV+LAB")

    def test_build_feature_matrix_concatenates(self) -> None:
        matrix, records = build_feature_matrix({"RGB": self.rgb, "HSV": self.hsv}, ["RGB", "HSV"])
        self.assertEqual(matrix.shape, (6, 18))
        np.testing.assert_allclose(matrix[4, 9:], np.full(9, 4.0))
        self.assertEqual(len(records), 6)

    def test_build_feature_matrix_rejects_misaligned_images(self) -> None:
        with self.assertRaises(ValueError):
            build_feature_matrix({"RGB": self.rgb, "HSV": self.hsv[::-1]}, ["RGB", "HSV"])

    def test_zscore_normalization(self) -> None:
        matrix = np.array([[1.0, 5.0], [3.0, 5.0]])
        normalized = normalize_features(matrix, "zscore")
        np.testing.assert_allclose(normalized, [[-1.0, 0.0], [1.0, 0.0]])
        self.assertIs(normalize_features(matrix, "none"), matrix)
        with self.assertRaises(ValueError):
            normalize_features(matrix, "minmax")

    def test_perfect_clusters_give_perfect_scores(self) -> None:
        matrix, records = build_feature_matrix({"RGB": self.rgb}, ["RGB"])
        queries = evaluate_configuration(matrix, records, "euclidean", ks=(1, 2))
        self.assertEqual(len(queries), 6)
        np.testing.assert_allclose(queries["precision_at_2"], 1.0)
        np.testing.assert_allclose(queries["recall_at_2"], 1.0)
        np.testing.assert_allclose(queries["average_precision"], 1.0)

        overall, by_category = summarize(queries)
        self.assertAlmostEqual(overall["map"], 1.0)
        self.assertEqual(overall["num_queries"], 6)
        self.assertEqual(list(by_category["num_queries"]), [3, 3])

    def test_top_k_confusion_rows_sum_to_100(self) -> None:
        matrix, records = build_feature_matrix({"RGB": self.rgb}, ["RGB"])
        confusion = top_k_confusion(matrix, records, "euclidean", k=2)
        np.testing.assert_allclose(confusion.sum(axis=1), 100.0)
        self.assertAlmostEqual(confusion.loc["a", "a"], 100.0)


if __name__ == "__main__":
    unittest.main()
