"""Unit tests for similarity metrics (Euclidean, Manhattan, Cosine)."""

import unittest

import numpy as np

from src.similarity import (
    calculate_distance,
    calculate_distances,
    compute_distances,
    euclidean_distance,
    euclidean_distances,
    validate_feature_vector,
)


class SimilarityTests(unittest.TestCase):
    def test_euclidean_distance_known_vectors(self) -> None:
        a = np.zeros(9)
        b = np.ones(9)
        self.assertAlmostEqual(calculate_distance(a, b, metric="euclidean"), 3.0)
        self.assertAlmostEqual(euclidean_distance(a, b), 3.0)

    def test_manhattan_distance_known_vectors(self) -> None:
        a = np.zeros(9)
        b = np.ones(9)
        self.assertAlmostEqual(calculate_distance(a, b, metric="manhattan"), 9.0)

        b[0] = 3.0
        b[1] = -4.0
        self.assertAlmostEqual(calculate_distance(a, b, metric="manhattan"), 14.0)

    def test_cosine_distance_known_vectors(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        # Orthogonal vectors -> cosine similarity 0 -> cosine distance 1.0
        self.assertAlmostEqual(calculate_distance(a, b, metric="cosine"), 1.0)

        c = np.array([2.0, 0.0, 0.0])
        # Parallel vectors -> cosine similarity 1.0 -> cosine distance 0.0
        self.assertAlmostEqual(calculate_distance(a, c, metric="cosine"), 0.0)

    def test_cosine_zero_vector_safety(self) -> None:
        a = np.zeros(9)
        b = np.ones(9)
        # Zero vector handled safely without division-by-zero, returning 1.0
        dist = calculate_distance(a, b, metric="cosine")
        self.assertEqual(dist, 1.0)

        matrix = np.vstack((np.zeros(9), np.ones(9)))
        distances = calculate_distances(a, matrix, metric="cosine")
        self.assertEqual(distances.shape, (2,))
        np.testing.assert_allclose(distances, [1.0, 1.0])

    def test_vector_dimension_validation(self) -> None:
        vec9 = np.zeros(9)
        vec18 = np.zeros(18)
        vec27 = np.zeros(27)

        np.testing.assert_array_equal(validate_feature_vector(vec9, expected_dim=9), vec9)
        np.testing.assert_array_equal(validate_feature_vector(vec18, expected_dim=18), vec18)
        np.testing.assert_array_equal(validate_feature_vector(vec27, expected_dim=27), vec27)

        with self.assertRaises(ValueError):
            validate_feature_vector(vec9, expected_dim=18)

    def test_nan_and_inf_raises_value_error(self) -> None:
        vector = np.zeros(9)
        vector[2] = np.nan
        with self.assertRaises(ValueError):
            validate_feature_vector(vector)

        vector[2] = np.inf
        with self.assertRaises(ValueError):
            validate_feature_vector(vector)

    def test_vectorized_output_shapes_and_metrics(self) -> None:
        query = np.zeros(9)
        matrix = np.vstack((np.zeros(9), np.ones(9), np.full(9, 2.0)))

        euc_dist = calculate_distances(query, matrix, metric="euclidean")
        self.assertEqual(euc_dist.shape, (3,))
        np.testing.assert_allclose(euc_dist, [0.0, 3.0, 6.0])

        man_dist = calculate_distances(query, matrix, metric="manhattan")
        self.assertEqual(man_dist.shape, (3,))
        np.testing.assert_allclose(man_dist, [0.0, 9.0, 18.0])

    def test_manhattan_distances(self) -> None:
        matrix = np.array([[3.0, 4.0], [0.0, 0.0], [-1.0, 1.0]])
        np.testing.assert_allclose(
            compute_distances([0.0, 0.0], matrix, "manhattan"), [7.0, 0.0, 2.0]
        )

    def test_cosine_distances(self) -> None:
        matrix = np.array([[2.0, 0.0], [0.0, 5.0], [-1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(
            compute_distances([1.0, 0.0], matrix, "cosine"), [0.0, 1.0, 2.0, 1.0]
        )

    def test_compute_distances_any_dimension(self) -> None:
        distances = compute_distances(np.zeros(27), np.ones((4, 27)), "euclidean")
        np.testing.assert_allclose(distances, np.full(4, np.sqrt(27)))

    def test_compute_distances_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            compute_distances(np.zeros(9), np.zeros((3, 18)))

    def test_unknown_metric_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            compute_distances(np.zeros(9), np.zeros((2, 9)), "chebyshev")

    def test_validate_any_dimension(self) -> None:
        self.assertEqual(validate_feature_vector(np.zeros(18), dimension=None).shape, (18,))
        with self.assertRaises(ValueError):
            validate_feature_vector(np.zeros((2, 9)), dimension=None)


if __name__ == "__main__":
    unittest.main()
