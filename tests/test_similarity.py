"""Unit tests for Color Moments distance functions."""

import unittest

import numpy as np

from src.similarity import (
    compute_distances,
    euclidean_distance,
    euclidean_distances,
    validate_feature_vector,
)


class SimilarityTests(unittest.TestCase):
    def test_euclidean_distance_known_vectors(self) -> None:
        a = np.zeros(9)
        b = np.ones(9)
        self.assertAlmostEqual(euclidean_distance(a, b), 3.0)

    def test_identical_vectors_have_zero_distance(self) -> None:
        vector = np.arange(9, dtype=np.float64)
        self.assertEqual(euclidean_distance(vector, vector), 0.0)

    def test_pythagorean_distance(self) -> None:
        a = np.zeros(9)
        b = np.zeros(9)
        b[0] = 3.0
        b[1] = 4.0
        self.assertAlmostEqual(euclidean_distance(a, b), 5.0)

    def test_wrong_dimension_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            validate_feature_vector(np.zeros(8))

    def test_nan_raises_value_error(self) -> None:
        vector = np.zeros(9)
        vector[2] = np.nan
        with self.assertRaises(ValueError):
            validate_feature_vector(vector)

    def test_inf_raises_value_error(self) -> None:
        vector = np.zeros(9)
        vector[2] = np.inf
        with self.assertRaises(ValueError):
            validate_feature_vector(vector)

    def test_vectorized_output_shape(self) -> None:
        query = np.zeros(9)
        matrix = np.vstack((np.zeros(9), np.ones(9), np.full(9, 2.0)))
        distances = euclidean_distances(query, matrix)
        self.assertEqual(distances.shape, (3,))
        np.testing.assert_allclose(distances, [0.0, 3.0, 6.0])

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
