"""Unit tests for 9D Euclidean similarity functions."""

import unittest

import numpy as np

from src.similarity import (
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


if __name__ == "__main__":
    unittest.main()
