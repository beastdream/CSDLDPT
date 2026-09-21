"""Unit tests for corpus-wide Z-Score normalization."""

import tempfile

import numpy as np
import unittest

from src.normalization import (
    ZScoreNormalizer,
    fit_normalizer,
    load_normalizer,
    save_normalizer,
    transform,
)


class NormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        # Synthetic corpus matrix: 100 samples, 27 dimensions
        self.corpus = np.random.normal(loc=50.0, scale=15.0, size=(100, 27))
        # Add a constant feature column (std = 0) at index 5
        self.corpus[:, 5] = 100.0

    def test_fit_normalizer_and_safe_std(self) -> None:
        params = fit_normalizer(self.corpus)
        mean = params["mean"]
        std = params["std"]

        self.assertEqual(mean.shape, (27,))
        self.assertEqual(std.shape, (27,))

        # Constant feature column std should be 1.0 (safe replacement)
        self.assertEqual(std[5], 1.0)
        self.assertTrue(np.all(std > 0))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.isfinite(std)))

    def test_transform_matrix_and_vector(self) -> None:
        params = fit_normalizer(self.corpus)
        normalized_matrix = transform(self.corpus, params["mean"], params["std"])

        self.assertEqual(normalized_matrix.shape, (100, 27))
        self.assertTrue(np.all(np.isfinite(normalized_matrix)))

        # Standardized non-constant columns should have mean ~ 0 and std ~ 1
        non_constant_indices = [i for i in range(27) if i != 5]
        means_after = np.mean(normalized_matrix[:, non_constant_indices], axis=0)
        stds_after = np.std(normalized_matrix[:, non_constant_indices], axis=0)

        np.testing.assert_allclose(means_after, 0.0, atol=1e-12)
        np.testing.assert_allclose(stds_after, 1.0, atol=1e-12)

        # Single vector transform
        single_vec = self.corpus[0]
        norm_vec = transform(single_vec, params["mean"], params["std"])
        self.assertEqual(norm_vec.shape, (27,))
        np.testing.assert_allclose(norm_vec, normalized_matrix[0])

    def test_zero_std_no_division_by_zero(self) -> None:
        params = fit_normalizer(self.corpus)
        normalized_matrix = transform(self.corpus, params["mean"], params["std"])
        # Column 5 (constant 100.0) should be normalized as (100.0 - 100.0) / 1.0 = 0.0
        np.testing.assert_allclose(normalized_matrix[:, 5], 0.0)
        self.assertFalse(np.isnan(normalized_matrix).any())
        self.assertFalse(np.isinf(normalized_matrix).any())

    def test_save_and_load_normalizer(self) -> None:
        params = fit_normalizer(self.corpus)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        save_normalizer(params, tmp_path)
        loaded_params = load_normalizer(tmp_path)

        np.testing.assert_array_equal(params["mean"], loaded_params["mean"])
        np.testing.assert_array_equal(params["std"], loaded_params["std"])

        transformed_orig = transform(self.corpus, params["mean"], params["std"])
        transformed_loaded = transform(
            self.corpus, loaded_params["mean"], loaded_params["std"]
        )
        np.testing.assert_array_equal(transformed_orig, transformed_loaded)

    def test_zscore_normalizer_class(self) -> None:
        normalizer = ZScoreNormalizer()
        normalizer.fit(self.corpus)

        norm_matrix = normalizer.transform(self.corpus)
        self.assertEqual(norm_matrix.shape, (100, 27))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        normalizer.save(tmp_path)
        loaded_normalizer = ZScoreNormalizer().load(tmp_path)
        norm_matrix_loaded = loaded_normalizer.transform(self.corpus)

        np.testing.assert_array_equal(norm_matrix, norm_matrix_loaded)


if __name__ == "__main__":
    unittest.main()
