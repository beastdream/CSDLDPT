"""Unit tests for multi color space Color Moments extraction."""

import unittest

import numpy as np

from src.color_moments import (
    compute_color_moments,
    convert_color_space,
    feature_names,
    normalize_color_space,
)


class ColorSpaceTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

    def test_rgb_is_unchanged(self) -> None:
        self.assertIs(convert_color_space(self.image, "rgb"), self.image)

    def test_hsv_ranges(self) -> None:
        hsv = convert_color_space(self.image, "HSV")
        self.assertEqual(hsv.shape, self.image.shape)
        self.assertTrue(np.all((hsv[..., 0] >= 0) & (hsv[..., 0] < 360)))
        self.assertTrue(np.all((hsv[..., 1:] >= 0) & (hsv[..., 1:] <= 1)))

    def test_lab_ranges(self) -> None:
        lab = convert_color_space(self.image, "LAB")
        self.assertTrue(np.all((lab[..., 0] >= 0) & (lab[..., 0] <= 100)))

    def test_pure_red_hsv_and_lab(self) -> None:
        red = np.zeros((2, 2, 3), dtype=np.uint8)
        red[..., 0] = 255
        hsv = compute_color_moments(convert_color_space(red, "HSV"))
        np.testing.assert_allclose(hsv[[0, 3, 6]], [0.0, 1.0, 1.0], atol=1e-5)
        lab = compute_color_moments(convert_color_space(red, "LAB"))
        np.testing.assert_allclose(lab[[0, 3, 6]], [53.24, 80.09, 67.20], atol=0.1)

    def test_feature_names(self) -> None:
        self.assertEqual(feature_names("hsv")[:3], ("h_mean", "h_std", "h_skew"))
        self.assertEqual(len(feature_names("LAB")), 9)

    def test_unknown_color_space(self) -> None:
        with self.assertRaises(ValueError):
            normalize_color_space("YCbCr")


if __name__ == "__main__":
    unittest.main()
