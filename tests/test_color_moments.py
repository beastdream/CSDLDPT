"""Unit tests for multi color space extraction, feature fusion, and BGR/RGB safety."""

import unittest

import cv2
import numpy as np

from src.color_moments import (
    convert_color_space,
    extract_color_moments,
    extract_descriptor,
)


class ColorMomentsMultiSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Synthetic solid color BGR image (100x100 pixels, pure red in BGR: B=0, G=0, R=255)
        cls.red_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        cls.red_bgr[:, :, 2] = 255

        # Synthetic gradient BGR image (100x100 pixels)
        gradient = np.linspace(0, 255, 100, dtype=np.uint8)
        grid_x, grid_y = np.meshgrid(gradient, gradient)
        cls.gradient_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        cls.gradient_bgr[:, :, 0] = grid_x
        cls.gradient_bgr[:, :, 1] = grid_y
        cls.gradient_bgr[:, :, 2] = 255 - grid_x

    def test_single_color_spaces_dimension(self) -> None:
        rgb_vec = extract_color_moments(self.gradient_bgr, color_space="RGB")
        hsv_vec = extract_color_moments(self.gradient_bgr, color_space="HSV")
        lab_vec = extract_color_moments(self.gradient_bgr, color_space="LAB")

        self.assertEqual(rgb_vec.shape, (9,))
        self.assertEqual(hsv_vec.shape, (9,))
        self.assertEqual(lab_vec.shape, (9,))

        self.assertTrue(np.all(np.isfinite(rgb_vec)))
        self.assertTrue(np.all(np.isfinite(hsv_vec)))
        self.assertTrue(np.all(np.isfinite(lab_vec)))

    def test_feature_fusion_dimensions(self) -> None:
        descriptors_expected = {
            "rgb": 9,
            "hsv": 9,
            "lab": 9,
            "rgb_hsv": 18,
            "rgb_lab": 18,
            "hsv_lab": 18,
            "rgb_hsv_lab": 27,
        }

        for desc, expected_dim in descriptors_expected.items():
            vec = extract_descriptor(self.gradient_bgr, descriptor=desc)
            self.assertEqual(
                vec.shape,
                (expected_dim,),
                f"Descriptor '{desc}' expected shape ({expected_dim},), got {vec.shape}.",
            )
            self.assertTrue(
                np.all(np.isfinite(vec)),
                f"Descriptor '{desc}' contains NaN or Inf.",
            )

    def test_no_nan_or_inf(self) -> None:
        for desc in ("rgb", "hsv", "lab", "rgb_hsv", "rgb_lab", "hsv_lab", "rgb_hsv_lab"):
            vec = extract_descriptor(self.red_bgr, descriptor=desc)
            self.assertFalse(np.isnan(vec).any(), f"NaN found in descriptor '{desc}'")
            self.assertFalse(np.isinf(vec).any(), f"Inf found in descriptor '{desc}'")

    def test_deterministic_output(self) -> None:
        vec1 = extract_descriptor(self.gradient_bgr, descriptor="rgb_hsv_lab")
        vec2 = extract_descriptor(self.gradient_bgr, descriptor="rgb_hsv_lab")
        np.testing.assert_array_equal(vec1, vec2)

    def test_bgr_to_rgb_conversion_correctness(self) -> None:
        # In BGR: red_bgr channel 0 is 0, channel 1 is 0, channel 2 is 255
        rgb_img = convert_color_space(self.red_bgr, color_space="RGB")
        # In RGB: channel 0 should be 255 (R), channel 1 should be 0 (G), channel 2 should be 0 (B)
        self.assertEqual(rgb_img[0, 0, 0], 255)
        self.assertEqual(rgb_img[0, 0, 1], 0)
        self.assertEqual(rgb_img[0, 0, 2], 0)

        vec = extract_color_moments(self.red_bgr, color_space="RGB")
        # R_mean = 255, G_mean = 0, B_mean = 0
        self.assertAlmostEqual(vec[0], 255.0)
        self.assertAlmostEqual(vec[3], 0.0)
        self.assertAlmostEqual(vec[6], 0.0)

    def test_bgr_to_hsv_and_lab_conversions(self) -> None:
        hsv_img = convert_color_space(self.red_bgr, color_space="HSV")
        self.assertEqual(hsv_img.shape, (100, 100, 3))

        lab_img = convert_color_space(self.red_bgr, color_space="LAB")
        self.assertEqual(lab_img.shape, (100, 100, 3))

    def test_default_color_space_is_rgb(self) -> None:
        default_vec = extract_color_moments(self.gradient_bgr)
        explicit_rgb = extract_color_moments(self.gradient_bgr, color_space="RGB")
        np.testing.assert_array_equal(default_vec, explicit_rgb)

    def test_unsupported_color_space_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            extract_color_moments(self.gradient_bgr, color_space="CMYK")

    def test_unsupported_descriptor_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            extract_descriptor(self.gradient_bgr, descriptor="invalid_descriptor")


if __name__ == "__main__":
    unittest.main()
