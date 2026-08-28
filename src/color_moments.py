"""Compute global RGB Color Moments for a single image."""

from os import PathLike
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray


ImagePath: TypeAlias = str | PathLike[str]
MomentVector: TypeAlias = NDArray[np.float64]


def load_image_rgb(image_path: ImagePath) -> NDArray[np.uint8]:
    """Read an image with OpenCV and return it as an RGB NumPy array.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        ValueError: If OpenCV cannot decode the image.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    # cv2.imread can fail on Unicode Windows paths. Reading the encoded bytes
    # with NumPy and decoding them with OpenCV preserves the same BGR result.
    try:
        encoded_image = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Could not read image file: {path}") from exc

    image_bgr = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def compute_channel_moments(channel: NDArray[np.generic]) -> tuple[float, float, float]:
    """Return mean, standard deviation, and signed third moment of a 2D channel.

    The third Color Moment is the cube root of the third central moment.
    """
    channel_array = np.asarray(channel)
    if channel_array.ndim != 2:
        raise ValueError(
            f"Channel must be a non-empty 2D array; received shape "
            f"{channel_array.shape}."
        )
    if channel_array.size == 0:
        raise ValueError("Channel must be a non-empty 2D array.")

    try:
        channel_float = channel_array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Channel must contain numeric pixel values.") from exc

    if not np.all(np.isfinite(channel_float)):
        raise ValueError("Channel contains NaN or infinite pixel values.")

    mean = np.mean(channel_float)
    centered = channel_float - mean
    std = np.sqrt(np.mean(centered**2))
    third_moment = np.cbrt(np.mean(centered**3))

    return float(mean), float(std), float(third_moment)


def compute_color_moments(image_rgb: NDArray[np.generic]) -> MomentVector:
    """Compute a global 9D Color Moments vector from an RGB image.

    Feature order:
        R_mean, R_std, R_skew,
        G_mean, G_std, G_skew,
        B_mean, B_std, B_skew.
    """
    if image_rgb is None:
        raise ValueError("RGB image must not be None.")

    image_array = np.asarray(image_rgb)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(
            "RGB image must have shape H x W x 3; "
            f"received shape {image_array.shape}."
        )
    if image_array.shape[0] == 0 or image_array.shape[1] == 0:
        raise ValueError("RGB image must have non-zero height and width.")

    features = [
        moment
        for channel_index in range(3)  # RGB order
        for moment in compute_channel_moments(image_array[:, :, channel_index])
    ]
    vector = np.asarray(features, dtype=np.float64)

    if vector.shape != (9,):
        raise ValueError(
            f"Color Moments output must have shape (9,), received {vector.shape}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("Color Moments output contains NaN or infinite values.")

    return vector


def extract_color_moments(image_path: ImagePath) -> MomentVector:
    """Load an image as RGB and return its global 9D Color Moments vector."""
    return compute_color_moments(load_image_rgb(image_path))
