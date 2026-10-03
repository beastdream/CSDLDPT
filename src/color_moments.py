"""Compute global Color Moments for a single image in RGB, HSV, or LAB."""

from os import PathLike
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray


ImagePath: TypeAlias = str | PathLike[str]
MomentVector: TypeAlias = NDArray[np.float64]

# Channel names per supported color space, in feature order. RGB keeps the raw
# 0-255 scale so the stored baseline stays unchanged. HSV and LAB use OpenCV's
# float32 conversion: H in [0, 360), S/V in [0, 1], L in [0, 100], a/b ~[-127, 127].
COLOR_SPACE_CHANNELS: dict[str, tuple[str, str, str]] = {
    "RGB": ("r", "g", "b"),
    "HSV": ("h", "s", "v"),
    "LAB": ("l", "a", "b"),
}
MOMENT_NAMES = ("mean", "std", "skew")


def normalize_color_space(color_space: str) -> str:
    """Return the canonical upper-case color space name or raise ValueError."""
    name = str(color_space).strip().upper()
    if name not in COLOR_SPACE_CHANNELS:
        raise ValueError(
            f"Unsupported color space {color_space!r}; expected one of "
            f"{', '.join(COLOR_SPACE_CHANNELS)}."
        )
    return name


def feature_names(color_space: str) -> tuple[str, ...]:
    """Return the 9 feature column names, e.g. ``h_mean, h_std, ..., v_skew``."""
    channels = COLOR_SPACE_CHANNELS[normalize_color_space(color_space)]
    return tuple(f"{channel}_{moment}" for channel in channels for moment in MOMENT_NAMES)


def convert_color_space(image_rgb: NDArray[np.generic], color_space: str) -> NDArray[np.generic]:
    """Convert an 8-bit RGB image to the requested color space."""
    name = normalize_color_space(color_space)
    if name == "RGB":
        return image_rgb
    image_float = np.asarray(image_rgb, dtype=np.float32) / 255.0
    code = cv2.COLOR_RGB2HSV if name == "HSV" else cv2.COLOR_RGB2LAB
    return cv2.cvtColor(image_float, code)


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
    """Compute a global 9D Color Moments vector from a 3-channel image.

    Feature order (channels as given, e.g. RGB):
        C1_mean, C1_std, C1_skew,
        C2_mean, C2_std, C2_skew,
        C3_mean, C3_std, C3_skew.
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


def extract_color_moments(image_path: ImagePath, color_space: str = "RGB") -> MomentVector:
    """Load an image and return its global 9D Color Moments in ``color_space``."""
    return compute_color_moments(convert_color_space(load_image_rgb(image_path), color_space))
