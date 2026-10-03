"""Compute Color Moments feature vectors across RGB, HSV, and LAB color spaces."""

from os import PathLike
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import (
    get_descriptor_dimension,
    validate_color_space,
    validate_descriptor,
)

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


def load_image_bgr(image_path: ImagePath) -> NDArray[np.uint8]:
    """Read an image using OpenCV and return it as a BGR NumPy array (H x W x 3).

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        ValueError: If OpenCV cannot decode the image.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    # cv2.imread can fail on Windows Unicode paths. Reading raw bytes with
    # NumPy and decoding them with OpenCV ensures Windows Unicode safety.
    try:
        encoded_image = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Could not read image file: {path}") from exc

    image_bgr = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {path}")

    return image_bgr


def convert_color_space(
    image_bgr: NDArray[np.generic], color_space: str = "RGB"
) -> NDArray[np.generic]:
    """Convert a 3-channel OpenCV BGR image into the specified color space (RGB, HSV, or LAB).

    RGB stays uint8 (0-255). HSV and LAB use OpenCV's float32 conversion so the
    values follow ``COLOR_SPACE_CHANNELS``' documented ranges.

    Raises:
        ValueError: If ``image_bgr`` is invalid or ``color_space`` is unsupported.
    """
    cs_upper = validate_color_space(color_space)

    image_array = np.asarray(image_bgr)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(
            "Input image must be a 3D BGR array with shape H x W x 3; "
            f"received shape {image_array.shape}."
        )

    if cs_upper == "RGB":
        return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_float = image_array.astype(np.float32) / 255.0
    if cs_upper == "HSV":
        return cv2.cvtColor(image_float, cv2.COLOR_BGR2HSV)
    if cs_upper == "LAB":
        return cv2.cvtColor(image_float, cv2.COLOR_BGR2LAB)

    raise ValueError(f"Unsupported color space: {color_space}")


def load_image_rgb(image_path: ImagePath) -> NDArray[np.uint8]:
    """Load an image with OpenCV and return it as an RGB NumPy array.

    Maintained for backward compatibility.
    """
    return convert_color_space(load_image_bgr(image_path), "RGB")


def compute_channel_moments(channel: NDArray[np.generic]) -> tuple[float, float, float]:
    """Return mean, standard deviation, and signed third moment of a 2D channel.

    The third Color Moment is the signed cube root of the third central moment.
    """
    channel_array = np.asarray(channel)
    if channel_array.ndim != 2:
        raise ValueError(
            f"Channel must be a non-empty 2D array; received shape {channel_array.shape}."
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


def compute_color_moments(image: NDArray[np.generic]) -> MomentVector:
    """Compute a global 9D Color Moments vector for a 3-channel image array.

    Feature order:
        c1_mean, c1_std, c1_skew,
        c2_mean, c2_std, c2_skew,
        c3_mean, c3_std, c3_skew.
    """
    if image is None:
        raise ValueError("Image array must not be None.")

    image_array = np.asarray(image)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(
            "Image must have shape H x W x 3; "
            f"received shape {image_array.shape}."
        )
    if image_array.shape[0] == 0 or image_array.shape[1] == 0:
        raise ValueError("Image must have non-zero height and width.")

    features = [
        moment
        for channel_index in range(3)
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


def extract_color_moments(
    image: ImagePath | NDArray[np.generic],
    color_space: str = "RGB",
) -> MomentVector:
    """Extract global 9D Color Moments vector for a specified color space.

    Args:
        image: File path to the image or an OpenCV BGR image NumPy array.
        color_space: Target color space ("RGB", "HSV", or "LAB"). Defaults to "RGB".

    Returns:
        9-dimensional float64 feature vector.
    """
    cs_upper = validate_color_space(color_space)

    if isinstance(image, (str, PathLike)):
        bgr_image = load_image_bgr(image)
    elif isinstance(image, np.ndarray):
        bgr_image = image
    else:
        raise ValueError(f"Invalid image type: {type(image).__name__}.")

    color_image = convert_color_space(bgr_image, cs_upper)
    return compute_color_moments(color_image)


def extract_descriptor(
    image: ImagePath | NDArray[np.generic],
    descriptor: str = "rgb_hsv_lab",
) -> MomentVector:
    """Extract single or fused Color Moments feature vector for an image.

    Supported descriptors:
        - "rgb" (9D)
        - "hsv" (9D)
        - "lab" (9D)
        - "rgb_hsv" (18D)
        - "rgb_lab" (18D)
        - "hsv_lab" (18D)
        - "rgb_hsv_lab" (27D)

    Args:
        image: File path to the image or an OpenCV BGR image NumPy array.
        descriptor: Descriptor name specifying color spaces to extract and fuse.

    Returns:
        1D float64 NumPy array of dimension 9, 18, or 27.
    """
    desc_clean = validate_descriptor(descriptor)
    expected_dim = get_descriptor_dimension(desc_clean)

    # Parse color spaces from descriptor name (e.g. "rgb_hsv" -> ["RGB", "HSV"])
    cs_list = [cs.upper() for cs in desc_clean.split("_")]

    if isinstance(image, (str, PathLike)):
        bgr_image = load_image_bgr(image)
    elif isinstance(image, np.ndarray):
        bgr_image = image
    else:
        raise ValueError(f"Invalid image type: {type(image).__name__}.")

    vectors: list[NDArray[np.float64]] = []
    for cs in cs_list:
        color_img = convert_color_space(bgr_image, cs)
        vector = compute_color_moments(color_img)
        vectors.append(vector)

    fused_vector = np.concatenate(vectors, axis=0).astype(np.float64, copy=False)

    if fused_vector.shape != (expected_dim,):
        raise ValueError(
            f"Descriptor '{descriptor}' output shape error: "
            f"expected ({expected_dim},), got {fused_vector.shape}."
        )
    if not np.all(np.isfinite(fused_vector)):
        raise ValueError(f"Descriptor '{descriptor}' feature vector contains NaN or Inf.")

    return fused_vector
