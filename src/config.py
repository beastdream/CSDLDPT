"""Configuration module for descriptors, color spaces, and normalization options."""

from typing import Final, Literal

ColorSpace: type = Literal["RGB", "HSV", "LAB"]
DescriptorType: type = Literal[
    "rgb", "hsv", "lab", "rgb_hsv", "rgb_lab", "hsv_lab", "rgb_hsv_lab"
]
NormalizationType: type = Literal["none", "zscore"]

SUPPORTED_COLOR_SPACES: Final[tuple[str, ...]] = ("RGB", "HSV", "LAB")

DESCRIPTOR_DIMENSIONS: Final[dict[str, int]] = {
    "rgb": 9,
    "hsv": 9,
    "lab": 9,
    "rgb_hsv": 18,
    "rgb_lab": 18,
    "hsv_lab": 18,
    "rgb_hsv_lab": 27,
}

SUPPORTED_DESCRIPTORS: Final[tuple[str, ...]] = tuple(DESCRIPTOR_DIMENSIONS.keys())

DEFAULT_DESCRIPTOR: Final[str] = "rgb"
DEFAULT_NORMALIZATION: Final[str] = "none"


def validate_color_space(color_space: str) -> str:
    """Validate and return the uppercase string of a color space.

    Raises:
        ValueError: If ``color_space`` is not one of RGB, HSV, LAB.
    """
    if not isinstance(color_space, str):
        raise ValueError(f"color_space must be a string, got {type(color_space).__name__}.")

    cs_upper = color_space.strip().upper()
    if cs_upper not in SUPPORTED_COLOR_SPACES:
        raise ValueError(
            f"Unsupported color space '{color_space}'. "
            f"Supported color spaces: {', '.join(SUPPORTED_COLOR_SPACES)}."
        )
    return cs_upper


def validate_descriptor(descriptor: str) -> str:
    """Validate and return the lowercase string of a descriptor identifier.

    Raises:
        ValueError: If ``descriptor`` is not supported.
    """
    if not isinstance(descriptor, str):
        raise ValueError(f"descriptor must be a string, got {type(descriptor).__name__}.")

    desc_lower = descriptor.strip().lower()
    if desc_lower not in DESCRIPTOR_DIMENSIONS:
        raise ValueError(
            f"Unsupported descriptor '{descriptor}'. "
            f"Supported descriptors: {', '.join(SUPPORTED_DESCRIPTORS)}."
        )
    return desc_lower


def get_descriptor_dimension(descriptor: str) -> int:
    """Return the expected feature vector dimension for a descriptor."""
    validated = validate_descriptor(descriptor)
    return DESCRIPTOR_DIMENSIONS[validated]
