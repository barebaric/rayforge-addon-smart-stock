"""Utility functions for smart stock addon."""

from typing import Tuple

DEFAULT_OUTPUT_WIDTH = 800


def get_output_size(
    physical_area: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Tuple[int, int]:
    """
    Calculate output size matching physical area aspect ratio.

    Args:
        physical_area: Tuple of ((min_x, min_y), (max_x, max_y)).

    Returns:
        Tuple of (width, height) in pixels.
    """
    (x_min, y_min), (x_max, y_max) = physical_area
    physical_width = x_max - x_min
    physical_height = y_max - y_min

    aspect = physical_width / physical_height if physical_height > 0 else 1

    if aspect >= 1:
        out_width = DEFAULT_OUTPUT_WIDTH
        out_height = int(DEFAULT_OUTPUT_WIDTH / aspect)
    else:
        out_height = DEFAULT_OUTPUT_WIDTH
        out_width = int(DEFAULT_OUTPUT_WIDTH * aspect)

    return (out_width, out_height)
