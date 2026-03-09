"""Image processing utilities for stock detection."""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters."""

    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    gaussian_blur_size: int = 5
    difference_threshold: int = 50
    morphology_kernel_size: int = 5
    min_mask_area_ratio: float = 0.001
    adaptive_blur: bool = True
    close_iterations: int = 1
    open_iterations: int = 1
    dilate_iterations: int = 0


class ImageProcessor:
    """
    Handles image normalization and preprocessing for stock detection.

    Uses background subtraction to detect new objects by comparing
    the current frame against a reference image of the empty machine.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()

    def set_threshold(self, threshold: int) -> None:
        """Update the difference threshold."""
        self.config.difference_threshold = threshold

    def normalize_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image brightness and contrast using CLAHE.

        Args:
            image: Input BGR or grayscale image.

        Returns:
            Normalized image in the same color space.
        """
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_size,
            )
            l_clahe = clahe.apply(l_channel)
            lab = cv2.merge((l_clahe, a_channel, b_channel))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_size,
            )
            return clahe.apply(image)

    def prepare_for_comparison(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare an image for comparison by normalizing and blurring.

        Args:
            image: Input BGR image.

        Returns:
            Preprocessed grayscale image.
        """
        normalized = self.normalize_brightness_contrast(image)
        gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

        blur_size = self.config.gaussian_blur_size
        if self.config.adaptive_blur:
            h, w = gray.shape[:2]
            min_dim = min(w, h)
            blur_size = max(5, min(15, min_dim // 100))
            if blur_size % 2 == 0:
                blur_size += 1

        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        return blurred

    def compute_difference(
        self,
        current: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the difference between current image and reference.

        Detects new objects using color-aware background subtraction.

        Args:
            current: Current BGR image from camera.
            reference: Reference BGR image of empty machine.

        Returns:
            Binary mask highlighting potential stock regions.
        """
        ref_h, ref_w = reference.shape[:2]
        curr_h, curr_w = current.shape[:2]

        if ref_w != curr_w or ref_h != curr_h:
            current = cv2.resize(current, (ref_w, ref_h))

        ref_norm = self.normalize_brightness_contrast(reference)
        curr_norm = self.normalize_brightness_contrast(current)

        blur_size = self.config.gaussian_blur_size
        if self.config.adaptive_blur:
            h, w = ref_norm.shape[:2]
            min_dim = min(w, h)
            blur_size = max(5, min(15, min_dim // 100))
            if blur_size % 2 == 0:
                blur_size += 1

        ref_blurred = cv2.GaussianBlur(ref_norm, (blur_size, blur_size), 0)
        curr_blurred = cv2.GaussianBlur(curr_norm, (blur_size, blur_size), 0)

        diff_b = cv2.absdiff(ref_blurred[:, :, 0], curr_blurred[:, :, 0])
        diff_g = cv2.absdiff(ref_blurred[:, :, 1], curr_blurred[:, :, 1])
        diff_r = cv2.absdiff(ref_blurred[:, :, 2], curr_blurred[:, :, 2])

        diff = np.maximum(np.maximum(diff_b, diff_g), diff_r)

        _, mask = cv2.threshold(
            diff, self.config.difference_threshold, 255, cv2.THRESH_BINARY
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                self.config.morphology_kernel_size,
                self.config.morphology_kernel_size,
            ),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.config.close_iterations,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=self.config.open_iterations,
        )

        mask = cv2.dilate(
            mask, kernel, iterations=self.config.dilate_iterations
        )

        return mask

    def apply_morphology(
        self, mask: np.ndarray, operation: str = "clean"
    ) -> np.ndarray:
        """
        Apply morphological operations to clean up a mask.

        Args:
            mask: Binary input mask.
            operation: Type of operation ('clean', 'dilate', 'erode').

        Returns:
            Processed mask.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                self.config.morphology_kernel_size,
                self.config.morphology_kernel_size,
            ),
        )

        if operation == "clean":
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == "dilate":
            mask = cv2.dilate(mask, kernel, iterations=1)
        elif operation == "erode":
            mask = cv2.erode(mask, kernel, iterations=1)

        return mask
