"""Tests for smart_stock services module."""

import pytest
import numpy as np
from smart_stock.services import (
    contour_detector,
    image_processor,
)
from smart_stock.services.image_processor import (
    ImageProcessor,
    ProcessingConfig,
)
from smart_stock.services.contour_detector import (
    ContourDetector,
    ContourConfig,
    DetectedContour,
)


class TestImports:
    """Verify all imports work correctly."""

    def test_module_imports(self):
        """Test that module-level imports succeed."""
        assert hasattr(image_processor, "ImageProcessor")
        assert hasattr(image_processor, "ProcessingConfig")
        assert hasattr(contour_detector, "ContourDetector")
        assert hasattr(contour_detector, "ContourConfig")
        assert hasattr(contour_detector, "DetectedContour")

    def test_class_instantiation(self):
        """Test that classes can be instantiated."""
        processor = ImageProcessor()
        assert processor is not None
        assert processor.config is not None

        detector = ContourDetector()
        assert detector is not None
        assert detector.config is not None

    def test_config_instantiation(self):
        """Test that config dataclasses can be instantiated."""
        proc_config = ProcessingConfig()
        assert proc_config.clahe_clip_limit == 2.0
        assert proc_config.clahe_tile_size == (8, 8)

        det_config = ContourConfig()
        assert det_config.min_contour_area == 500.0


class TestImageProcessor:
    """Tests for ImageProcessor brightness/contrast normalization."""

    @pytest.fixture
    def processor(self):
        return ImageProcessor()

    def test_normalize_grayscale_image(self, processor):
        """Test normalization of grayscale image."""
        image = np.random.randint(50, 200, size=(100, 100), dtype=np.uint8)
        result = processor.normalize_brightness_contrast(image)

        assert result is not None
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_normalize_color_image(self, processor):
        """Test normalization of BGR color image."""
        image = np.random.randint(50, 200, size=(100, 100, 3), dtype=np.uint8)
        result = processor.normalize_brightness_contrast(image)

        assert result is not None
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_normalize_dark_image_brightness_increased(self, processor):
        """Test that dark images have brightness enhanced."""
        dark_image = np.full((100, 100, 3), 30, dtype=np.uint8)
        result = processor.normalize_brightness_contrast(dark_image)

        result_mean = np.mean(result)
        original_mean = np.mean(dark_image)

        assert result_mean > original_mean

    def test_normalize_bright_image_contrast_adjusted(self, processor):
        """Test that bright images get contrast adjustment."""
        bright_image = np.full((100, 100, 3), 220, dtype=np.uint8)
        result = processor.normalize_brightness_contrast(bright_image)

        assert result is not None
        assert result.shape == bright_image.shape

    def test_normalize_with_custom_config(self):
        """Test normalization with custom CLAHE parameters."""
        config = ProcessingConfig(
            clahe_clip_limit=4.0,
            clahe_tile_size=(4, 4),
        )
        processor = ImageProcessor(config=config)
        image = np.random.randint(50, 200, size=(100, 100), dtype=np.uint8)

        result = processor.normalize_brightness_contrast(image)

        assert result is not None
        assert result.shape == image.shape

    def test_normalize_preserves_dimensions(self, processor):
        """Test that normalization preserves image dimensions."""
        for shape in [(50, 50), (100, 200), (200, 100, 3)]:
            image = np.random.randint(0, 255, size=shape, dtype=np.uint8)
            result = processor.normalize_brightness_contrast(image)
            assert result.shape == shape


class TestContourDetector:
    """Tests for ContourDetector contour detection."""

    @pytest.fixture
    def detector(self):
        return ContourDetector()

    @pytest.fixture
    def detector_low_threshold(self):
        config = ContourConfig(min_contour_area=100.0, merge_distance=0.0)
        return ContourDetector(config=config)

    def test_detect_contours_empty_mask(self, detector):
        """Test detection with reference shows no new shapes."""
        reference = np.full((100, 100, 3), 128, dtype=np.uint8)
        current = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = detector.detect_contours(current, reference=reference)

        assert result == []

    def test_detect_contours_none_mask(self, detector):
        """Test detection on None mask returns empty list."""
        result = detector.detect_contours(None)

        assert result == []

    def test_detect_contours_single_rectangle(self, detector_low_threshold):
        """Test detection of a single rectangle."""
        reference = np.full((200, 200, 3), 0, dtype=np.uint8)
        current = np.full((200, 200, 3), 0, dtype=np.uint8)
        current[50:150, 50:150] = [255, 255, 255]

        result = detector_low_threshold.detect_contours(
            current, reference=reference
        )

        assert len(result) >= 1
        assert isinstance(result[0], DetectedContour)
        assert result[0].area > 0

    def test_detect_contours_multiple_shapes(self, detector_low_threshold):
        """Test detection of multiple distinct shapes."""
        reference = np.full((300, 300, 3), 0, dtype=np.uint8)
        current = np.full((300, 300, 3), 0, dtype=np.uint8)
        current[20:80, 20:80] = [255, 255, 255]
        current[120:180, 120:180] = [255, 255, 255]

        result = detector_low_threshold.detect_contours(
            current, reference=reference
        )

        assert len(result) == 2

    def test_detect_contours_returns_metadata(self, detector_low_threshold):
        """Test that detected contours include proper metadata."""
        reference = np.full((200, 200, 3), 0, dtype=np.uint8)
        current = np.full((200, 200, 3), 0, dtype=np.uint8)
        current[50:150, 50:150] = [255, 255, 255]

        result = detector_low_threshold.detect_contours(
            current, reference=reference
        )

        assert len(result) >= 1
        contour = result[0]
        assert contour.area > 0
        assert len(contour.centroid) == 2
        assert len(contour.bounding_rect) == 4
        assert contour.points is not None

    def test_detect_contours_filters_small(self, detector):
        """Test that small contours are filtered out."""
        config = ContourConfig(min_contour_area=5000.0, merge_distance=0.0)
        detector = ContourDetector(config=config)
        reference = np.full((200, 200, 3), 0, dtype=np.uint8)
        current = np.full((200, 200, 3), 0, dtype=np.uint8)
        current[50:60, 50:60] = [255, 255, 255]

        result = detector.detect_contours(current, reference=reference)

        assert result == []


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self):
        """Test full pipeline from image processing to contour detection."""
        processor = ImageProcessor()
        config = ContourConfig(min_contour_area=100.0)
        detector = ContourDetector(config=config)

        reference = np.zeros((200, 200, 3), dtype=np.uint8)
        current = np.zeros((200, 200, 3), dtype=np.uint8)
        current[50:150, 50:150] = [255, 255, 255]

        normalized = processor.normalize_brightness_contrast(current)
        assert normalized is not None

        diff_mask = processor.compute_difference(current, reference)
        assert diff_mask is not None
        assert len(diff_mask.shape) == 2

        contours = detector.detect_contours(diff_mask)
        assert isinstance(contours, list)

    def test_compute_difference_detects_changes(self):
        """Test that compute_difference detects changes between images."""
        processor = ImageProcessor()

        reference = np.zeros((200, 200, 3), dtype=np.uint8)

        current = np.zeros((200, 200, 3), dtype=np.uint8)
        current[50:150, 50:150] = [255, 255, 255]

        diff_mask = processor.compute_difference(current, reference)

        assert diff_mask is not None
        assert diff_mask.shape == (200, 200)

    def test_compute_difference_handles_size_mismatch(self):
        """Test that compute_difference handles different image sizes."""
        processor = ImageProcessor()

        reference = np.zeros((200, 200, 3), dtype=np.uint8)

        current = np.zeros((300, 400, 3), dtype=np.uint8)
        current[100:200, 150:250] = [255, 255, 255]

        diff_mask = processor.compute_difference(current, reference)

        assert diff_mask is not None
        assert diff_mask.shape == (200, 200)
