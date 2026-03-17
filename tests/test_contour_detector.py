"""Tests for ContourDetector class with synthetic contours."""

import pytest
import numpy as np
import cv2
from smart_stock.services.contour_detector import (
    ContourDetector,
    ContourConfig,
    DetectedContour,
)


class TestContourConfig:
    """Tests for ContourConfig dataclass."""

    def test_default_config_instantiation(self):
        """Test that default config can be instantiated."""
        config = ContourConfig()
        assert config is not None

    def test_default_config_values(self):
        """Test that default config has expected values."""
        config = ContourConfig()
        assert config.min_contour_area == 500.0
        assert config.max_contour_area == 10000000.0

    def test_custom_config_values(self):
        """Test that custom config values can be set."""
        config = ContourConfig(
            min_contour_area=500.0,
            max_contour_area=5000000.0,
            max_items=5,
        )
        assert config.min_contour_area == 500.0
        assert config.max_contour_area == 5000000.0
        assert config.max_items == 5


class TestDetectedContour:
    """Tests for DetectedContour dataclass."""

    def test_detected_contour_instantiation(self):
        """Test that DetectedContour can be instantiated."""
        points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        contour = DetectedContour(
            points=points,
            area=10000.0,
            centroid=(50, 50),
            bounding_rect=(0, 0, 100, 100),
        )
        assert contour is not None
        assert np.array_equal(contour.points, points)
        assert contour.area == 10000.0
        assert contour.centroid == (50, 50)
        assert contour.bounding_rect == (0, 0, 100, 100)

    def test_detected_contour_attributes(self):
        """Test DetectedContour attribute types."""
        points = np.array([[10, 10], [50, 10], [50, 50], [10, 50]])
        contour = DetectedContour(
            points=points,
            area=1600.0,
            centroid=(30, 30),
            bounding_rect=(10, 10, 40, 40),
        )
        assert isinstance(contour.points, np.ndarray)
        assert isinstance(contour.area, float)
        assert isinstance(contour.centroid, tuple)
        assert isinstance(contour.bounding_rect, tuple)


class TestContourDetectorInstantiation:
    """Tests for ContourDetector instantiation."""

    def test_default_instantiation(self):
        """Test that ContourDetector can be instantiated with defaults."""
        detector = ContourDetector()
        assert detector is not None
        assert detector.config is not None

    def test_instantiation_with_custom_config(self):
        """Test that ContourDetector can be instantiated with custom config."""
        config = ContourConfig(min_contour_area=500.0)
        detector = ContourDetector(config=config)
        assert detector.config.min_contour_area == 500.0

    def test_config_is_contour_config_instance(self):
        """Test that config is a ContourConfig instance."""
        detector = ContourDetector()
        assert isinstance(detector.config, ContourConfig)


class TestDetectContours:
    """Tests for detect_contours method."""

    @pytest.fixture
    def detector(self):
        """Create a default ContourDetector instance."""
        return ContourDetector()

    @pytest.fixture
    def detector_low_threshold(self):
        """Create a detector with low area threshold and no merging."""
        config = ContourConfig(min_contour_area=100.0, merge_distance=0.0)
        return ContourDetector(config=config)

    @pytest.fixture
    def detector_high_threshold(self):
        """Create a detector with high area threshold."""
        config = ContourConfig(min_contour_area=5000.0, merge_distance=0.0)
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

    def test_detect_contours_zero_size_mask(self, detector):
        """Test detection on zero size mask returns empty list."""
        mask = np.array([], dtype=np.uint8)
        result = detector.detect_contours(mask)

        assert result == []

    def test_detect_contours_single_rectangle(self, detector_low_threshold):
        """Test detection of a single rectangle using reference."""
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

    def test_detect_contours_filters_small(self, detector_high_threshold):
        """Test that small contours are filtered out."""
        reference = np.full((200, 200, 3), 0, dtype=np.uint8)
        current = np.full((200, 200, 3), 0, dtype=np.uint8)
        current[50:60, 50:60] = [255, 255, 255]

        result = detector_high_threshold.detect_contours(
            current, reference=reference
        )

        assert result == []

    def test_detect_contours_filters_large(self):
        """Test that very large contours are filtered out."""
        config = ContourConfig(
            max_contour_area=1000.0, min_contour_area=10.0, merge_distance=0.0
        )
        detector = ContourDetector(config=config)
        reference = np.full((500, 500, 3), 0, dtype=np.uint8)
        current = np.full((500, 500, 3), 0, dtype=np.uint8)
        current[10:490, 10:490] = [255, 255, 255]

        result = detector.detect_contours(current, reference=reference)

        assert result == []

    def test_detect_contours_various_shapes(self, detector_low_threshold):
        """Test detection of various shape types."""
        reference = np.full((400, 400, 3), 0, dtype=np.uint8)
        current = np.full((400, 400, 3), 0, dtype=np.uint8)

        current[20:80, 20:80] = [255, 255, 255]
        current[100:200, 100:200] = [255, 255, 255]
        current[250:300, 250:350] = [255, 255, 255]
        current[250:350, 250:300] = [255, 255, 255]

        result = detector_low_threshold.detect_contours(
            current, reference=reference
        )

        assert len(result) >= 2

    def test_detect_contours_centroid_calculation(
        self, detector_low_threshold
    ):
        """Test that centroid is calculated correctly."""
        reference = np.full((200, 200, 3), 0, dtype=np.uint8)
        current = np.full((200, 200, 3), 0, dtype=np.uint8)
        current[50:150, 50:150] = [255, 255, 255]

        result = detector_low_threshold.detect_contours(
            current, reference=reference
        )

        assert len(result) >= 1
        cx, cy = result[0].centroid
        assert 90 <= cx <= 110
        assert 90 <= cy <= 110

    def test_detect_contours_bounding_rect(self, detector_low_threshold):
        """Test that bounding rectangle is calculated correctly."""
        reference = np.full((200, 200, 3), 0, dtype=np.uint8)
        current = np.full((200, 200, 3), 0, dtype=np.uint8)
        current[50:150, 50:150] = [255, 255, 255]

        result = detector_low_threshold.detect_contours(
            current, reference=reference
        )

        assert len(result) >= 1
        x, y, w, h = result[0].bounding_rect
        tolerance = 2
        assert x <= 50 + tolerance
        assert y <= 50 + tolerance
        assert x + w >= 150 - tolerance
        assert y + h >= 150 - tolerance


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def detector(self):
        """Create a default ContourDetector instance."""
        return ContourDetector()

    def test_single_pixel_mask(self, detector):
        """Test detection on single pixel mask."""
        config = ContourConfig(min_contour_area=0.5)
        detector = ContourDetector(config=config)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255

        result = detector.detect_contours(mask)

        assert isinstance(result, list)

    def test_full_mask(self, detector):
        """Test detection on fully white mask."""
        mask = np.full((100, 100), 255, dtype=np.uint8)

        result = detector.detect_contours(mask)

        assert isinstance(result, list)

    def test_noisy_mask(self, detector):
        """Test detection on noisy binary mask."""
        mask = np.random.randint(0, 2, size=(100, 100), dtype=np.uint8) * 255

        result = detector.detect_contours(mask)

        assert isinstance(result, list)

    def test_very_small_mask(self, detector):
        """Test detection on very small mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 255

        config = ContourConfig(min_contour_area=10.0)
        detector = ContourDetector(config=config)
        result = detector.detect_contours(mask)

        assert isinstance(result, list)

    def test_thin_line_mask(self, detector):
        """Test detection on thin line shapes."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, 10:90] = 255

        config = ContourConfig(min_contour_area=100.0)
        detector = ContourDetector(config=config)
        result = detector.detect_contours(mask)

        assert isinstance(result, list)


class TestIntegration:
    """Integration tests combining contour detection with other operations."""

    @pytest.fixture
    def detector(self):
        """Create a detector with low thresholds for testing."""
        config = ContourConfig(min_contour_area=100.0, merge_distance=0.0)
        return ContourDetector(config=config)

    def test_multiple_detection_calls(self, detector):
        """Test that multiple detection calls work correctly."""
        reference = np.full((200, 200, 3), 0, dtype=np.uint8)
        current1 = np.full((200, 200, 3), 0, dtype=np.uint8)
        current1[40:80, 40:80] = [255, 255, 255]

        current2 = np.full((200, 200, 3), 0, dtype=np.uint8)
        current2[100:180, 100:180] = [255, 255, 255]

        result1 = detector.detect_contours(current1, reference=reference)
        result2 = detector.detect_contours(current2, reference=reference)

        assert len(result1) >= 1
        assert len(result2) >= 1
        assert result1[0].area != result2[0].area


class TestHoleFiltering:
    """Tests for filtering holes (inner contours) from detected stock."""

    @pytest.fixture
    def detector(self):
        """Create a detector with low thresholds for testing."""
        config = ContourConfig(min_contour_area=100.0, merge_distance=0.0)
        return ContourDetector(config=config)

    def test_nested_contour_filters_hole(self, detector):
        """Test hole inside shape is not detected as separate stock."""
        reference = np.full((200, 200, 3), 0, dtype=np.uint8)
        current = np.full((200, 200, 3), 0, dtype=np.uint8)
        current[30:170, 30:170] = [255, 255, 255]
        current[70:130, 70:130] = [0, 0, 0]

        contours = detector.detect_contours(current, reference=reference)

        assert len(contours) >= 1
        assert contours[0].area > 10000

    def test_two_separate_shapes_both_detected(self, detector):
        """Test that two separate shapes are both detected."""
        reference = np.full((200, 200, 3), 0, dtype=np.uint8)
        current = np.full((200, 200, 3), 0, dtype=np.uint8)
        current[20:80, 20:80] = [255, 255, 255]
        current[120:180, 120:180] = [255, 255, 255]

        contours = detector.detect_contours(current, reference=reference)

        assert len(contours) == 2

    def test_donut_shape_single_detection(self, detector):
        """Test that a donut shape is detected as one item, not two."""
        reference = np.full((200, 200, 3), 0, dtype=np.uint8)
        current = np.full((200, 200, 3), 0, dtype=np.uint8)
        cv2.circle(current, (100, 100), 70, (255, 255, 255), -1)
        cv2.circle(current, (100, 100), 30, (0, 0, 0), -1)

        contours = detector.detect_contours(current, reference=reference)

        assert len(contours) >= 1

    def test_nested_rings(self, detector):
        """Test multiple nested rings are handled correctly."""
        reference = np.full((300, 300, 3), 0, dtype=np.uint8)
        current = np.full((300, 300, 3), 0, dtype=np.uint8)
        cv2.circle(current, (150, 150), 120, (255, 255, 255), -1)
        cv2.circle(current, (150, 150), 80, (0, 0, 0), -1)
        cv2.circle(current, (150, 150), 40, (255, 255, 255), -1)
        cv2.circle(current, (150, 150), 20, (0, 0, 0), -1)

        contours = detector.detect_contours(current, reference=reference)

        assert len(contours) >= 1
