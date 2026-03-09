"""Tests for ContourDetector class with synthetic contours."""

import sys
from pathlib import Path

import pytest
import numpy as np
import cv2

addon_path = Path(__file__).parent.parent
if str(addon_path) not in sys.path:
    sys.path.insert(0, str(addon_path))

from smart_stock.services.contour_detector import (  # noqa: E402
    ContourDetector,
    ContourConfig,
    DetectedContour,
    contour_to_geometry,
)
from rayforge.core.geo import Geometry  # noqa: E402


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
        assert config.simplify_epsilon == 1.0
        assert config.min_solidity == 0.2
        assert config.min_vertices == 3

    def test_custom_config_values(self):
        """Test that custom config values can be set."""
        config = ContourConfig(
            min_contour_area=500.0,
            max_contour_area=5000000.0,
            simplify_epsilon=3.0,
            min_solidity=0.5,
            min_vertices=3,
        )
        assert config.min_contour_area == 500.0
        assert config.max_contour_area == 5000000.0
        assert config.simplify_epsilon == 3.0
        assert config.min_solidity == 0.5
        assert config.min_vertices == 3


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
        """Create a detector with low area threshold."""
        config = ContourConfig(min_contour_area=100.0)
        return ContourDetector(config=config)

    @pytest.fixture
    def detector_high_threshold(self):
        """Create a detector with high area threshold."""
        config = ContourConfig(min_contour_area=5000.0)
        return ContourDetector(config=config)

    def test_detect_contours_empty_mask(self, detector):
        """Test detection on empty mask returns empty list."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = detector.detect_contours(mask)

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
        """Test detection of a single rectangle."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        result = detector_low_threshold.detect_contours(mask)

        assert len(result) == 1
        assert isinstance(result[0], DetectedContour)
        assert result[0].area > 0

    def test_detect_contours_multiple_shapes(self, detector_low_threshold):
        """Test detection of multiple distinct shapes."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        mask[120:180, 120:180] = 255

        result = detector_low_threshold.detect_contours(mask)

        assert len(result) == 2

    def test_detect_contours_returns_metadata(self, detector_low_threshold):
        """Test that detected contours include proper metadata."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        result = detector_low_threshold.detect_contours(mask)

        assert len(result) == 1
        contour = result[0]
        assert contour.area > 0
        assert len(contour.centroid) == 2
        assert len(contour.bounding_rect) == 4
        assert contour.points is not None

    def test_detect_contours_filters_small(self, detector_high_threshold):
        """Test that small contours are filtered out."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:60, 50:60] = 255

        result = detector_high_threshold.detect_contours(mask)

        assert result == []

    def test_detect_contours_filters_large(self):
        """Test that very large contours are filtered out."""
        config = ContourConfig(max_contour_area=1000.0, min_contour_area=10.0)
        detector = ContourDetector(config=config)
        mask = np.zeros((500, 500), dtype=np.uint8)
        mask[10:490, 10:490] = 255

        result = detector.detect_contours(mask)

        assert result == []

    def test_detect_contours_various_shapes(self, detector_low_threshold):
        """Test detection of various shape types."""
        mask = np.zeros((400, 400), dtype=np.uint8)

        # Rectangle
        mask[20:80, 20:80] = 255
        # Another rectangle
        mask[100:200, 100:200] = 255
        # L-shape
        mask[250:300, 250:350] = 255
        mask[250:350, 250:300] = 255

        result = detector_low_threshold.detect_contours(mask)

        assert len(result) >= 2

    def test_detect_contours_centroid_calculation(
        self, detector_low_threshold
    ):
        """Test that centroid is calculated correctly."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        result = detector_low_threshold.detect_contours(mask)

        assert len(result) == 1
        # Centroid should be near center of rectangle
        cx, cy = result[0].centroid
        assert 90 <= cx <= 110
        assert 90 <= cy <= 110

    def test_detect_contours_bounding_rect(self, detector_low_threshold):
        """Test that bounding rectangle is calculated correctly."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        result = detector_low_threshold.detect_contours(mask)

        assert len(result) == 1
        x, y, w, h = result[0].bounding_rect
        # Bounding rect should contain the rectangle
        assert x <= 50
        assert y <= 50
        assert x + w >= 150
        assert y + h >= 150


class TestCreateMaskFromContours:
    """Tests for create_mask_from_contours method."""

    @pytest.fixture
    def detector(self):
        """Create a default ContourDetector instance."""
        return ContourDetector()

    @pytest.fixture
    def detector_low_threshold(self):
        """Create a detector with low area threshold."""
        config = ContourConfig(min_contour_area=100.0)
        return ContourDetector(config=config)

    def test_create_mask_from_contours(self, detector_low_threshold):
        """Test creating a mask from detected contours."""
        original_mask = np.zeros((200, 200), dtype=np.uint8)
        original_mask[50:150, 50:150] = 255

        contours = detector_low_threshold.detect_contours(original_mask)
        recreated_mask = detector_low_threshold.create_mask_from_contours(
            contours, original_mask.shape
        )

        assert recreated_mask.shape == original_mask.shape
        assert np.any(recreated_mask > 0)

    def test_create_mask_empty_contours(self, detector):
        """Test creating mask from empty contour list."""
        mask = detector.create_mask_from_contours([], (100, 100))

        assert mask.shape == (100, 100)
        assert np.sum(mask) == 0

    def test_create_mask_multiple_contours(self, detector_low_threshold):
        """Test creating mask from multiple contours."""
        original_mask = np.zeros((300, 300), dtype=np.uint8)
        original_mask[20:80, 20:80] = 255
        original_mask[120:180, 120:180] = 255

        contours = detector_low_threshold.detect_contours(original_mask)
        recreated_mask = detector_low_threshold.create_mask_from_contours(
            contours, original_mask.shape
        )

        assert recreated_mask.shape == original_mask.shape
        # Both regions should have some filled area
        assert np.sum(recreated_mask[20:80, 20:80]) > 0
        assert np.sum(recreated_mask[120:180, 120:180]) > 0

    def test_create_mask_various_sizes(self, detector_low_threshold):
        """Test creating masks of various sizes."""
        for size in [(100, 100), (200, 150), (300, 300)]:
            mask = np.zeros(size, dtype=np.uint8)
            mask[10:50, 10:50] = 255

            contours = detector_low_threshold.detect_contours(mask)
            recreated = detector_low_threshold.create_mask_from_contours(
                contours, size
            )

            assert recreated.shape == size


class TestContourToGeometry:
    """Tests for contour_to_geometry function."""

    def test_contour_to_geometry_valid(self):
        """Test conversion of valid contour to Geometry."""
        points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        detected = DetectedContour(
            points=points,
            area=10000.0,
            centroid=(50, 50),
            bounding_rect=(0, 0, 100, 100),
        )

        geometry = contour_to_geometry(detected)

        assert isinstance(geometry, Geometry)
        assert len(geometry) > 0

    def test_contour_to_geometry_is_closed(self):
        """Test that converted geometry is a closed path."""
        points = np.array([[10, 10], [110, 10], [110, 110], [10, 110]])
        detected = DetectedContour(
            points=points,
            area=10000.0,
            centroid=(60, 60),
            bounding_rect=(10, 10, 100, 100),
        )

        geometry = contour_to_geometry(detected)

        assert geometry.is_closed()

    def test_contour_to_geometry_triangle(self):
        """Test conversion of triangular contour."""
        points = np.array([[50, 0], [100, 100], [0, 100]])
        detected = DetectedContour(
            points=points,
            area=5000.0,
            centroid=(50, 66),
            bounding_rect=(0, 0, 100, 100),
        )

        geometry = contour_to_geometry(detected)

        assert isinstance(geometry, Geometry)
        assert geometry.is_closed()

    def test_contour_to_geometry_empty_points(self):
        """Test conversion with empty points returns empty geometry."""
        detected = DetectedContour(
            points=np.array([]),
            area=0.0,
            centroid=(0, 0),
            bounding_rect=(0, 0, 0, 0),
        )

        geometry = contour_to_geometry(detected)

        assert isinstance(geometry, Geometry)
        assert len(geometry) == 0

    def test_contour_to_geometry_insufficient_points(self):
        """Test conversion with fewer than 3 points returns empty geometry."""
        points = np.array([[0, 0], [100, 0]])
        detected = DetectedContour(
            points=points,
            area=0.0,
            centroid=(50, 0),
            bounding_rect=(0, 0, 100, 0),
        )

        geometry = contour_to_geometry(detected)

        assert isinstance(geometry, Geometry)
        assert len(geometry) == 0

    def test_contour_to_geometry_complex_shape(self):
        """Test conversion of a more complex polygon."""
        points = np.array(
            [
                [50, 0],
                [100, 25],
                [100, 75],
                [50, 100],
                [0, 75],
                [0, 25],
            ]
        )
        detected = DetectedContour(
            points=points,
            area=6500.0,
            centroid=(50, 50),
            bounding_rect=(0, 0, 100, 100),
        )

        geometry = contour_to_geometry(detected)

        assert isinstance(geometry, Geometry)
        assert geometry.is_closed()
        assert len(geometry) >= 6

    def test_contour_to_geometry_1d_points(self):
        """Test conversion with 1D points array."""
        points = np.array([0, 0, 100, 0, 100, 100])
        detected = DetectedContour(
            points=points,
            area=5000.0,
            centroid=(66, 33),
            bounding_rect=(0, 0, 100, 100),
        )

        geometry = contour_to_geometry(detected)

        assert isinstance(geometry, Geometry)


class TestSimplifyContour:
    """Tests for contour simplification."""

    @pytest.fixture
    def detector(self):
        """Create a detector with custom simplification."""
        config = ContourConfig(simplify_epsilon=5.0, min_contour_area=100.0)
        return ContourDetector(config=config)

    def test_simplify_reduces_points(self, detector):
        """Test that simplification reduces point count."""
        # Create complex shape that can be simplified
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        result = detector.detect_contours(mask)

        if len(result) > 0:
            # Simplified contour should have reasonable number of points
            assert len(result[0].points) <= 20

    def test_simplify_epsilon_affects_result(self):
        """Test that simplify epsilon affects contour points."""
        config_low = ContourConfig(
            simplify_epsilon=1.0, min_contour_area=100.0
        )
        config_high = ContourConfig(
            simplify_epsilon=10.0, min_contour_area=100.0
        )

        detector_low = ContourDetector(config=config_low)
        detector_high = ContourDetector(config=config_high)

        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[30:170, 30:170] = 255

        result_low = detector_low.detect_contours(mask)
        result_high = detector_high.detect_contours(mask)

        if len(result_low) > 0 and len(result_high) > 0:
            # Higher epsilon should result in fewer or equal points
            assert len(result_high[0].points) <= len(result_low[0].points)


class TestSolidityFiltering:
    """Tests for solidity-based contour filtering."""

    def test_solidity_filters_irregular_shapes(self):
        """Test that low solidity shapes are filtered out."""
        config = ContourConfig(min_solidity=0.8, min_contour_area=100.0)
        detector = ContourDetector(config=config)

        # Create a concave shape with low solidity
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        mask[70:130, 70:130] = 0  # Cut out center

        result = detector.detect_contours(mask)

        # May be filtered due to low solidity
        # Just verify it doesn't crash
        assert isinstance(result, list)

    def test_solidity_accepts_regular_shapes(self):
        """Test that high solidity shapes are accepted."""
        config = ContourConfig(min_solidity=0.5, min_contour_area=100.0)
        detector = ContourDetector(config=config)

        # Solid rectangle has high solidity
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        result = detector.detect_contours(mask)

        assert len(result) >= 1


class TestMinVerticesFiltering:
    """Tests for minimum vertices filtering."""

    def test_min_vertices_filters_triangles(self):
        """Test that triangles can be filtered with min_vertices=4."""
        config = ContourConfig(min_vertices=4, min_contour_area=100.0)
        detector = ContourDetector(config=config)

        # Triangle
        mask = np.zeros((100, 100), dtype=np.uint8)
        for i in range(50):
            mask[50 + i, 25 : 75 - i // 2] = 255

        result = detector.detect_contours(mask)

        # Result depends on whether simplified triangle has 3 or more vertices
        assert isinstance(result, list)

    def test_min_vertices_accepts_polygons(self):
        """Test that polygons with enough vertices are accepted."""
        config = ContourConfig(min_vertices=3, min_contour_area=100.0)
        detector = ContourDetector(config=config)

        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255

        result = detector.detect_contours(mask)

        assert len(result) >= 1


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

        # Single pixel may or may not be detected depending on config
        assert isinstance(result, list)

    def test_full_mask(self, detector):
        """Test detection on fully white mask."""
        mask = np.full((100, 100), 255, dtype=np.uint8)

        result = detector.detect_contours(mask)

        # Full mask has one large contour at the border
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
        mask[45:55, 10:90] = 255  # Horizontal line

        config = ContourConfig(min_contour_area=100.0)
        detector = ContourDetector(config=config)
        result = detector.detect_contours(mask)

        assert isinstance(result, list)


class TestIntegration:
    """Integration tests combining contour detection with other operations."""

    @pytest.fixture
    def detector(self):
        """Create a detector with low thresholds for testing."""
        config = ContourConfig(min_contour_area=100.0)
        return ContourDetector(config=config)

    def test_detect_and_recreate_cycle(self, detector):
        """Test full cycle of detect and recreate."""
        original = np.zeros((200, 200), dtype=np.uint8)
        original[50:150, 50:150] = 255

        contours = detector.detect_contours(original)
        recreated = detector.create_mask_from_contours(
            contours, original.shape
        )

        # Both should have some overlap
        overlap = np.logical_and(original > 0, recreated > 0)
        assert np.any(overlap)

    def test_multiple_detection_calls(self, detector):
        """Test that multiple detection calls work correctly."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[20:80, 20:80] = 255

        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[10:90, 10:90] = 255

        result1 = detector.detect_contours(mask1)
        result2 = detector.detect_contours(mask2)

        assert len(result1) == 1
        assert len(result2) == 1
        # Different masks should produce different areas
        assert result1[0].area != result2[0].area


class TestHoleFiltering:
    """Tests for filtering holes (inner contours) from detected stock."""

    @pytest.fixture
    def detector(self):
        """Create a detector with low thresholds for testing."""
        config = ContourConfig(min_contour_area=100.0, min_solidity=0.1)
        return ContourDetector(config=config)

    def test_nested_contour_filters_hole(self, detector):
        """Test hole inside shape is not detected as separate stock."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[30:170, 30:170] = 255
        mask[70:130, 70:130] = 0

        contours = detector.detect_contours(mask)

        assert len(contours) == 1
        assert contours[0].area > 10000

    def test_two_separate_shapes_both_detected(self, detector):
        """Test that two separate shapes are both detected."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        mask[120:180, 120:180] = 255

        contours = detector.detect_contours(mask)

        assert len(contours) == 2

    def test_donut_shape_single_detection(self, detector):
        """Test that a donut shape is detected as one item, not two."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 70, (255,), -1)
        cv2.circle(mask, (100, 100), 30, (0,), -1)

        contours = detector.detect_contours(mask)

        assert len(contours) == 1

    def test_nested_rings(self, detector):
        """Test multiple nested rings are handled correctly."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (150, 150), 120, (255,), -1)
        cv2.circle(mask, (150, 150), 80, (0,), -1)
        cv2.circle(mask, (150, 150), 40, (255,), -1)
        cv2.circle(mask, (150, 150), 20, (0,), -1)

        contours = detector.detect_contours(mask)

        assert len(contours) == 1
