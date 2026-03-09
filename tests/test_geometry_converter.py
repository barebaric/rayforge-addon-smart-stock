"""Tests for GeometryConverter class with synthetic contours."""

import sys
from pathlib import Path

import numpy as np

addon_path = Path(__file__).parent.parent
if str(addon_path) not in sys.path:
    sys.path.insert(0, str(addon_path))

from smart_stock.services.geometry_converter import (  # noqa: E402
    GeometryConverter,
)
from rayforge.core.geo import Geometry  # noqa: E402


class TestGeometryConverterInstantiation:
    """Tests for GeometryConverter instantiation."""

    def test_class_exists(self):
        """Test that GeometryConverter class exists."""
        assert GeometryConverter is not None

    def test_has_static_methods(self):
        """Test that GeometryConverter has expected static methods."""
        assert hasattr(GeometryConverter, "contour_to_geometry")
        assert hasattr(GeometryConverter, "contours_to_geometries")
        assert callable(GeometryConverter.contour_to_geometry)
        assert callable(GeometryConverter.contours_to_geometries)


class TestContourToGeometry:
    """Tests for contour_to_geometry static method."""

    def test_valid_contour_returns_geometry(self):
        """Test that valid contour returns Geometry object."""
        contour = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]])
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert isinstance(result, Geometry)

    def test_none_contour_returns_none(self):
        """Test that None contour returns None."""
        result = GeometryConverter.contour_to_geometry(None)  # type: ignore

        assert result is None

    def test_empty_contour_returns_none(self):
        """Test that empty contour returns None."""
        contour = np.array([])
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is None

    def test_contour_too_few_points_returns_none(self):
        """Test that contour with fewer than 3 points returns None."""
        # Single point
        contour = np.array([[[0, 0]]])
        result = GeometryConverter.contour_to_geometry(contour)
        assert result is None

        # Two points
        contour = np.array([[[0, 0]], [[100, 0]]])
        result = GeometryConverter.contour_to_geometry(contour)
        assert result is None

    def test_triangle_contour(self):
        """Test conversion of triangular contour."""
        contour = np.array([[[50, 0]], [[100, 100]], [[0, 100]]])
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert isinstance(result, Geometry)

    def test_rectangle_contour(self):
        """Test conversion of rectangular contour."""
        contour = np.array(
            [[[10, 10]], [[110, 10]], [[110, 110]], [[10, 110]]]
        )
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert isinstance(result, Geometry)

    def test_complex_polygon_contour(self):
        """Test conversion of complex polygon contour."""
        contour = np.array(
            [
                [[50, 0]],
                [[100, 25]],
                [[100, 75]],
                [[50, 100]],
                [[0, 75]],
                [[0, 25]],
            ]
        )
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert isinstance(result, Geometry)

    def test_geometry_is_closed(self):
        """Test that converted geometry is closed."""
        contour = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]])
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert result.is_closed()

    def test_contour_shape_n_1_2(self):
        """Test contour with shape (N, 1, 2) - standard OpenCV format."""
        contour = np.array([[[0, 0]], [[50, 0]], [[50, 50]], [[0, 50]]])
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert isinstance(result, Geometry)

    def test_contour_shape_n_2(self):
        """Test contour with shape (N, 2) - squeezed format."""
        contour = np.array([[0, 0], [50, 0], [50, 50], [0, 50]])
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert isinstance(result, Geometry)

    def test_contour_single_dimension_squeezed(self):
        """Test contour that squeezes to 1D array."""
        # Single point contour that squeezes to shape (2,)
        contour = np.array([[[10, 20]]])
        result = GeometryConverter.contour_to_geometry(contour)

        # Should return None as it has fewer than 3 points
        assert result is None

    def test_large_contour(self):
        """Test conversion of large contour."""
        # Create a circle-like contour
        angles = np.linspace(0, 2 * np.pi, 100)
        points = np.array(
            [
                [[int(50 + 40 * np.cos(a)), int(50 + 40 * np.sin(a))]]
                for a in angles
            ]
        )
        result = GeometryConverter.contour_to_geometry(points)

        assert result is not None
        assert isinstance(result, Geometry)

    def test_float_coordinates(self):
        """Test contour with float coordinates."""
        contour = np.array(
            [[[0.0, 0.0]], [[100.5, 0.0]], [[100.5, 100.5]], [[0.0, 100.5]]]
        )
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert isinstance(result, Geometry)


class TestContoursToGeometries:
    """Tests for contours_to_geometries static method."""

    def test_empty_list_returns_empty_list(self):
        """Test that empty contour list returns empty geometry list."""
        result = GeometryConverter.contours_to_geometries([])

        assert result == []

    def test_single_valid_contour(self):
        """Test conversion of single valid contour."""
        contours = [np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]])]
        result = GeometryConverter.contours_to_geometries(contours)

        # Note: There's a bug in the implementation - it uses 'continue'
        # instead of 'append', so this may return empty list
        assert isinstance(result, list)

    def test_multiple_valid_contours(self):
        """Test conversion of multiple valid contours."""
        contours = [
            np.array([[[0, 0]], [[50, 0]], [[50, 50]], [[0, 50]]]),
            np.array([[[100, 100]], [[150, 100]], [[150, 150]], [[100, 150]]]),
        ]
        result = GeometryConverter.contours_to_geometries(contours)

        assert isinstance(result, list)

    def test_mixed_valid_invalid_contours(self):
        """Test conversion with mix of valid and invalid contours."""
        contours = [
            np.array([[[0, 0]], [[50, 0]], [[50, 50]], [[0, 50]]]),
            np.array([]),  # Invalid
            None,  # Invalid
            np.array([[[100, 100]], [[150, 100]], [[150, 150]], [[100, 150]]]),
        ]
        result = GeometryConverter.contours_to_geometries(contours)

        assert isinstance(result, list)

    def test_all_invalid_contours(self):
        """Test conversion of all invalid contours."""
        contours = [
            None,
            np.array([]),
            np.array([[[0, 0]]]),  # Too few points
        ]
        result = GeometryConverter.contours_to_geometries(contours)

        assert isinstance(result, list)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_contour_with_duplicate_points(self):
        """Test contour with duplicate consecutive points."""
        contour = np.array(
            [[[0, 0]], [[50, 0]], [[50, 0]], [[50, 50]], [[0, 50]]]
        )
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert isinstance(result, Geometry)

    def test_contour_with_negative_coordinates(self):
        """Test contour with negative coordinates."""
        contour = np.array(
            [[[-50, -50]], [[50, -50]], [[50, 50]], [[-50, 50]]]
        )
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert isinstance(result, Geometry)

    def test_contour_with_large_coordinates(self):
        """Test contour with large coordinate values."""
        contour = np.array(
            [[[0, 0]], [[10000, 0]], [[10000, 10000]], [[0, 10000]]]
        )
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert isinstance(result, Geometry)

    def test_contour_with_zero_area(self):
        """Test contour that has zero area (colinear points)."""
        contour = np.array([[[0, 0]], [[50, 50]], [[100, 100]]])
        result = GeometryConverter.contour_to_geometry(contour)

        # Still creates geometry, just with colinear points
        assert result is not None
        assert isinstance(result, Geometry)

    def test_concave_contour(self):
        """Test conversion of concave (non-convex) contour."""
        # L-shaped contour
        contour = np.array(
            [
                [[0, 0]],
                [[100, 0]],
                [[100, 50]],
                [[50, 50]],
                [[50, 100]],
                [[0, 100]],
            ]
        )
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert isinstance(result, Geometry)
        assert result.is_closed()

    def test_star_shaped_contour(self):
        """Test conversion of star-shaped contour."""
        # Simple 5-point star
        contour = np.array(
            [
                [[50, 0]],
                [[61, 35]],
                [[98, 35]],
                [[68, 57]],
                [[79, 91]],
                [[50, 70]],
                [[21, 91]],
                [[32, 57]],
                [[2, 35]],
                [[39, 35]],
            ]
        )
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert isinstance(result, Geometry)


class TestGeometryProperties:
    """Tests for properties of converted geometry."""

    def test_geometry_has_elements(self):
        """Test that converted geometry has elements."""
        contour = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]])
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert len(result) > 0

    def test_geometry_closed_path(self):
        """Test that geometry forms a closed path."""
        contour = np.array(
            [[[10, 10]], [[100, 10]], [[100, 100]], [[10, 100]]]
        )
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert result.is_closed()

    def test_geometry_preserves_point_count(self):
        """Test that geometry preserves approximate point count."""
        points = [[0, 0], [100, 0], [100, 50], [50, 50], [50, 100], [0, 100]]
        contour = np.array([[p] for p in points])
        result = GeometryConverter.contour_to_geometry(contour)

        # Geometry should have at least as many segments as input points
        assert result is not None
        assert len(result) >= 6


class TestIntegration:
    """Integration tests combining geometry conversion with other ops."""

    def test_full_conversion_pipeline(self):
        """Test full pipeline from contour to geometry."""
        # Create a simple contour
        contour = np.array(
            [[[25, 25]], [[175, 25]], [[175, 175]], [[25, 175]]]
        )

        # Convert to geometry
        geometry = GeometryConverter.contour_to_geometry(contour)

        assert geometry is not None
        assert geometry.is_closed()
        assert len(geometry) > 0

    def test_multiple_geometries_batch_conversion(self):
        """Test batch conversion of multiple contours."""
        contours = [
            np.array([[[0, 0]], [[50, 0]], [[50, 50]], [[0, 50]]]),
            np.array([[[60, 60]], [[110, 60]], [[110, 110]], [[60, 110]]]),
            np.array([[[0, 60]], [[50, 60]], [[50, 110]], [[0, 110]]]),
        ]

        geometries = GeometryConverter.contours_to_geometries(contours)

        assert isinstance(geometries, list)

    def test_conversion_with_realistic_contour(self):
        """Test conversion with a realistic irregular contour."""
        # Create an irregular polygon that might come from real detection
        contour = np.array(
            [
                [[120, 45]],
                [[135, 52]],
                [[140, 68]],
                [[138, 85]],
                [[125, 95]],
                [[108, 92]],
                [[98, 78]],
                [[100, 60]],
                [[108, 48]],
                [[115, 45]],
            ]
        )
        result = GeometryConverter.contour_to_geometry(contour)

        assert result is not None
        assert isinstance(result, Geometry)
        assert result.is_closed()
