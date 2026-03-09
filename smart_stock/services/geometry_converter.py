"""Convert OpenCV contours to rayforge Geometry objects."""

import logging
from typing import List, Optional

import numpy as np

from rayforge.core.geo import Geometry

logger = logging.getLogger(__name__)


class GeometryConverter:
    """
    Converts OpenCV contours to rayforge Geometry objects.
    """

    @staticmethod
    def contour_to_geometry(contour: np.ndarray) -> Optional[Geometry]:
        """
        Convert a single OpenCV contour to a Geometry object.

        Args:
            contour: OpenCV contour array of shape (N, 1, 2) or None.

        Returns:
            Geometry object or or None if conversion fails.
        """
        if contour is None or len(contour) == 0:
            return None

        points = contour.squeeze()
        if len(points.shape) == 1:
            points = contour.reshape(-1, 2)

        if len(points.shape) != 2 or len(points) < 3:
            return None

        poly = [(float(p[0]), float(p[1])) for p in points]
        return Geometry.from_points(poly)

    @staticmethod
    def contours_to_geometries(contours: List[np.ndarray]) -> List[Geometry]:
        """
        Convert multiple OpenCV contours to Geometry objects.

        Args:
            contours: List of OpenCV contour arrays.

        Returns:
            List of Geometry objects.
        """
        geometries = []
        for contour in contours:
            geo = GeometryConverter.contour_to_geometry(contour)
            if geo is None:
                continue
            geometries.append(geo)
        return geometries
