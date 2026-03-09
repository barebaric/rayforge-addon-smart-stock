"""Contour detection for stock items."""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from rayforge.core.geo import Geometry
from rayforge.core.geo.contours import filter_to_external_contours
from rayforge.core.geo.polygon import (
    convex_hull,
    polygon_offset,
    polygon_union,
    polygon_area,
    polygon_centroid,
)


@dataclass
class ContourConfig:
    """Configuration for contour detection."""

    min_contour_area: float = 500.0
    max_contour_area: float = 10000000.0
    simplify_epsilon: float = 1.0
    min_solidity: float = 0.2
    min_vertices: int = 3
    max_items: int = 10
    merge_distance: float = 10.0
    use_convex_hull: bool = False


@dataclass
class DetectedContour:
    """Represents a detected contour with metadata."""

    points: np.ndarray
    area: float
    centroid: tuple
    bounding_rect: tuple


class ContourDetector:
    """
    Detects and filters contours in binary masks.

    Identifies potential stock items by finding external contours
    and filtering them based on geometric properties.
    """

    def __init__(self, config: Optional[ContourConfig] = None):
        self.config = config or ContourConfig()

    def detect_contours(self, mask: np.ndarray) -> List[DetectedContour]:
        if mask is None or mask.size == 0:
            return []

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        geometries = []
        for contour in contours:
            detected = self._process_contour(contour)
            if detected:
                points = self._extract_points(detected.points)
                if points and len(points) >= 3:
                    geo = Geometry.from_points(points)
                    if not geo.is_empty():
                        geometries.append(geo)

        if not geometries:
            return []

        external_geometries = filter_to_external_contours(geometries)

        polygons = []
        for geo in external_geometries:
            for poly in geo.to_polygons():
                if poly and len(poly) >= 3:
                    polygons.append(poly)

        if not polygons:
            return []

        if self.config.merge_distance > 0:
            polygons = self._merge_nearby_polygons(
                polygons, self.config.merge_distance
            )

        if self.config.use_convex_hull:
            polygons = [convex_hull(p) for p in polygons]

        results = []
        for poly in polygons:
            if poly and len(poly) >= 3:
                detected = self._polygon_to_detected(poly)
                if detected:
                    results.append(detected)

        results.sort(key=lambda c: c.area, reverse=True)
        results = results[: self.config.max_items]

        return results

    def _extract_points(self, points: np.ndarray) -> Optional[List[tuple]]:
        """Extract 2D points from OpenCV contour array."""
        if points is None or len(points) < 3:
            return None

        if points.ndim == 3:
            points = points.squeeze()

        if len(points) < 3:
            return None

        return [(float(p[0]), float(p[1])) for p in points]

    def _polygon_to_detected(
        self, polygon: List[tuple]
    ) -> Optional[DetectedContour]:
        """Convert polygon back to DetectedContour."""
        if not polygon or len(polygon) < 3:
            return None

        points = np.array(polygon, dtype=np.float32)
        area = abs(polygon_area(polygon))

        if area < self.config.min_contour_area:
            return None

        centroid = polygon_centroid(polygon)
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        bounding_rect = (x, y, w, h)

        return DetectedContour(
            points=points,
            area=area,
            centroid=centroid,
            bounding_rect=bounding_rect,
        )

    def _merge_nearby_polygons(
        self, polygons: List[List[tuple]], distance: float
    ) -> List[List[tuple]]:
        """Expand polygons, union overlapping ones, then shrink back."""
        if not polygons:
            return []

        expanded = []
        for poly in polygons:
            offset_polys = polygon_offset(poly, distance)
            expanded.extend(offset_polys)

        merged = polygon_union(expanded)

        result = []
        for poly in merged:
            shrunk = polygon_offset(poly, -distance)
            if shrunk:
                result.extend(shrunk)

        return result

    def _process_contour(
        self, contour: np.ndarray
    ) -> Optional[DetectedContour]:
        """Process and filter a single contour."""
        area = cv2.contourArea(contour)
        if area < self.config.min_contour_area:
            return None
        if area > self.config.max_contour_area:
            return None

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < self.config.min_solidity:
                return None

        simplified = self._simplify_contour(contour)
        if simplified is None or len(simplified) < self.config.min_vertices:
            return None

        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            centroid = (cx, cy)
        else:
            centroid = (0, 0)

        x, y, w, h = cv2.boundingRect(contour)
        bounding_rect = (x, y, w, h)

        return DetectedContour(
            points=simplified,
            area=area,
            centroid=centroid,
            bounding_rect=bounding_rect,
        )

    def _simplify_contour(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """Simplify contour using Douglas-Peucker algorithm."""
        epsilon = self.config.simplify_epsilon
        simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(simplified) < 3:
            return None
        return simplified.squeeze()

    def create_mask_from_contours(
        self,
        contours: List[DetectedContour],
        shape: tuple,
    ) -> np.ndarray:
        """
        Create a binary mask from detected contours.

        Args:
            contours: List of DetectedContour objects.
            shape: Output mask shape (height, width).

        Returns:
            Binary mask with contours filled.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        for detected in contours:
            points = detected.points.astype(np.int32)
            cv2.drawContours(mask, [points], -1, (255,), cv2.FILLED)
        return mask


def contour_to_geometry(detected: DetectedContour) -> "Geometry":
    """
    Convert a DetectedContour to a Geometry object.

    Args:
        detected: A DetectedContour instance.

    Returns:
        A Geometry object containing the contour as a closed polygon.
    """
    points = detected.points

    if points is None or len(points) < 3:
        return Geometry()

    if points.ndim == 1:
        points = points.reshape(-1, 2)

    poly = [(float(p[0]), float(p[1])) for p in points]
    return Geometry.from_points(poly)
