"""Contour detection for stock items."""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from rayforge.core.geo import Edge, Point, Polygon, Rect
from rayforge.core.geo.contours import filter_to_external_contours
from rayforge.core.geo.polygon import (
    convex_hull,
    polygon_offset,
    polygon_union,
    polygon_area,
    polygon_centroid,
    polygon_bounds,
    polygon_perimeter,
    point_line_distance,
    extract_polygon_edges,
)
from rayforge.core.geo.smooth import smooth_polyline
from rayforge.image.tracing import trace_color_image

logger = logging.getLogger(__name__)


@dataclass
class ContourConfig:
    """Configuration for contour detection."""

    min_contour_area: float = 500.0
    max_contour_area: float = 10000000.0
    max_items: int = 10
    merge_distance: float = 10.0
    use_convex_hull: bool = False
    smoothing_amount: int = 0
    corner_angle_threshold: float = 120.0


@dataclass
class DetectedContour:
    """Represents a detected contour with metadata."""

    points: np.ndarray
    area: float
    centroid: Point
    bounding_rect: Rect


class ContourDetector:
    """
    Detects and filters contours in images.

    Uses vtracer for high-quality vectorization of color images.
    """

    def __init__(self, config: Optional[ContourConfig] = None):
        self.config = config or ContourConfig()

    def detect_contours(
        self,
        current: np.ndarray,
        reference: Optional[np.ndarray] = None,
        sensitivity: float = 50.0,
    ) -> List[DetectedContour]:
        """
        Detect contours in an image using vtracer.

        Args:
            current: Current BGR color image from camera.
            reference: Optional reference BGR image of empty machine.
            sensitivity: Detection sensitivity (0-100). Higher values detect
                         more changes. 0=low sensitivity, 100=high sensitivity.

        Returns:
            List of DetectedContour objects.
        """
        if current is None or current.size == 0:
            return []

        if reference is not None:
            return self._detect_with_reference(current, reference, sensitivity)

        return self._detect_without_reference(current)

    def _detect_with_reference(
        self,
        current: np.ndarray,
        reference: np.ndarray,
        sensitivity: float,
    ) -> List[DetectedContour]:
        """
        Detect contours by comparing current to reference.

        Traces both images and removes edges from current polygons
        that exist in reference, then returns remaining shapes.
        """
        logger.debug("Tracing reference image with vtracer")
        ref_geometries = trace_color_image(reference)
        logger.debug(f"Reference: {len(ref_geometries)} geometries")

        logger.debug("Tracing current image with vtracer")
        curr_geometries = trace_color_image(current)
        logger.debug(f"Current: {len(curr_geometries)} geometries")

        if not curr_geometries:
            return []

        tolerance = (100 - sensitivity) / 2.5 + 1
        logger.debug(f"Edge matching tolerance: {tolerance}")

        ref_edges = []
        for geo in ref_geometries:
            for poly in geo.to_polygons():
                if poly and len(poly) >= 2:
                    ref_edges.extend(extract_polygon_edges(poly))
        logger.debug(f"Reference has {len(ref_edges)} edges")

        curr_polygons = []
        for geo in curr_geometries:
            for poly in geo.to_polygons():
                if poly and len(poly) >= 3:
                    curr_polygons.append(poly)

        logger.debug(f"Current has {len(curr_polygons)} polygons")

        new_polygons = []
        for poly in curr_polygons:
            if self._polygon_has_significant_new_edges(
                poly, ref_edges, tolerance
            ):
                new_polygons.append(poly)

        logger.debug(
            f"Polygons with significant new edges: {len(new_polygons)}"
        )

        if not new_polygons:
            return []

        solid_polygons = []
        for poly in new_polygons:
            if len(poly) >= 3:
                if self._is_solid_polygon(poly):
                    solid_polygons.append(poly)

        logger.debug(f"After solid filter: {len(solid_polygons)} polygons")

        if not solid_polygons:
            return []

        if self.config.merge_distance > 0:
            logger.debug(
                f"Merging {len(solid_polygons)} polygons "
                f"with distance {self.config.merge_distance}"
            )
            solid_polygons = self._merge_nearby_polygons(
                solid_polygons, self.config.merge_distance
            )
            logger.debug(f"After merge: {len(solid_polygons)} polygons")

        if self.config.use_convex_hull:
            solid_polygons = [convex_hull(p) for p in solid_polygons]

        if self.config.smoothing_amount > 0:
            solid_polygons = [self._smooth_polygon(p) for p in solid_polygons]

        results = []
        for i, poly in enumerate(solid_polygons):
            if poly and len(poly) >= 3:
                detected = self._polygon_to_detected(poly)
                if detected:
                    results.append(detected)
                    logger.debug(f"Polygon {i}: {len(poly)} pts - Accepted")

        results.sort(key=lambda c: c.area, reverse=True)
        results = results[: self.config.max_items]

        logger.debug(
            f"_detect_with_reference: returning {len(results)} contours"
        )
        return results

    def _polygon_has_significant_new_edges(
        self,
        polygon: Polygon,
        ref_edges: List[Edge],
        tolerance: float,
    ) -> bool:
        """
        Check if polygon has a significant number of new edges.

        A polygon is considered "new" if more than half of its edges
        don't match any reference edge.
        """
        if len(polygon) < 3:
            return False

        n = len(polygon)
        new_edge_count = 0

        for i in range(n):
            start = polygon[i]
            end = polygon[(i + 1) % n]

            ref_idx = self._find_shared_ref_edge(
                start, end, ref_edges, tolerance
            )
            if ref_idx is None:
                new_edge_count += 1

        return new_edge_count > n / 2

    def _is_solid_polygon(self, polygon: Polygon) -> bool:
        """
        Check if polygon is solid (not a thin strip).

        Uses area-to-perimeter ratio. Thin strips have high
        perimeter relative to their area.
        """
        if len(polygon) < 3:
            return False

        area = abs(polygon_area(polygon))
        if area < 1.0:
            return False

        perimeter = polygon_perimeter(polygon)
        if perimeter < 1.0:
            return False

        compactness = area / perimeter

        return compactness >= 3.0

    def _find_shared_ref_edge(
        self,
        p1: Point,
        p2: Point,
        ref_edges: List[Edge],
        tolerance: float,
    ) -> Optional[int]:
        """
        Find a reference edge that both points are near.

        Returns the index of the reference edge, or None if no single
        edge is near both points.
        """
        for idx, (ref_start, ref_end) in enumerate(ref_edges):
            dist1 = point_line_distance(p1, ref_start, ref_end)
            dist2 = point_line_distance(p2, ref_start, ref_end)

            if dist1 <= tolerance and dist2 <= tolerance:
                return idx

        return None

    def _detect_without_reference(
        self, image: np.ndarray
    ) -> List[DetectedContour]:
        """Detect contours without reference comparison."""
        geometries = trace_color_image(image)

        if not geometries:
            logger.debug("No geometries found by tracer")
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

        if self.config.smoothing_amount > 0:
            polygons = [self._smooth_polygon(p) for p in polygons]

        results = []
        for poly in polygons:
            if poly and len(poly) >= 3:
                detected = self._polygon_to_detected(poly)
                if detected:
                    results.append(detected)

        results.sort(key=lambda c: c.area, reverse=True)
        results = results[: self.config.max_items]

        return results

    def _polygon_to_detected(
        self, polygon: Polygon
    ) -> Optional[DetectedContour]:
        """Convert polygon back to DetectedContour."""
        if not polygon or len(polygon) < 3:
            return None

        points = np.array(polygon, dtype=np.float32)
        area = abs(polygon_area(polygon))

        if area < self.config.min_contour_area:
            return None

        if area > self.config.max_contour_area:
            return None

        centroid = polygon_centroid(polygon)
        min_x, min_y, max_x, max_y = polygon_bounds(polygon)
        bounding_rect = (min_x, min_y, max_x - min_x, max_y - min_y)

        return DetectedContour(
            points=points,
            area=area,
            centroid=centroid,
            bounding_rect=bounding_rect,
        )

    def _merge_nearby_polygons(
        self, polygons: List[Polygon], distance: float
    ) -> List[Polygon]:
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

    def _smooth_polygon(self, polygon: Polygon) -> Polygon:
        """
        Smooth a polygon using Gaussian filtering from the geo module.

        Angles sharper than the configured threshold are preserved as corners.

        Args:
            polygon: List of (x, y) points.

        Returns:
            Smoothed polygon.
        """
        if len(polygon) < 3 or self.config.smoothing_amount == 0:
            return polygon

        points_3d = [(p[0], p[1], 0.0) for p in polygon]

        smoothed_3d = smooth_polyline(
            points_3d,
            self.config.smoothing_amount,
            self.config.corner_angle_threshold,
            is_closed=True,
        )

        if not smoothed_3d:
            return polygon

        return [(p[0], p[1]) for p in smoothed_3d]
