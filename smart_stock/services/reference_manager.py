"""Reference manager for storing and loading camera reference images."""

import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import yaml
from blinker import Signal

from rayforge.config import get_addon_data_dir
from ..models.reference_image import ReferenceImage

logger = logging.getLogger(__name__)


class ReferenceManager:
    """
    Manages reference images per camera with persistence to disk.

    References are stored in the user config directory:
    - Raw frames saved as PNG files for easy viewing/editing
    - Metadata stored in a YAML index file
    """

    def __init__(self):
        self._references: Dict[str, ReferenceImage] = {}
        self._storage_dir = get_addon_data_dir("smart_stock") / "references"
        self._index_file = self._storage_dir / "index.yaml"

        self.reference_added = Signal()
        self.reference_removed = Signal()
        self.reference_updated = Signal()

        self._ensure_storage_dir()
        self.load()

    def _ensure_storage_dir(self):
        """Create storage directory if it doesn't exist."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def get_reference(self, camera_id: str) -> Optional[ReferenceImage]:
        """
        Get the reference image for a specific camera.

        Args:
            camera_id: The unique identifier for the camera.

        Returns:
            The ReferenceImage if it exists, None otherwise.
        """
        return self._references.get(camera_id)

    def set_reference(self, camera_id: str, reference: ReferenceImage) -> None:
        """
        Set or update the reference image for a camera.

        Args:
            camera_id: The unique identifier for the camera.
            reference: The ReferenceImage to store.
        """
        logger.info(f"Setting reference for camera {camera_id}")
        is_new = camera_id not in self._references
        self._references[camera_id] = reference
        self._save_reference(camera_id, reference)
        self._save_index()

        if is_new:
            self.reference_added.send(self, camera_id=camera_id)
        else:
            self.reference_updated.send(self, camera_id=camera_id)

    def clear_reference(self, camera_id: str) -> bool:
        """
        Remove the reference image for a camera.

        Args:
            camera_id: The unique identifier for the camera.

        Returns:
            True if a reference was removed, False if none existed.
        """
        if camera_id not in self._references:
            return False

        del self._references[camera_id]
        self._delete_reference_files(camera_id)
        self._save_index()
        self.reference_removed.send(self, camera_id=camera_id)
        return True

    def _get_raw_frame_path(self, camera_id: str) -> Path:
        """Get the file path for a camera's raw frame."""
        return self._storage_dir / f"{camera_id}_raw.png"

    def _save_reference(
        self, camera_id: str, reference: ReferenceImage
    ) -> None:
        """Save a reference's raw frame to disk as PNG."""
        if reference.raw_frame is not None:
            raw_path = self._get_raw_frame_path(camera_id)
            try:
                cv2.imwrite(str(raw_path), reference.raw_frame)
                logger.debug(f"Saved raw frame PNG for camera {camera_id}")
            except Exception as e:
                logger.error(f"Failed to save raw frame for {camera_id}: {e}")
                raise

    def _delete_reference_files(self, camera_id: str) -> None:
        """Delete a reference's data files from disk."""
        raw_path = self._get_raw_frame_path(camera_id)
        try:
            if raw_path.exists():
                raw_path.unlink()
                logger.debug(f"Deleted raw frame for camera {camera_id}")
        except OSError as e:
            logger.error(f"Failed to delete raw frame for {camera_id}: {e}")

    def _save_index(self) -> None:
        """Save the reference index to disk."""
        index_data = {}
        for camera_id, ref in self._references.items():
            ref_data = {
                "capture_timestamp": ref.capture_timestamp,
                "calibration": ref.calibration,
            }
            if ref.physical_area is not None:
                (x_min, y_min), (x_max, y_max) = ref.physical_area
                ref_data["physical_area"] = {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                }
            if ref.output_size is not None:
                ref_data["output_size"] = list(ref.output_size)
            index_data[camera_id] = ref_data

        try:
            with open(self._index_file, "w") as f:
                yaml.safe_dump(index_data, f, default_flow_style=False)
            logger.debug("Saved reference index")
        except IOError as e:
            logger.error(f"Failed to save reference index: {e}")

    def load(self) -> None:
        """Load all references from disk."""
        if not self._index_file.exists():
            logger.debug("No reference index found, starting fresh")
            return

        try:
            with open(self._index_file, "r") as f:
                index_data = yaml.safe_load(f) or {}

            for camera_id, metadata in index_data.items():
                raw_path = self._get_raw_frame_path(camera_id)

                raw_frame = None

                if raw_path.exists():
                    try:
                        raw_frame = cv2.imread(str(raw_path))
                        if raw_frame is None:
                            raise ValueError("cv2.imread returned None")
                    except Exception as e:
                        logger.warning(
                            f"Failed to load raw frame for {camera_id}: {e}"
                        )

                if raw_frame is None:
                    logger.warning(
                        f"No reference data found for camera {camera_id}, "
                        "skipping"
                    )
                    continue

                physical_area = None
                if "physical_area" in metadata:
                    pa = metadata["physical_area"]
                    physical_area = (
                        (pa["x_min"], pa["y_min"]),
                        (pa["x_max"], pa["y_max"]),
                    )

                output_size = None
                if "output_size" in metadata:
                    output_size = tuple(metadata["output_size"])

                reference = ReferenceImage(
                    raw_frame=raw_frame,
                    capture_timestamp=metadata.get("capture_timestamp", 0.0),
                    camera_id=camera_id,
                    calibration=metadata.get("calibration"),
                    physical_area=physical_area,
                    output_size=output_size,
                )
                self._references[camera_id] = reference
                logger.debug(f"Loaded reference for camera {camera_id}")

            logger.info(
                f"Loaded {len(self._references)} reference(s) from disk"
            )
        except (yaml.YAMLError, IOError) as e:
            logger.error(f"Failed to load reference index: {e}")

    def get_all_camera_ids(self) -> list:
        """Get a list of all camera IDs with stored references."""
        ids = list(self._references.keys())
        logger.debug(f"get_all_camera_ids called, returning: {ids}")
        return ids

    def clear_all(self) -> None:
        """Remove all stored references."""
        for camera_id in list(self._references.keys()):
            self.clear_reference(camera_id)
