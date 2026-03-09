"""Reference image model for storing."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ReferenceImage:
    """Stores reference image data for stock detection."""

    raw_frame: Optional[np.ndarray] = None
    capture_timestamp: float = 0.0
    camera_id: str = ""
    calibration: Optional[dict] = None
    physical_area: Optional[
        Tuple[Tuple[float, float], Tuple[float, float]]
    ] = None
    output_size: Optional[Tuple[int, int]] = None

    def to_dict(self):
        return {
            "raw_frame": self.raw_frame,
            "capture_timestamp": self.capture_timestamp,
            "camera_id": self.camera_id,
            "calibration": self.calibration,
            "physical_area": self.physical_area,
            "output_size": self.output_size,
        }
