"""
detector.py
-----------
ArUco marker detection for the Auriga Virtual Fiducial Analyzer.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ARUCO_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class DetectionResult:
    filename: str
    success: bool
    image_width: int = 0
    image_height: int = 0
    marker_width_px: float = 0.0
    marker_height_px: float = 0.0
    marker_area_px: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0
    corners: list = field(default_factory=list)
    error: Optional[str] = None


def _get_detector() -> cv2.aruco.ArucoDetector:
    """Build and return an ArUco detector for DICT_4X4_50."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


_DETECTOR = _get_detector()


def detect_marker(image_path: Path) -> DetectionResult:
    """
    Detect a single ArUco marker in the image at *image_path*.

    Returns a DetectionResult with geometry fields populated on success
    or success=False with an error message on failure.
    """
    filename = image_path.name

    img = cv2.imread(str(image_path))
    if img is None:
        return DetectionResult(filename=filename, success=False,
                               error="Could not read image file.")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = _DETECTOR.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        logger.debug("No marker found in %s", filename)
        return DetectionResult(filename=filename, success=False,
                               image_width=w, image_height=h,
                               error="No ArUco marker detected.")

    # Use the first detected marker
    pts = corners[0].reshape(4, 2).astype(np.float32)
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]

    marker_w = float(x_coords.max() - x_coords.min())
    marker_h = float(y_coords.max() - y_coords.min())
    area = float(cv2.contourArea(pts))
    cx = float(x_coords.mean())
    cy = float(y_coords.mean())

    logger.debug("Detected marker in %s — w=%.1f h=%.1f", filename, marker_w, marker_h)

    return DetectionResult(
        filename=filename,
        success=True,
        image_width=w,
        image_height=h,
        marker_width_px=marker_w,
        marker_height_px=marker_h,
        marker_area_px=area,
        center_x=cx,
        center_y=cy,
        corners=pts.tolist(),
    )


def annotate_image(image_path: Path, result: DetectionResult, out_path: Path) -> None:
    """
    Save a copy of *image_path* with the detected marker boundary drawn on it.
    Does nothing if detection was unsuccessful.
    """
    if not result.success:
        return

    img = cv2.imread(str(image_path))
    if img is None:
        return

    pts = np.array(result.corners, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.circle(img, (int(result.center_x), int(result.center_y)), 5, (0, 0, 255), -1)

    label = f"w={result.marker_width_px:.0f} h={result.marker_height_px:.0f}"
    cv2.putText(img, label, (int(result.center_x) + 8, int(result.center_y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
