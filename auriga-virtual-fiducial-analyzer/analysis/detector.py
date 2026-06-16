"""
detector.py
-----------
ArUco marker detection for the Auriga Virtual Fiducial Analyzer.

Detection strategy (applied in order until a marker is found):
  1. CLAHE-enhanced grayscale  + DICT_4X4_250, permissive params
  2. CLAHE on 2× upscaled img  + DICT_4X4_250, permissive params
  3. CLAHE-enhanced grayscale  + DICT_4X4_50,  permissive params

This cascade handles real-world webcam images with blur, low contrast,
and varying distances without introducing significant false positives.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------  setup


def _make_params() -> cv2.aruco.DetectorParameters:
    """Permissive detection params suited for real-world webcam images."""
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin    = 3
    p.adaptiveThreshWinSizeMax    = 53
    p.adaptiveThreshWinSizeStep   = 4
    p.minMarkerPerimeterRate      = 0.005
    p.maxMarkerPerimeterRate      = 4.0
    p.polygonalApproxAccuracyRate = 0.08
    p.cornerRefinementMethod      = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


_CLAHE  = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
_PARAMS = _make_params()

_DET_250 = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250), _PARAMS)
_DET_50  = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),  _PARAMS)


# ------------------------------------------------------------------ types

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
    strategy: str = ""
    error: Optional[str] = None


# ------------------------------------------------------------------ helpers

def _extract_geometry(corners_raw, scale: float = 1.0) -> tuple:
    """Return (width, height, area, cx, cy, corner_list) from raw corners."""
    pts = corners_raw[0].reshape(4, 2).astype(np.float32) / scale
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    w   = float(x_coords.max() - x_coords.min())
    h   = float(y_coords.max() - y_coords.min())
    area = float(cv2.contourArea(pts))
    cx  = float(x_coords.mean())
    cy  = float(y_coords.mean())
    return w, h, area, cx, cy, pts.tolist()


# ------------------------------------------------------------------ main API

def detect_marker(image_path: Path) -> DetectionResult:
    """
    Detect a single ArUco marker in the image at *image_path*.

    Applies a 3-pass cascade to handle challenging real-world images:
    native resolution → 2× upscale → fallback dict.

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

    # ---- pass 1: CLAHE at native resolution, DICT_4X4_250
    g1 = _CLAHE.apply(gray)
    corners, ids, _ = _DET_250.detectMarkers(g1)
    if ids is not None and len(ids) > 0:
        mw, mh, area, cx, cy, pts = _extract_geometry(corners)
        logger.debug("Pass-1 hit in %s", filename)
        return DetectionResult(filename=filename, success=True,
                               image_width=w, image_height=h,
                               marker_width_px=mw, marker_height_px=mh,
                               marker_area_px=area, center_x=cx, center_y=cy,
                               corners=pts, strategy="pass1-250-clahe")

    # ---- pass 2: CLAHE on 2× upscaled image, DICT_4X4_250
    gray2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    g2 = _CLAHE.apply(gray2)
    corners, ids, _ = _DET_250.detectMarkers(g2)
    if ids is not None and len(ids) > 0:
        mw, mh, area, cx, cy, pts = _extract_geometry(corners, scale=2.0)
        logger.debug("Pass-2 hit (2× upscale) in %s", filename)
        return DetectionResult(filename=filename, success=True,
                               image_width=w, image_height=h,
                               marker_width_px=mw, marker_height_px=mh,
                               marker_area_px=area, center_x=cx, center_y=cy,
                               corners=pts, strategy="pass2-250-2x")

    # ---- pass 3: CLAHE at native resolution, DICT_4X4_50 fallback
    corners, ids, _ = _DET_50.detectMarkers(g1)
    if ids is not None and len(ids) > 0:
        mw, mh, area, cx, cy, pts = _extract_geometry(corners)
        logger.debug("Pass-3 hit (DICT_4X4_50 fallback) in %s", filename)
        return DetectionResult(filename=filename, success=True,
                               image_width=w, image_height=h,
                               marker_width_px=mw, marker_height_px=mh,
                               marker_area_px=area, center_x=cx, center_y=cy,
                               corners=pts, strategy="pass3-50-clahe")

    logger.debug("All passes failed for %s", filename)
    return DetectionResult(filename=filename, success=False,
                           image_width=w, image_height=h,
                           error="No ArUco marker detected.")


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

    label = f"w={result.marker_width_px:.0f} h={result.marker_height_px:.0f} [{result.strategy}]"
    cv2.putText(img, label, (int(result.center_x) + 8, int(result.center_y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
