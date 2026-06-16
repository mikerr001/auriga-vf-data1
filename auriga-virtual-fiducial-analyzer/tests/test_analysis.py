"""
test_analysis.py
----------------
Automated tests for the Auriga Virtual Fiducial Analyzer.

Run with:
  cd auriga-virtual-fiducial-analyzer
  python -m pytest tests/ -v
"""

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ------------------------------------------------------------------ helpers

def make_blank_jpg_bytes(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal JPEG byte string (white image) without cv2."""
    try:
        import cv2
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        return buf.tobytes()
    except ImportError:
        # fallback: 1×1 white JPEG (valid minimal JPEG)
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
            b"C  C\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
            b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa"
            b"\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n"
            b"\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZ"
            b"cdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95"
            b"\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3"
            b"\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca"
            b"\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7"
            b"\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00"
            b"\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00\x00\x00\x1f\xff\xd9"
        )


def make_aruco_jpg_bytes(marker_id: int = 0) -> bytes:
    """Create a JPEG with a visible 4x4_50 ArUco marker."""
    import cv2
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = np.full((300, 300, 3), 200, dtype=np.uint8)
    marker = np.zeros((100, 100), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, 100, marker)
    marker_rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    img[100:200, 100:200] = marker_rgb
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def make_csv(rows: list[dict]) -> str:
    cols = ["filename", "objectName", "distanceMeters", "cameraHeightCm",
            "deviceName", "orientation", "notes", "capturedAt", "sourceType"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


# ------------------------------------------------------------------ CSV validation

class TestCSVValidation:
    def test_valid_csv_passes(self):
        from analysis.pipeline import validate_csv
        df = pd.DataFrame([{
            "filename": "a.jpg", "distanceMeters": 1.0,
            "orientation": "Center", "cameraHeightCm": 50,
            "deviceName": "Cam",
        }])
        assert validate_csv(df) == []

    def test_missing_columns_flagged(self):
        from analysis.pipeline import validate_csv
        df = pd.DataFrame([{"filename": "a.jpg"}])
        errors = validate_csv(df)
        assert len(errors) > 0
        assert "distanceMeters" in errors[0]

    def test_all_required_columns_present(self):
        from analysis.pipeline import validate_csv, REQUIRED_CSV_COLUMNS
        df = pd.DataFrame(columns=list(REQUIRED_CSV_COLUMNS))
        assert validate_csv(df) == []


# ------------------------------------------------------------------ ZIP extraction

class TestZipExtraction:
    def test_extracts_supported_images(self, tmp_path):
        from analysis.pipeline import extract_zip
        zip_bytes = make_zip({
            "img1.jpg": make_blank_jpg_bytes(),
            "img2.png": make_blank_jpg_bytes(),
            "ignored.txt": b"not an image",
        })
        zip_path = tmp_path / "test.zip"
        zip_path.write_bytes(zip_bytes)
        extracted = extract_zip(zip_path, tmp_path / "out")
        names = {p.name for p in extracted}
        assert "img1.jpg" in names
        assert "img2.png" in names
        assert "ignored.txt" not in names

    def test_flattens_nested_dirs(self, tmp_path):
        from analysis.pipeline import extract_zip
        zip_bytes = make_zip({"subdir/nested.jpg": make_blank_jpg_bytes()})
        zip_path = tmp_path / "test.zip"
        zip_path.write_bytes(zip_bytes)
        extracted = extract_zip(zip_path, tmp_path / "out")
        assert extracted[0].name == "nested.jpg"


# ------------------------------------------------------------------ ArUco detection

class TestArucoDetection:
    def test_no_marker_returns_failure(self, tmp_path):
        from analysis.detector import detect_marker
        img_path = tmp_path / "blank.jpg"
        img_path.write_bytes(make_blank_jpg_bytes())
        result = detect_marker(img_path)
        assert result.success is False

    def test_missing_file_returns_failure(self, tmp_path):
        from analysis.detector import detect_marker
        result = detect_marker(tmp_path / "nonexistent.jpg")
        assert result.success is False
        assert "Could not read" in result.error

    def test_aruco_marker_detected(self, tmp_path):
        import cv2
        from analysis.detector import detect_marker
        img_path = tmp_path / "marker.jpg"
        img_path.write_bytes(make_aruco_jpg_bytes(marker_id=0))
        result = detect_marker(img_path)
        assert result.success is True
        assert result.marker_width_px > 0
        assert result.marker_height_px > 0
        assert result.marker_area_px > 0


# ------------------------------------------------------------------ Regression

class TestRegression:
    def _detected_df(self):
        distances = [0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0]
        widths    = [200, 195, 140, 145, 100, 98, 72, 75]
        return pd.DataFrame({
            "distanceMeters": distances,
            "markerWidthPx":  widths,
            "markerHeightPx": widths,
            "markerAreaPx":   [w**2 for w in widths],
            "detectionSuccess": [True] * 8,
        })

    def test_returns_metrics_for_detected_rows(self):
        from analysis.regression import fit_regression
        df = self._detected_df()
        metrics = fit_regression(df, "markerWidthPx")
        assert metrics is not None
        assert 0 <= metrics.r2 <= 1

    def test_insufficient_data_returns_none(self):
        from analysis.regression import fit_regression
        df = pd.DataFrame({
            "distanceMeters": [1.0],
            "markerWidthPx": [100.0],
            "detectionSuccess": [True],
        })
        assert fit_regression(df, "markerWidthPx") is None

    def test_all_regressions_run(self):
        from analysis.regression import run_all_regressions
        df = self._detected_df()
        results = run_all_regressions(df)
        assert "markerWidthPx" in results
        assert "markerHeightPx" in results
        assert "markerAreaPx" in results


# ------------------------------------------------------------------ Full pipeline

class TestPipeline:
    def _make_session(self, tmp_path, with_marker: bool = False):
        img_bytes = make_aruco_jpg_bytes(0) if with_marker else make_blank_jpg_bytes()
        zip_bytes = make_zip({"img001.jpg": img_bytes})
        csv_text = make_csv([{
            "filename": "img001.jpg", "objectName": "aruco",
            "distanceMeters": 1.0, "cameraHeightCm": 50,
            "deviceName": "Camera-1", "orientation": "Center",
            "notes": "", "capturedAt": "2026-06-16T08:00:00Z",
            "sourceType": "webcam",
        }])
        zip_path = tmp_path / "imgs.zip"
        csv_path = tmp_path / "meta.csv"
        zip_path.write_bytes(zip_bytes)
        csv_path.write_text(csv_text)
        return zip_path, csv_path

    def test_pipeline_runs_without_crash(self, tmp_path):
        from analysis.pipeline import run_pipeline
        zip_p, csv_p = self._make_session(tmp_path)
        result = run_pipeline(zip_p, csv_p, tmp_path / "session")
        assert "df" in result
        assert len(result["df"]) == 1

    def test_missing_image_produces_warning(self, tmp_path):
        from analysis.pipeline import run_pipeline
        zip_bytes = make_zip({"other.jpg": make_blank_jpg_bytes()})
        csv_text = make_csv([{
            "filename": "missing.jpg", "objectName": "aruco",
            "distanceMeters": 1.0, "cameraHeightCm": 50,
            "deviceName": "Cam", "orientation": "Center",
        }])
        (tmp_path / "t.zip").write_bytes(zip_bytes)
        (tmp_path / "t.csv").write_text(csv_text)
        result = run_pipeline(tmp_path / "t.zip", tmp_path / "t.csv", tmp_path / "s")
        assert any("missing.jpg" in w for w in result["warnings"])

    def test_invalid_csv_raises_pipeline_error(self, tmp_path):
        from analysis.pipeline import run_pipeline, PipelineError
        zip_bytes = make_zip({"img.jpg": make_blank_jpg_bytes()})
        (tmp_path / "t.zip").write_bytes(zip_bytes)
        (tmp_path / "bad.csv").write_text("col1,col2\nval1,val2\n")
        with pytest.raises(PipelineError):
            run_pipeline(tmp_path / "t.zip", tmp_path / "bad.csv", tmp_path / "s")

    def test_csv_and_results_exported(self, tmp_path):
        from analysis.pipeline import run_pipeline
        zip_p, csv_p = self._make_session(tmp_path)
        result = run_pipeline(zip_p, csv_p, tmp_path / "session")
        assert (result["export_dir"] / "analysis_results.csv").exists()

    def test_pdf_report_generated(self, tmp_path):
        from analysis.pipeline import run_pipeline
        zip_p, csv_p = self._make_session(tmp_path, with_marker=True)
        result = run_pipeline(zip_p, csv_p, tmp_path / "session")
        assert result["report_path"].exists()
        assert result["report_path"].stat().st_size > 1000
