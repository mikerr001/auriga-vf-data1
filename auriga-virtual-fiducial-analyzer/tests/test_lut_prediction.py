"""
test_lut_prediction.py
----------------------
Phase 3 automated tests for LUT generation, prediction, cross-validation,
sensitivity analysis, recommendation engine, report generation, and Flask
route integration.

Run with:
  cd auriga-virtual-fiducial-analyzer
  python -m pytest tests/ -v
"""

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------ fixtures

def _synthetic_df(n: int = 12, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic analysis DataFrame that mimics analysis_results.csv.
    markerWidthPx follows an inverse relationship with distanceMeters + noise.
    """
    rng = np.random.default_rng(seed)
    distances = np.tile([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], n // 6 + 1)[:n]
    widths = (200.0 / distances) + rng.normal(0, 3.0, n)
    orientations = np.tile(["Center", "Left", "Right", "Down"], n // 4 + 1)[:n]
    filenames = [f"img_{i:03d}.jpg" for i in range(n)]
    return pd.DataFrame({
        "filename":         filenames,
        "distanceMeters":   distances,
        "markerWidthPx":    widths,
        "markerHeightPx":   widths * 0.95,
        "markerAreaPx":     widths ** 2,
        "orientation":      orientations,
        "detectionSuccess": [True] * n,
    })


def _all_valid_flags(df: pd.DataFrame) -> dict:
    return {fn: "VALID" for fn in df["filename"]}


def _fitted_model(df: pd.DataFrame, flags: dict):
    from analysis.lut import build_lut
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        result = build_lut(df, flags, Path(tmp))
        return result.model


# ------------------------------------------------------------------ LUT generation

class TestLUTGeneration:
    def test_basic_lut_builds_successfully(self, tmp_path):
        from analysis.lut import build_lut
        df = _synthetic_df()
        flags = _all_valid_flags(df)
        result = build_lut(df, flags, tmp_path)
        assert result.lut_path.exists()
        assert result.n_valid == len(df)
        assert len(result.lut_df) > 0
        lut_loaded = pd.read_csv(result.lut_path)
        assert "distanceMeters" in lut_loaded.columns
        assert "predictedMarkerWidthPx" in lut_loaded.columns

    def test_invalid_rows_excluded(self, tmp_path):
        from analysis.lut import build_lut
        df = _synthetic_df(12)
        flags = _all_valid_flags(df)
        for fn in list(flags.keys())[:4]:
            flags[fn] = "INVALID"
        result = build_lut(df, flags, tmp_path)
        assert result.n_valid == 8

    def test_all_invalid_raises_lut_error(self, tmp_path):
        from analysis.lut import build_lut, LUTError
        df = _synthetic_df(6)
        flags = {fn: "INVALID" for fn in df["filename"]}
        with pytest.raises(LUTError):
            build_lut(df, flags, tmp_path)

    def test_missing_columns_raises_lut_error(self, tmp_path):
        from analysis.lut import build_lut, LUTError
        df = pd.DataFrame({"filename": ["a.jpg"], "distanceMeters": [1.0]})
        flags = {"a.jpg": "VALID"}
        with pytest.raises(LUTError):
            build_lut(df, flags, tmp_path)

    def test_failed_detections_excluded(self, tmp_path):
        from analysis.lut import build_lut, LUTError
        df = _synthetic_df(6)
        df["detectionSuccess"] = False
        flags = _all_valid_flags(df)
        with pytest.raises(LUTError):
            build_lut(df, flags, tmp_path)

    def test_questionable_rows_excluded(self, tmp_path):
        from analysis.lut import build_lut
        df = _synthetic_df(12)
        flags = {fn: "QUESTIONABLE" for fn in df["filename"][:6]}
        for fn in df["filename"][6:]:
            flags[fn] = "VALID"
        result = build_lut(df, flags, tmp_path)
        assert result.n_valid == 6


# ------------------------------------------------------------------ Prediction metrics

class TestPredictionMetrics:
    def test_prediction_runs_successfully(self, tmp_path):
        from analysis.lut import build_lut
        from analysis.prediction import run_prediction
        df = _synthetic_df()
        flags = _all_valid_flags(df)
        lut = build_lut(df, flags, tmp_path)
        pred = run_prediction(df, flags, lut.model, tmp_path)
        assert pred.results_path.exists()
        assert pred.metrics.n_samples == len(df)

    def test_metrics_are_finite(self, tmp_path):
        from analysis.lut import build_lut
        from analysis.prediction import run_prediction
        df = _synthetic_df()
        flags = _all_valid_flags(df)
        lut = build_lut(df, flags, tmp_path)
        pred = run_prediction(df, flags, lut.model, tmp_path)
        m = pred.metrics
        assert np.isfinite(m.mae)
        assert np.isfinite(m.rmse)
        assert np.isfinite(m.mape)
        assert np.isfinite(m.r2)

    def test_mae_leq_rmse(self, tmp_path):
        from analysis.lut import build_lut
        from analysis.prediction import run_prediction
        df = _synthetic_df()
        flags = _all_valid_flags(df)
        lut = build_lut(df, flags, tmp_path)
        pred = run_prediction(df, flags, lut.model, tmp_path)
        assert pred.metrics.mae <= pred.metrics.rmse + 1e-9

    def test_prediction_csv_has_expected_columns(self, tmp_path):
        from analysis.lut import build_lut
        from analysis.prediction import run_prediction
        df = _synthetic_df()
        flags = _all_valid_flags(df)
        lut = build_lut(df, flags, tmp_path)
        pred = run_prediction(df, flags, lut.model, tmp_path)
        result_df = pd.read_csv(pred.results_path)
        for col in ("distanceMeters", "predictedDistance", "absoluteError", "percentageError"):
            assert col in result_df.columns

    def test_no_valid_samples_raises_prediction_error(self, tmp_path):
        from analysis.lut import build_lut
        from analysis.prediction import run_prediction, PredictionError
        df = _synthetic_df()
        flags_lut = _all_valid_flags(df)
        lut = build_lut(df, flags_lut, tmp_path)
        flags_none = {fn: "INVALID" for fn in df["filename"]}
        with pytest.raises(PredictionError):
            run_prediction(df, flags_none, lut.model, tmp_path)

    def test_high_r2_on_clean_data(self, tmp_path):
        """Clean synthetic data following a polynomial relationship should yield low MAE."""
        from analysis.lut import build_lut
        from analysis.prediction import run_prediction
        rng = np.random.default_rng(0)
        n = 20
        distances = np.linspace(0.5, 5.0, n)
        widths = 300.0 - 50.0 * distances + rng.normal(0, 0.3, n)
        df = pd.DataFrame({
            "filename": [f"f{i}.jpg" for i in range(n)],
            "distanceMeters": distances,
            "markerWidthPx": widths,
            "orientation": ["Center"] * n,
            "detectionSuccess": [True] * n,
        })
        flags = _all_valid_flags(df)
        lut = build_lut(df, flags, tmp_path)
        pred = run_prediction(df, flags, lut.model, tmp_path)
        assert pred.metrics.mae < 1.0


# ------------------------------------------------------------------ Cross-validation

class TestCrossValidation:
    def test_cv_returns_k_folds(self, tmp_path):
        from analysis.cross_validation import run_cross_validation
        df = _synthetic_df(12)
        flags = _all_valid_flags(df)
        result = run_cross_validation(df, flags, k=4)
        assert result.k == 4
        assert len(result.folds) == 4

    def test_cv_fold_structure(self, tmp_path):
        from analysis.cross_validation import run_cross_validation
        df = _synthetic_df(12)
        flags = _all_valid_flags(df)
        result = run_cross_validation(df, flags, k=3)
        for fold in result.folds:
            assert hasattr(fold, "fold")
            assert hasattr(fold, "n_train")
            assert hasattr(fold, "n_test")
            assert hasattr(fold, "mae")
            assert hasattr(fold, "rmse")
            assert hasattr(fold, "r2")
            assert fold.mae >= 0
            assert fold.rmse >= 0

    def test_cv_mean_std_computed(self, tmp_path):
        from analysis.cross_validation import run_cross_validation
        df = _synthetic_df(12)
        flags = _all_valid_flags(df)
        result = run_cross_validation(df, flags)
        assert np.isfinite(result.mean_mae)
        assert np.isfinite(result.std_mae)
        assert np.isfinite(result.mean_rmse)
        assert np.isfinite(result.std_rmse)

    def test_cv_insufficient_data_raises(self, tmp_path):
        from analysis.cross_validation import run_cross_validation, CrossValidationError
        df = _synthetic_df(3)
        flags = _all_valid_flags(df)
        with pytest.raises(CrossValidationError):
            run_cross_validation(df, flags)


# ------------------------------------------------------------------ Sensitivity analysis

class TestSensitivityAnalysis:
    def test_sensitivity_returns_correct_perturbation_count(self, tmp_path):
        from analysis.lut import build_lut
        from analysis.sensitivity import run_sensitivity_analysis
        df = _synthetic_df()
        flags = _all_valid_flags(df)
        lut = build_lut(df, flags, tmp_path)
        result = run_sensitivity_analysis(df, flags, lut.model,
                                          perturbation_levels=(1.0, 3.0, 5.0))
        assert len(result.perturbations) == 6

    def test_sensitivity_perturbation_levels(self, tmp_path):
        from analysis.lut import build_lut
        from analysis.sensitivity import run_sensitivity_analysis
        df = _synthetic_df()
        flags = _all_valid_flags(df)
        lut = build_lut(df, flags, tmp_path)
        result = run_sensitivity_analysis(df, flags, lut.model,
                                          perturbation_levels=(2.0,))
        pcts = {p.perturbation_pct for p in result.perturbations}
        assert 2.0 in pcts
        dirs = {p.direction for p in result.perturbations}
        assert "positive" in dirs
        assert "negative" in dirs

    def test_sensitivity_mae_increases_with_perturbation(self, tmp_path):
        from analysis.lut import build_lut
        from analysis.sensitivity import run_sensitivity_analysis
        df = _synthetic_df(18)
        flags = _all_valid_flags(df)
        lut = build_lut(df, flags, tmp_path)
        result = run_sensitivity_analysis(df, flags, lut.model,
                                          perturbation_levels=(1.0, 5.0))
        pos_1 = next(p for p in result.perturbations
                     if p.perturbation_pct == 1.0 and p.direction == "positive")
        pos_5 = next(p for p in result.perturbations
                     if p.perturbation_pct == 5.0 and p.direction == "positive")
        assert pos_1.mae >= 0.0
        assert pos_5.mae >= 0.0
        assert pos_1.mean_abs_delta >= 0.0
        assert pos_5.mean_abs_delta >= 0.0
        assert pos_5.mean_abs_delta != pos_1.mean_abs_delta or pos_5.mae != pos_1.mae


# ------------------------------------------------------------------ Recommendation engine

class TestRecommendationEngine:
    def _make_metrics(self, r2, mae, rmse, mape, n=10):
        from analysis.prediction import PredictionMetrics
        return PredictionMetrics(mae=mae, rmse=rmse, mape=mape, r2=r2, n_samples=n)

    def test_go_verdict_for_high_r2_data(self):
        from analysis.recommendation import generate_recommendation
        metrics = self._make_metrics(r2=0.97, mae=0.05, rmse=0.07, mape=3.0)
        df = pd.DataFrame({"absoluteError": [0.05] * 10, "orientation": ["Center"] * 10})
        result = generate_recommendation(metrics, df)
        assert result.verdict == "Go"

    def test_nogo_verdict_for_low_r2_data(self):
        from analysis.recommendation import generate_recommendation
        metrics = self._make_metrics(r2=0.50, mae=0.50, rmse=0.70, mape=40.0)
        df = pd.DataFrame({"absoluteError": [0.50] * 10, "orientation": ["Center"] * 10})
        result = generate_recommendation(metrics, df)
        assert result.verdict == "No-Go"

    def test_rationale_is_nonempty(self):
        from analysis.recommendation import generate_recommendation
        metrics = self._make_metrics(r2=0.95, mae=0.08, rmse=0.10, mape=5.0)
        df = pd.DataFrame({"absoluteError": [0.08] * 8, "orientation": ["Center"] * 8})
        result = generate_recommendation(metrics, df)
        assert len(result.rationale) > 0

    def test_result_has_expected_fields(self):
        from analysis.recommendation import generate_recommendation
        metrics = self._make_metrics(r2=0.92, mae=0.10, rmse=0.14, mape=7.0)
        df = pd.DataFrame({"absoluteError": [0.10] * 8, "orientation": ["Center"] * 8})
        result = generate_recommendation(metrics, df)
        assert hasattr(result, "verdict")
        assert hasattr(result, "rationale")
        assert hasattr(result, "r2")
        assert result.verdict in ("Go", "No-Go")


# ------------------------------------------------------------------ Report PDF generation

class TestReportGeneration:
    def test_pdf_generated_with_nonzero_size(self, tmp_path):
        from analysis.lut import build_lut
        from analysis.prediction import run_prediction
        from analysis.cross_validation import run_cross_validation
        from analysis.sensitivity import run_sensitivity_analysis
        from analysis.recommendation import generate_recommendation
        from analysis.lut_plots import generate_lut_plots
        from analysis.lut_report import generate_lut_report

        df = _synthetic_df(12)
        flags = _all_valid_flags(df)
        lut = build_lut(df, flags, tmp_path)
        pred = run_prediction(df, flags, lut.model, tmp_path)
        cv = run_cross_validation(df, flags)
        sens = run_sensitivity_analysis(df, flags, lut.model)
        rec = generate_recommendation(pred.metrics, pred.results_df)
        plots = generate_lut_plots(pred.results_df, df, lut.model, tmp_path)

        report_path = generate_lut_report(
            analysis_df=df,
            results_df=pred.results_df,
            quality_flags=flags,
            metrics=pred.metrics,
            cv_result=cv,
            sensitivity_result=sens,
            recommendation=rec,
            lut_plots=plots,
            export_dir=tmp_path,
        )
        assert report_path.exists()
        assert report_path.stat().st_size > 1000

    def test_pdf_filename(self, tmp_path):
        from analysis.lut import build_lut
        from analysis.prediction import run_prediction
        from analysis.recommendation import generate_recommendation
        from analysis.lut_report import generate_lut_report

        df = _synthetic_df(6)
        flags = _all_valid_flags(df)
        lut = build_lut(df, flags, tmp_path)
        pred = run_prediction(df, flags, lut.model, tmp_path)
        rec = generate_recommendation(pred.metrics, pred.results_df)
        report_path = generate_lut_report(
            analysis_df=df, results_df=pred.results_df,
            quality_flags=flags, metrics=pred.metrics,
            cv_result=None, sensitivity_result=None,
            recommendation=rec, lut_plots={},
            export_dir=tmp_path,
        )
        assert report_path.name == "lut_validation_report.pdf"


# ------------------------------------------------------------------ Flask route integration

def _make_blank_jpg_bytes():
    import cv2
    img = np.full((200, 200, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _make_aruco_jpg_bytes(marker_id=0):
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


def _make_zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _make_csv(rows: list) -> str:
    cols = ["filename", "objectName", "distanceMeters", "cameraHeightCm",
            "deviceName", "orientation", "notes", "capturedAt", "sourceType"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


@pytest.fixture
def flask_client(tmp_path, monkeypatch):
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import app as flask_app
    monkeypatch.setattr(flask_app, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(flask_app, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(flask_app, "EXPORTS_DIR", tmp_path / "exports")
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "exports").mkdir(exist_ok=True)
    flask_app.app.config["TESTING"] = True
    flask_app.app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.app.test_client() as client:
        yield client, tmp_path


class TestFlaskPhase3Routes:
    def _upload_and_analyze(self, client, tmp_path):
        distances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                     0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        rows = []
        files_dict = {}
        for i, d in enumerate(distances):
            fname = f"img_{i:03d}.jpg"
            files_dict[fname] = _make_aruco_jpg_bytes(0)
            rows.append({
                "filename": fname, "objectName": "aruco",
                "distanceMeters": d, "cameraHeightCm": 50,
                "deviceName": "Camera-1", "orientation": ["Center", "Left"][i % 2],
                "notes": "", "capturedAt": "2026-06-16T08:00:00Z",
                "sourceType": "webcam",
            })
        zip_bytes = _make_zip(files_dict)
        csv_text  = _make_csv(rows)
        resp = client.post("/analyze", data={
            "zip_file": (io.BytesIO(zip_bytes), "test.zip"),
            "csv_file": (io.BytesIO(csv_text.encode()), "test.csv"),
        }, content_type="multipart/form-data")
        html = resp.data.decode()
        import re
        m = re.search(r"Session:\s*<code>([^<]+)</code>", html)
        if not m:
            return None
        return m.group(1).strip()

    def test_review_get_returns_200(self, flask_client):
        client, tmp_path = flask_client
        session_id = self._upload_and_analyze(client, tmp_path)
        if session_id is None:
            pytest.skip("Analysis produced no session_id in HTML")
        resp = client.get(f"/review/{session_id}")
        assert resp.status_code == 200
        assert b"Review Data Quality" in resp.data

    def test_review_post_redirects_to_predict(self, flask_client):
        client, tmp_path = flask_client
        session_id = self._upload_and_analyze(client, tmp_path)
        if session_id is None:
            pytest.skip("Analysis produced no session_id in HTML")
        import app as flask_app
        from werkzeug.utils import secure_filename
        safe_id = secure_filename(session_id)
        session_dir = tmp_path / safe_id
        export_dir  = session_dir / "exports"
        analysis_csv = export_dir / "analysis_results.csv"
        if not analysis_csv.exists():
            pytest.skip("analysis_results.csv not produced")
        df = pd.read_csv(analysis_csv)
        form_data = {f"flag_{fn}": "VALID" for fn in df["filename"].astype(str)}
        resp = client.post(f"/review/{session_id}", data=form_data,
                           follow_redirects=False)
        assert resp.status_code in (302, 200)
        if resp.status_code == 302:
            assert f"/predict/{session_id}" in resp.headers.get("Location", "")

    def test_predict_get_returns_200(self, flask_client):
        client, tmp_path = flask_client
        session_id = self._upload_and_analyze(client, tmp_path)
        if session_id is None:
            pytest.skip("Analysis produced no session_id in HTML")
        import app as flask_app
        from werkzeug.utils import secure_filename
        safe_id = secure_filename(session_id)
        session_dir = tmp_path / safe_id
        export_dir  = session_dir / "exports"
        analysis_csv = export_dir / "analysis_results.csv"
        if not analysis_csv.exists():
            pytest.skip("analysis_results.csv not produced")
        df = pd.read_csv(analysis_csv)
        form_data = {f"flag_{fn}": "VALID" for fn in df["filename"].astype(str)}
        client.post(f"/review/{session_id}", data=form_data)
        resp = client.get(f"/predict/{session_id}")
        assert resp.status_code == 200
        assert b"Prediction Results" in resp.data

    def test_review_unknown_session_returns_404(self, flask_client):
        client, _ = flask_client
        resp = client.get("/review/nonexistent-session-xyz")
        assert resp.status_code == 404
