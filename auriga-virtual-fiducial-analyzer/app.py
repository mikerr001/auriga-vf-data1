"""
app.py
------
Auriga Virtual Fiducial Analyzer — Flask application entry point.

Demo dataset is pre-built once and persisted to disk (demo_cache.json).
On every subsequent restart the cache loads instantly from disk — no
re-running the pipeline.
"""

import json
import logging
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import (Flask, render_template, request, send_from_directory,
                   redirect, url_for, Response, jsonify)
from werkzeug.utils import secure_filename

from analysis.pipeline import PipelineError, run_pipeline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "auriga-dev-secret-change-in-prod")

BASE_DIR    = Path(__file__).parent
UPLOAD_DIR  = BASE_DIR / "uploads"
TEMP_DIR    = BASE_DIR / "temp"
EXPORTS_DIR = BASE_DIR / "exports"
FIXTURE_DIR = BASE_DIR / "static" / "fixtures"
FIXTURE_ZIP = FIXTURE_DIR / "calibration_demo.zip"
FIXTURE_CSV = FIXTURE_DIR / "calibration_demo.csv"

DEMO_SESSION_DIR   = TEMP_DIR / "demo-prebuilt"
DEMO_CACHE_JSON    = DEMO_SESSION_DIR / "demo_cache.json"
SESSIONS_INDEX_PATH = BASE_DIR / "sessions.json"

for d in (UPLOAD_DIR, TEMP_DIR, EXPORTS_DIR):
    d.mkdir(exist_ok=True)

ALLOWED_ARCHIVE  = {".zip"}
ALLOWED_METADATA = {".csv"}
MAX_CONTENT_MB   = 200
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

# -------------------------------------------------------------- demo cache

_demo_cache: dict = {}
_demo_lock  = threading.Lock()


def _load_cache_from_disk() -> bool:
    """Try to load a previously built demo cache from JSON. Returns True on success."""
    if not DEMO_CACHE_JSON.exists():
        return False
    try:
        with open(DEMO_CACHE_JSON) as f:
            data = json.load(f)
        with _demo_lock:
            _demo_cache.update(data)
        logger.info("Demo cache loaded from disk — detected=%d/%d",
                    data["stats"]["detected"], data["stats"]["total"])
        return True
    except Exception as e:
        logger.warning("Could not load demo cache from disk: %s", e)
        return False


def _save_cache_to_disk(data: dict) -> None:
    DEMO_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEMO_CACHE_JSON, "w") as f:
        json.dump(data, f)


def _build_demo_cache() -> None:
    """Run the full pipeline on the bundled fixture and persist results."""
    if not FIXTURE_ZIP.exists() or not FIXTURE_CSV.exists():
        logger.warning("Demo fixture files missing — /demo will not be available.")
        return

    logger.info("Pre-building demo session (first run)…")
    DEMO_SESSION_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = DEMO_SESSION_DIR / "calibration_demo.zip"
    csv_path = DEMO_SESSION_DIR / "calibration_demo.csv"
    shutil.copy2(FIXTURE_ZIP, zip_path)
    shutil.copy2(FIXTURE_CSV, csv_path)

    try:
        result = run_pipeline(zip_path, csv_path, DEMO_SESSION_DIR)
        ctx    = _build_dashboard_context(result)
        cache_data = {
            "session_id":  "demo-prebuilt",
            "stats":       ctx["stats"],
            "regressions": ctx["regressions"],
            "plots":       result["plots"],
            "warnings":    result["warnings"],
            "ready":       True,
        }
        _save_cache_to_disk(cache_data)
        with _demo_lock:
            _demo_cache.update(cache_data)
        logger.info("Demo session ready — detected=%d/%d",
                    ctx["stats"]["detected"], ctx["stats"]["total"])
    except Exception:
        logger.exception("Demo pre-build failed")
        with _demo_lock:
            _demo_cache["ready"] = False


def _init_demo_cache() -> None:
    """Load from disk if available, otherwise build in background thread."""
    if _load_cache_from_disk():
        return
    threading.Thread(target=_build_demo_cache, daemon=True).start()


_init_demo_cache()


# ---------------------------------------------------------- session index

_sessions_lock = threading.Lock()


def _load_sessions_index() -> list:
    """Return the list of session metadata dicts from sessions.json."""
    if not SESSIONS_INDEX_PATH.exists():
        return []
    try:
        with open(SESSIONS_INDEX_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("Could not read sessions index: %s", e)
        return []


def _save_sessions_index(sessions: list) -> None:
    with open(SESSIONS_INDEX_PATH, "w") as f:
        json.dump(sessions, f, indent=2, default=str)


def _upsert_session_index(session_id: str, **fields) -> None:
    """Insert or update a session entry in sessions.json (thread-safe)."""
    with _sessions_lock:
        sessions = _load_sessions_index()
        existing = next((s for s in sessions if s["id"] == session_id), None)
        if existing is None:
            entry = {"id": session_id, "created_at": None, "export_dir": None,
                     "phase": 1, "verdict": None}
            entry.update(fields)
            sessions.append(entry)
        else:
            existing.update(fields)
        _save_sessions_index(sessions)


def _get_recent_sessions(n: int = 5) -> list:
    """Return the n most-recently created sessions from the index."""
    sessions = _load_sessions_index()
    sessions_sorted = sorted(
        sessions,
        key=lambda s: s.get("created_at") or "",
        reverse=True,
    )
    return sessions_sorted[:n]


# ------------------------------------------------------------------ helpers

def _allowed(filename: str, allowed: set) -> bool:
    return Path(filename).suffix.lower() in allowed


def _build_dashboard_context(result: dict) -> dict:
    df       = result["df"]
    total    = len(df)
    detected = int(df["detectionSuccess"].sum())
    det_rate = round(detected / total * 100, 1) if total else 0
    return {
        "stats": {
            "total":        total,
            "detected":     detected,
            "det_rate":     det_rate,
            "failed":       total - detected,
            "distances":    {str(k): int(v) for k, v in
                             df["distanceMeters"].value_counts().sort_index().items()},
            "orientations": {str(k): int(v) for k, v in
                             df["orientation"].value_counts().items()},
        },
        "regressions": {
            k: {
                "r2":     round(v.r2, 4),
                "rmse":   round(v.rmse, 2),
                "mae":    round(v.mae, 2),
                "n":      v.n_samples,
                "interp": v.interpretation,
            }
            for k, v in result["regressions"].items() if v
        },
    }


# ------------------------------------------------------------------ routes

@app.route("/")
def index():
    with _demo_lock:
        demo_ready = _demo_cache.get("ready", False)
    recent = _get_recent_sessions(n=5)
    return render_template("index.html", demo_ready=demo_ready, recent_sessions=recent)


@app.route("/analyze", methods=["POST"])
def analyze():
    errors   = []
    zip_file = request.files.get("zip_file")
    csv_file = request.files.get("csv_file")

    if not zip_file or zip_file.filename == "":
        errors.append("No image ZIP uploaded.")
    elif not _allowed(zip_file.filename, ALLOWED_ARCHIVE):
        errors.append("Invalid archive format. Please upload a .zip file.")

    if not csv_file or csv_file.filename == "":
        errors.append("No metadata CSV uploaded.")
    elif not _allowed(csv_file.filename, ALLOWED_METADATA):
        errors.append("Invalid metadata format. Please upload a .csv file.")

    if errors:
        return render_template("index.html", errors=errors)

    session_id  = uuid.uuid4().hex
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(parents=True)

    zip_path = session_dir / secure_filename(zip_file.filename)
    csv_path = session_dir / secure_filename(csv_file.filename)
    zip_file.save(str(zip_path))
    csv_file.save(str(csv_path))

    try:
        result = run_pipeline(zip_path, csv_path, session_dir)
    except PipelineError as e:
        return render_template("index.html", errors=[str(e)])
    except Exception as e:
        logger.exception("Unexpected error during pipeline")
        return render_template("index.html", errors=[f"Unexpected error: {e}"])

    ctx = _build_dashboard_context(result)

    phase2_cache = {
        "stats":       ctx["stats"],
        "regressions": ctx["regressions"],
        "plots":       result["plots"],
        "warnings":    result["warnings"],
    }
    try:
        with open(session_dir / "phase2_cache.json", "w") as f:
            json.dump(phase2_cache, f, default=str)
    except Exception:
        logger.warning("Could not save phase2_cache for session %s", session_id)

    export_dir = session_dir / "exports"
    created_at = datetime.now(timezone.utc).isoformat()
    _upsert_session_index(
        session_id,
        created_at=created_at,
        export_dir=str(export_dir),
        phase=2,
        verdict=None,
    )

    return render_template(
        "dashboard.html",
        session_id=session_id,
        stats=ctx["stats"],
        regressions=ctx["regressions"],
        plots=result["plots"],
        warnings=result["warnings"],
        is_demo=False,
    )


@app.route("/demo")
def demo():
    """Serve pre-built demo results (instant after first build)."""
    with _demo_lock:
        cache = dict(_demo_cache)

    if not cache.get("ready"):
        return render_template("demo_loading.html"), 202

    return render_template(
        "dashboard.html",
        session_id=cache["session_id"],
        stats=cache["stats"],
        regressions=cache["regressions"],
        plots=cache["plots"],
        warnings=cache["warnings"],
        is_demo=True,
    )


@app.route("/demo/status")
def demo_status():
    """JSON endpoint polled by demo_loading.html."""
    with _demo_lock:
        ready = _demo_cache.get("ready", False)
    return {"ready": ready}


@app.route("/exports/<session_id>/<path:filename>")
def serve_export(session_id: str, filename: str):
    safe_id    = secure_filename(session_id)
    export_dir = TEMP_DIR / safe_id / "exports"
    return send_from_directory(str(export_dir), filename)


@app.route("/download/<session_id>/<path:filename>")
def download_file(session_id: str, filename: str):
    safe_id    = secure_filename(session_id)
    export_dir = TEMP_DIR / safe_id / "exports"
    return send_from_directory(str(export_dir), filename, as_attachment=True)


# ---------------------------------------------------------------- Phase 3 helpers

def _load_analysis_df(session_dir: Path) -> "pd.DataFrame":
    import pandas as pd
    csv_path = session_dir / "exports" / "analysis_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"analysis_results.csv not found in {session_dir / 'exports'}")
    return pd.read_csv(csv_path)


def _load_quality_flags(session_dir: Path) -> dict:
    flags_path = session_dir / "quality_flags.json"
    if flags_path.exists():
        with open(flags_path) as f:
            return json.load(f)
    return {}


def _save_quality_flags(session_dir: Path, flags: dict) -> None:
    with open(session_dir / "quality_flags.json", "w") as f:
        json.dump(flags, f)


def _load_phase3_cache(session_dir: Path) -> dict:
    cache_path = session_dir / "phase3_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_phase3_cache(session_dir: Path, data: dict) -> None:
    with open(session_dir / "phase3_cache.json", "w") as f:
        json.dump(data, f, default=str)


# ---------------------------------------------------------------- Phase 3 routes

@app.route("/review/<session_id>", methods=["GET"])
def review(session_id: str):
    import pandas as pd
    safe_id     = secure_filename(session_id)
    session_dir = TEMP_DIR / safe_id

    try:
        df = _load_analysis_df(session_dir)
    except FileNotFoundError as e:
        return render_template("error.html", code=404, message=str(e)), 404

    existing_flags = _load_quality_flags(session_dir)
    samples = []
    for _, row in df.iterrows():
        fname = str(row.get("filename", ""))
        samples.append({
            "filename":       fname,
            "distanceMeters": row.get("distanceMeters"),
            "orientation":    row.get("orientation"),
            "markerWidthPx":  row.get("markerWidthPx"),
            "detectionSuccess": bool(row.get("detectionSuccess", False)),
            "flag": existing_flags.get(fname, "VALID"),
        })

    return render_template("review.html",
                           session_id=session_id,
                           samples=samples)


@app.route("/review/<session_id>", methods=["POST"])
def review_submit(session_id: str):
    import pandas as pd
    from analysis.lut import build_lut, LUTError
    from analysis.prediction import run_prediction, PredictionError
    from analysis.cross_validation import run_cross_validation, CrossValidationError
    from analysis.sensitivity import run_sensitivity_analysis, SensitivityError
    from analysis.recommendation import generate_recommendation
    from analysis.lut_plots import generate_lut_plots

    safe_id     = secure_filename(session_id)
    session_dir = TEMP_DIR / safe_id
    export_dir  = session_dir / "exports"

    try:
        df = _load_analysis_df(session_dir)
    except FileNotFoundError as e:
        return render_template("error.html", code=404, message=str(e)), 404

    quality_flags = {}
    for fname in df["filename"].astype(str):
        flag = request.form.get(f"flag_{fname}", "VALID")
        if flag not in ("VALID", "QUESTIONABLE", "INVALID"):
            flag = "VALID"
        quality_flags[fname] = flag

    _save_quality_flags(session_dir, quality_flags)

    errors = []

    try:
        lut_result = build_lut(df, quality_flags, export_dir)
    except LUTError as e:
        return render_template(
            "review.html",
            session_id=session_id,
            samples=[{
                "filename": r["filename"], "distanceMeters": r.get("distanceMeters"),
                "orientation": r.get("orientation"), "markerWidthPx": r.get("markerWidthPx"),
                "detectionSuccess": bool(r.get("detectionSuccess", False)),
                "flag": quality_flags.get(r["filename"], "VALID"),
            } for r in df.to_dict("records")],
            error=str(e),
        )

    try:
        pred_result = run_prediction(df, quality_flags, lut_result.model, export_dir)
    except PredictionError as e:
        errors.append(f"Prediction error: {e}")
        pred_result = None

    cv_result = None
    try:
        cv_result = run_cross_validation(df, quality_flags)
    except CrossValidationError as e:
        errors.append(f"Cross-validation skipped: {e}")

    sensitivity_result = None
    try:
        sensitivity_result = run_sensitivity_analysis(df, quality_flags, lut_result.model)
    except SensitivityError as e:
        errors.append(f"Sensitivity analysis skipped: {e}")

    recommendation = None
    if pred_result:
        results_df = pred_result.results_df
        recommendation = generate_recommendation(pred_result.metrics, results_df)

    lut_plots = {}
    if pred_result:
        lut_plots = generate_lut_plots(pred_result.results_df, df, lut_result.model, export_dir)

    cache_data = {
        "quality_flags": quality_flags,
        "metrics": {
            "mae":       pred_result.metrics.mae if pred_result else None,
            "rmse":      pred_result.metrics.rmse if pred_result else None,
            "mape":      pred_result.metrics.mape if pred_result else None,
            "r2":        pred_result.metrics.r2 if pred_result else None,
            "n_samples": pred_result.metrics.n_samples if pred_result else None,
        },
        "cv": {
            "k":         cv_result.k if cv_result else None,
            "mean_mae":  cv_result.mean_mae if cv_result else None,
            "std_mae":   cv_result.std_mae if cv_result else None,
            "mean_rmse": cv_result.mean_rmse if cv_result else None,
            "std_rmse":  cv_result.std_rmse if cv_result else None,
            "mean_r2":   cv_result.mean_r2 if cv_result else None,
            "std_r2":    cv_result.std_r2 if cv_result else None,
            "n_samples": cv_result.n_samples if cv_result else None,
            "folds": [
                {"fold": f.fold, "n_train": f.n_train, "n_test": f.n_test,
                 "mae": f.mae, "rmse": f.rmse, "r2": f.r2}
                for f in cv_result.folds
            ] if cv_result else [],
        },
        "sensitivity": {
            "baseline_mae":  sensitivity_result.baseline_mae if sensitivity_result else None,
            "baseline_rmse": sensitivity_result.baseline_rmse if sensitivity_result else None,
            "n_samples":     sensitivity_result.n_samples if sensitivity_result else None,
            "perturbations": [
                {"perturbation_pct": p.perturbation_pct, "direction": p.direction,
                 "mae": p.mae, "rmse": p.rmse, "mean_abs_delta": p.mean_abs_delta}
                for p in sensitivity_result.perturbations
            ] if sensitivity_result else [],
        },
        "recommendation": {
            "verdict":   recommendation.verdict if recommendation else None,
            "rationale": recommendation.rationale if recommendation else [],
            "r2":        recommendation.r2 if recommendation else None,
            "mae":       recommendation.mae if recommendation else None,
            "rmse":      recommendation.rmse if recommendation else None,
            "mape":      recommendation.mape if recommendation else None,
            "orientation_anova_p": recommendation.orientation_anova_p if recommendation else None,
        },
        "lut_plots": lut_plots,
        "errors": errors,
        "n_valid": lut_result.n_valid,
        "n_total": lut_result.n_total,
    }
    _save_phase3_cache(session_dir, cache_data)

    verdict = cache_data["recommendation"].get("verdict")
    _upsert_session_index(session_id, phase=3, verdict=verdict)

    return redirect(url_for("predict", session_id=session_id))


@app.route("/predict/<session_id>", methods=["GET"])
def predict(session_id: str):
    safe_id     = secure_filename(session_id)
    session_dir = TEMP_DIR / safe_id

    cache = _load_phase3_cache(session_dir)
    if not cache:
        return redirect(url_for("review", session_id=session_id))

    return render_template(
        "prediction_results.html",
        session_id=session_id,
        metrics=cache.get("metrics", {}),
        cv=cache.get("cv", {}),
        sensitivity=cache.get("sensitivity", {}),
        recommendation=cache.get("recommendation", {}),
        lut_plots=cache.get("lut_plots", {}),
        errors=cache.get("errors", []),
        n_valid=cache.get("n_valid", 0),
        n_total=cache.get("n_total", 0),
    )


@app.route("/predict-live/<session_id>", methods=["GET", "POST"])
def predict_live(session_id: str):
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from analysis.lut import build_lut, LUTError
    from analysis.prediction import _predict_distance_from_width

    safe_id     = secure_filename(session_id)
    session_dir = TEMP_DIR / safe_id
    export_dir  = session_dir / "exports"

    try:
        df = _load_analysis_df(session_dir)
    except FileNotFoundError as e:
        return render_template("error.html", code=404, message=str(e)), 404

    quality_flags = _load_quality_flags(session_dir)
    if not quality_flags:
        return render_template(
            "error.html", code=400,
            message="No quality flags found. Please complete the review step first."
        ), 400

    cache = _load_phase3_cache(session_dir)
    cv    = cache.get("cv") or {}

    # Rebuild the LUT model (cheap — just refitting the sklearn pipeline)
    try:
        lut_result = build_lut(df, quality_flags, export_dir)
    except LUTError as e:
        return render_template("error.html", code=500, message=str(e)), 500

    model = lut_result.model

    # Derive distance range from training data for axis limits
    valid_distances = df[
        df["filename"].isin({k for k, v in quality_flags.items() if v == "VALID"})
        & (df["detectionSuccess"] == True)
        & df["distanceMeters"].notna()
        & (df["distanceMeters"] > 0)
    ]["distanceMeters"]

    d_min = float(valid_distances.min()) if not valid_distances.empty else 0.01
    d_max = float(valid_distances.max()) if not valid_distances.empty else 100.0

    result    = None
    error_msg = None

    if request.method == "POST":
        raw = request.form.get("marker_width_px", "").strip()
        try:
            marker_width_px = float(raw)
            if marker_width_px <= 0:
                raise ValueError("Must be positive")
        except ValueError:
            error_msg = "Please enter a valid positive number for marker width."
        else:
            predicted_dist = _predict_distance_from_width(
                model, marker_width_px,
                distance_min=max(0.001, d_min * 0.5),
                distance_max=d_max * 2.0,
            )

            mean_mae  = cv.get("mean_mae")
            mean_rmse = cv.get("mean_rmse")

            lower_bound = (predicted_dist - mean_mae)  if mean_mae  is not None else None
            upper_bound = (predicted_dist + mean_mae)  if mean_mae  is not None else None
            rmse_lower  = (predicted_dist - mean_rmse) if mean_rmse is not None else None
            rmse_upper  = (predicted_dist + mean_rmse) if mean_rmse is not None else None

            # Generate inline curve plot with query point highlighted
            plot_b64 = None
            try:
                x_curve = np.linspace(d_min, d_max, 300)
                y_curve = model.predict(x_curve.reshape(-1, 1))

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(x_curve, y_curve, "--", color="#4A90D9", linewidth=2, label="Fitted curve")

                # Scatter training data
                detected = df[
                    (df["detectionSuccess"] == True)
                    & df["markerWidthPx"].notna()
                    & df["distanceMeters"].notna()
                ]
                if not detected.empty:
                    ax.scatter(
                        detected["distanceMeters"], detected["markerWidthPx"],
                        color="#4A90D9", s=35, alpha=0.4, label="Training samples"
                    )

                # Highlight the query point
                ax.scatter(
                    [predicted_dist], [marker_width_px],
                    color="#E74C3C", s=120, zorder=5, label=f"Query ({marker_width_px:.1f} px)"
                )

                # Draw crosshairs at query point
                ax.axhline(marker_width_px, color="#E74C3C", linewidth=0.8, alpha=0.5, linestyle=":")
                ax.axvline(predicted_dist, color="#E74C3C", linewidth=0.8, alpha=0.5, linestyle=":")

                # Shade MAE error band around predicted distance
                if mean_mae is not None:
                    ax.axvspan(
                        max(0, lower_bound), upper_bound,
                        alpha=0.15, color="#E74C3C", label=f"±MAE ({mean_mae:.4f} m)"
                    )

                ax.set_xlabel("Distance (m)", fontsize=11)
                ax.set_ylabel("Marker Width (px)", fontsize=11)
                ax.set_title("Fitted Calibration Curve — Live Query", fontsize=12)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                plot_b64 = base64.b64encode(buf.read()).decode("ascii")
            except Exception:
                logger.exception("Live prediction plot generation failed")

            result = {
                "marker_width_px": marker_width_px,
                "predicted_dist":  round(predicted_dist, 4),
                "mean_mae":        round(mean_mae,  4) if mean_mae  is not None else None,
                "mean_rmse":       round(mean_rmse, 4) if mean_rmse is not None else None,
                "lower_mae":       round(max(0, lower_bound), 4) if lower_bound is not None else None,
                "upper_mae":       round(upper_bound, 4)         if upper_bound is not None else None,
                "lower_rmse":      round(max(0, rmse_lower), 4)  if rmse_lower  is not None else None,
                "upper_rmse":      round(rmse_upper, 4)          if rmse_upper  is not None else None,
                "plot_b64":        plot_b64,
            }

    return render_template(
        "predict_live.html",
        session_id=session_id,
        result=result,
        error_msg=error_msg,
        cv=cv,
        d_min=round(d_min, 3),
        d_max=round(d_max, 3),
    )


@app.route("/lut-report/<session_id>", methods=["GET"])
def lut_report(session_id: str):
    import pandas as pd
    from analysis.lut import build_lut, LUTError
    from analysis.prediction import run_prediction, PredictionError
    from analysis.cross_validation import run_cross_validation, CrossValidationError
    from analysis.sensitivity import run_sensitivity_analysis, SensitivityError
    from analysis.recommendation import generate_recommendation
    from analysis.lut_plots import generate_lut_plots
    from analysis.lut_report import generate_lut_report

    safe_id     = secure_filename(session_id)
    session_dir = TEMP_DIR / safe_id
    export_dir  = session_dir / "exports"

    try:
        df = _load_analysis_df(session_dir)
    except FileNotFoundError as e:
        return render_template("error.html", code=404, message=str(e)), 404

    quality_flags = _load_quality_flags(session_dir)
    if not quality_flags:
        return render_template("error.html", code=404,
                               message="No quality flags found. Please complete the review step first."), 404

    cache = _load_phase3_cache(session_dir)
    lut_plots = cache.get("lut_plots", {})

    try:
        lut_result = build_lut(df, quality_flags, export_dir)
    except LUTError as e:
        return render_template("error.html", code=500, message=str(e)), 500

    try:
        pred_result = run_prediction(df, quality_flags, lut_result.model, export_dir)
    except PredictionError as e:
        return render_template("error.html", code=500, message=str(e)), 500

    cv_result = None
    try:
        cv_result = run_cross_validation(df, quality_flags)
    except CrossValidationError:
        pass

    sensitivity_result = None
    try:
        sensitivity_result = run_sensitivity_analysis(df, quality_flags, lut_result.model)
    except SensitivityError:
        pass

    recommendation = generate_recommendation(pred_result.metrics, pred_result.results_df)

    if not lut_plots:
        lut_plots = generate_lut_plots(pred_result.results_df, df, lut_result.model, export_dir)

    report_path = generate_lut_report(
        analysis_df=df,
        results_df=pred_result.results_df,
        quality_flags=quality_flags,
        metrics=pred_result.metrics,
        cv_result=cv_result,
        sensitivity_result=sensitivity_result,
        recommendation=recommendation,
        lut_plots=lut_plots,
        export_dir=export_dir,
    )

    return send_from_directory(
        str(export_dir),
        report_path.name,
        as_attachment=True,
        mimetype="application/pdf",
    )


# ---------------------------------------------------------------- Session resume

@app.route("/session/<session_id>/resume")
def resume_session(session_id: str):
    """Rehydrate a previous session from its on-disk cache."""
    safe_id     = secure_filename(session_id)
    session_dir = TEMP_DIR / safe_id

    if not session_dir.is_dir():
        return render_template("error.html", code=404,
                               message=f"Session '{safe_id}' not found on disk."), 404

    phase3 = _load_phase3_cache(session_dir)
    if phase3:
        return redirect(url_for("predict", session_id=safe_id))

    cache_path = session_dir / "phase2_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            return render_template(
                "dashboard.html",
                session_id=safe_id,
                stats=cache["stats"],
                regressions=cache["regressions"],
                plots=cache["plots"],
                warnings=cache.get("warnings", []),
                is_demo=False,
            )
        except Exception as e:
            logger.warning("Could not load phase2_cache for session %s: %s", safe_id, e)

    return render_template("error.html", code=404,
                           message="Session data not found. The session may have been lost before the cache was saved."), 404


# ---------------------------------------------------------------- Sessions comparison

def _load_session_label(session_dir: Path) -> str:
    label_path = session_dir / "session_label.json"
    if label_path.exists():
        try:
            with open(label_path) as f:
                return json.load(f).get("label", "")
        except Exception:
            return ""
    return ""


def _save_session_label(session_dir: Path, label: str) -> None:
    with open(session_dir / "session_label.json", "w") as f:
        json.dump({"label": label}, f)


@app.route("/sessions")
def sessions():
    entries = []
    if TEMP_DIR.exists():
        for session_dir in sorted(TEMP_DIR.iterdir()):
            if not session_dir.is_dir():
                continue
            cache_path = session_dir / "phase3_cache.json"
            if not cache_path.exists():
                continue
            try:
                with open(cache_path) as f:
                    cache = json.load(f)
            except Exception:
                continue
            metrics = cache.get("metrics") or {}
            recommendation = cache.get("recommendation") or {}
            entries.append({
                "session_id": session_dir.name,
                "label":      _load_session_label(session_dir),
                "n_valid":    cache.get("n_valid"),
                "n_total":    cache.get("n_total"),
                "r2":         metrics.get("r2"),
                "mae":        metrics.get("mae"),
                "rmse":       metrics.get("rmse"),
                "mape":       metrics.get("mape"),
                "verdict":    recommendation.get("verdict"),
            })
    return render_template("sessions.html", sessions=entries)


@app.route("/sessions/<session_id>/label", methods=["POST"])
def set_session_label(session_id: str):
    safe_id     = secure_filename(session_id)
    session_dir = TEMP_DIR / safe_id
    if not session_dir.is_dir():
        return jsonify({"error": "Session not found"}), 404
    label = (request.json or {}).get("label", "").strip()
    _save_session_label(session_dir, label)
    return jsonify({"ok": True, "label": label})


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html",
                           errors=[f"Upload too large. Maximum is {MAX_CONTENT_MB} MB."]), 413


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404, message="Page not found."), 404


@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", code=500, message="Internal server error."), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
