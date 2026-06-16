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
from pathlib import Path

from flask import (Flask, render_template, request, send_from_directory)
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

DEMO_SESSION_DIR = TEMP_DIR / "demo-prebuilt"
DEMO_CACHE_JSON  = DEMO_SESSION_DIR / "demo_cache.json"

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
    return render_template("index.html", demo_ready=demo_ready)


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
