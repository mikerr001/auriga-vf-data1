"""
app.py
------
Auriga Virtual Fiducial Analyzer — Flask application entry point.
"""

import logging
import os
import uuid
from pathlib import Path

from flask import (Flask, jsonify, redirect, render_template,
                   request, send_from_directory, url_for)
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

for d in (UPLOAD_DIR, TEMP_DIR, EXPORTS_DIR):
    d.mkdir(exist_ok=True)

ALLOWED_ARCHIVE  = {".zip"}
ALLOWED_METADATA = {".csv"}
MAX_CONTENT_MB   = 200
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024


def _allowed(filename: str, allowed: set) -> bool:
    return Path(filename).suffix.lower() in allowed


# ------------------------------------------------------------------ routes

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    errors = []

    zip_file  = request.files.get("zip_file")
    csv_file  = request.files.get("csv_file")

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
        return render_template("index.html",
                               errors=[f"Unexpected error: {e}"])

    df = result["df"]
    total       = len(df)
    detected    = int(df["detectionSuccess"].sum())
    det_rate    = round(detected / total * 100, 1) if total else 0
    dist_counts = df["distanceMeters"].value_counts().sort_index().to_dict()
    ori_counts  = df["orientation"].value_counts().to_dict()

    stats = {
        "total":     total,
        "detected":  detected,
        "det_rate":  det_rate,
        "failed":    total - detected,
        "distances": dist_counts,
        "orientations": ori_counts,
    }

    regressions = {
        k: {
            "r2":      round(v.r2, 4),
            "rmse":    round(v.rmse, 2),
            "mae":     round(v.mae, 2),
            "n":       v.n_samples,
            "interp":  v.interpretation,
        }
        for k, v in result["regressions"].items() if v
    }

    return render_template(
        "dashboard.html",
        session_id=session_id,
        stats=stats,
        regressions=regressions,
        plots=result["plots"],
        warnings=result["warnings"],
    )


@app.route("/exports/<session_id>/<path:filename>")
def serve_export(session_id: str, filename: str):
    safe_id  = secure_filename(session_id)
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
    return render_template("error.html", code=404,
                           message="Page not found."), 404


@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", code=500,
                           message="Internal server error."), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
