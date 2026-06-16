"""
pipeline.py
-----------
Orchestrates the full analysis: detection → DataFrame → plots → regression → PDF.
"""

import logging
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from .detector import detect_marker, annotate_image
from .plots import generate_all_plots
from .regression import run_all_regressions
from .report import generate_report

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
REQUIRED_CSV_COLUMNS = {
    "filename", "distanceMeters", "orientation",
    "cameraHeightCm", "deviceName",
}


class PipelineError(Exception):
    pass


def validate_csv(df: pd.DataFrame) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors = []
    missing = REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing required CSV columns: {', '.join(sorted(missing))}")
    return errors


def extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """
    Extract all supported image files from *zip_path* into *dest_dir*.
    Returns a list of extracted image paths.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            suffix = Path(member.filename).suffix.lower()
            if suffix not in ALLOWED_IMAGE_SUFFIXES:
                continue
            # Flatten directory structure — store by basename only
            basename = Path(member.filename).name
            out_path = dest_dir / basename
            out_path.write_bytes(zf.read(member.filename))
            extracted.append(out_path)

    logger.info("Extracted %d image(s) from ZIP", len(extracted))
    return extracted


def run_pipeline(zip_path: Path,
                 csv_path: Path,
                 session_dir: Path) -> dict:
    """
    Full analysis pipeline.

    Parameters
    ----------
    zip_path    : path to uploaded image ZIP
    csv_path    : path to uploaded metadata CSV
    session_dir : working directory for this analysis session

    Returns
    -------
    dict with keys: df, regressions, plots, report_path, warnings, errors
    """
    warnings: list[str] = []
    errors: list[str] = []

    images_dir = session_dir / "images"
    export_dir = session_dir / "exports"
    annotated_dir = session_dir / "annotated"
    export_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load CSV
    try:
        df_meta = pd.read_csv(csv_path)
    except Exception as e:
        raise PipelineError(f"Could not parse metadata CSV: {e}") from e

    csv_errors = validate_csv(df_meta)
    if csv_errors:
        raise PipelineError("; ".join(csv_errors))

    df_meta["distanceMeters"] = pd.to_numeric(df_meta["distanceMeters"], errors="coerce")
    df_meta["cameraHeightCm"] = pd.to_numeric(df_meta["cameraHeightCm"], errors="coerce")

    # 2. Extract ZIP
    try:
        extracted_paths = extract_zip(zip_path, images_dir)
    except Exception as e:
        raise PipelineError(f"Could not extract ZIP: {e}") from e

    extracted_by_name = {p.name: p for p in extracted_paths}

    # 3. Detect markers and build results DataFrame
    records = []
    for _, row in df_meta.iterrows():
        fname = str(row.get("filename", "")).strip()
        img_path = extracted_by_name.get(fname)

        if img_path is None:
            warnings.append(f"Image not found in ZIP: {fname}")
            records.append({
                "filename": fname,
                "distanceMeters": row.get("distanceMeters"),
                "orientation": row.get("orientation"),
                "cameraHeightCm": row.get("cameraHeightCm"),
                "deviceName": row.get("deviceName"),
                "markerWidthPx": None,
                "markerHeightPx": None,
                "markerAreaPx": None,
                "centerX": None,
                "centerY": None,
                "detectionSuccess": False,
            })
            continue

        result = detect_marker(img_path)
        if not result.success:
            warnings.append(f"No marker detected in {fname}: {result.error}")

        annotate_image(img_path, result, annotated_dir / fname)

        records.append({
            "filename": fname,
            "distanceMeters": row.get("distanceMeters"),
            "orientation": row.get("orientation"),
            "cameraHeightCm": row.get("cameraHeightCm"),
            "deviceName": row.get("deviceName"),
            "markerWidthPx": result.marker_width_px if result.success else None,
            "markerHeightPx": result.marker_height_px if result.success else None,
            "markerAreaPx": result.marker_area_px if result.success else None,
            "centerX": result.center_x if result.success else None,
            "centerY": result.center_y if result.success else None,
            "detectionSuccess": result.success,
        })

    # 4. Images in ZIP with no CSV row
    csv_filenames = set(df_meta["filename"].astype(str).str.strip())
    for name in extracted_by_name:
        if name not in csv_filenames:
            warnings.append(f"Image in ZIP has no metadata row: {name}")

    df_results = pd.DataFrame(records)
    df_results.to_csv(export_dir / "analysis_results.csv", index=False)

    # 5. Plots
    plots = generate_all_plots(df_results, export_dir)

    # 6. Regression
    regressions = run_all_regressions(df_results)

    # 7. PDF report
    report_path = generate_report(df_results, regressions, plots, export_dir)

    return {
        "df": df_results,
        "regressions": regressions,
        "plots": plots,
        "report_path": report_path,
        "export_dir": export_dir,
        "annotated_dir": annotated_dir,
        "warnings": warnings,
        "errors": errors,
    }
