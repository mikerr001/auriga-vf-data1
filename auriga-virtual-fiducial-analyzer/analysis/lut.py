"""
lut.py
------
Builds a calibration Look-Up Table (LUT) from quality-flagged VALID rows.
Writes virtual_fiducial_lut.csv to the session's exports folder.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)


class LUTError(Exception):
    pass


@dataclass
class LUTResult:
    lut_df: pd.DataFrame
    lut_path: Path
    model: object
    n_valid: int
    n_total: int
    poly_coefficients: list


def build_lut(
    analysis_df: pd.DataFrame,
    quality_flags: dict,
    export_dir: Path,
    out_filename: str = "virtual_fiducial_lut.csv",
) -> LUTResult:
    """
    Build a LUT from VALID rows only.

    Parameters
    ----------
    analysis_df   : DataFrame from analysis_results.csv
    quality_flags : dict mapping filename -> flag string ("VALID","QUESTIONABLE","INVALID")
    export_dir    : directory to write the LUT CSV
    out_filename  : output CSV filename

    Returns
    -------
    LUTResult dataclass

    Raises
    ------
    LUTError if no VALID rows remain or required columns are missing.
    """
    required_cols = {"filename", "distanceMeters", "markerWidthPx", "orientation", "detectionSuccess"}
    missing = required_cols - set(analysis_df.columns)
    if missing:
        raise LUTError(f"Missing required columns: {', '.join(sorted(missing))}")

    n_total = len(analysis_df)

    valid_filenames = {k for k, v in quality_flags.items() if v == "VALID"}
    valid_df = analysis_df[
        analysis_df["filename"].isin(valid_filenames)
        & (analysis_df["detectionSuccess"] == True)
        & analysis_df["markerWidthPx"].notna()
        & analysis_df["distanceMeters"].notna()
        & (analysis_df["markerWidthPx"] > 0)
        & (analysis_df["distanceMeters"] > 0)
    ].copy()

    n_valid = len(valid_df)

    if n_valid == 0:
        raise LUTError(
            "No VALID rows with successful detection remain after quality filtering. "
            "Cannot build LUT."
        )

    if n_valid < 3:
        raise LUTError(
            f"Only {n_valid} VALID row(s) available — need at least 3 to fit a reliable model."
        )

    X = valid_df["distanceMeters"].values.reshape(-1, 1)
    y = valid_df["markerWidthPx"].values

    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=True),
        LinearRegression(),
    )
    model.fit(X, y)

    poly_coefficients = list(
        model.named_steps["linearregression"].coef_.tolist()
    )
    poly_coefficients[0] = float(model.named_steps["linearregression"].intercept_)

    distances = np.unique(np.sort(valid_df["distanceMeters"].values))
    predicted_widths = model.predict(distances.reshape(-1, 1))

    lut_df = pd.DataFrame({
        "distanceMeters": distances,
        "predictedMarkerWidthPx": np.round(predicted_widths, 3),
    })

    lut_df["source"] = "VALID calibration data"
    lut_df["n_valid_samples"] = n_valid

    export_dir.mkdir(parents=True, exist_ok=True)
    lut_path = export_dir / out_filename
    lut_df.to_csv(lut_path, index=False)
    logger.info("LUT written to %s (%d rows)", lut_path, len(lut_df))

    return LUTResult(
        lut_df=lut_df,
        lut_path=lut_path,
        model=model,
        n_valid=n_valid,
        n_total=n_total,
        poly_coefficients=poly_coefficients,
    )
