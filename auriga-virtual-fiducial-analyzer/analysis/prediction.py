"""
prediction.py
-------------
Inverse-model prediction from a fitted LUT model.
Computes per-sample actualDistance / predictedDistance / absoluteError / percentageError,
writes prediction_results.csv, computes MAE / RMSE / MAPE / R².
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    pass


@dataclass
class PredictionMetrics:
    mae: float
    rmse: float
    mape: float
    r2: float
    n_samples: int


@dataclass
class PredictionResult:
    results_df: pd.DataFrame
    results_path: Path
    metrics: PredictionMetrics


def _predict_distance_from_width(model, marker_width_px: float,
                                  distance_min: float = 0.01,
                                  distance_max: float = 100.0) -> float:
    """
    Invert the polynomial model: given markerWidthPx, find distanceMeters.
    Uses Brent's method to find the root of (model(d) - marker_width_px) = 0.
    Falls back to a grid search on the sampled curve if bracketing fails.
    """
    def objective(d):
        return float(model.predict(np.array([[d]]))[0]) - marker_width_px

    try:
        f_min = objective(distance_min)
        f_max = objective(distance_max)
        if f_min * f_max < 0:
            return brentq(objective, distance_min, distance_max, xtol=1e-6)
    except Exception:
        pass

    # Grid search fallback
    distances = np.linspace(distance_min, distance_max, 2000)
    widths = model.predict(distances.reshape(-1, 1))
    idx = np.argmin(np.abs(widths - marker_width_px))
    return float(distances[idx])


def run_prediction(
    analysis_df: pd.DataFrame,
    quality_flags: dict,
    model,
    export_dir: Path,
    out_filename: str = "prediction_results.csv",
) -> PredictionResult:
    """
    Run distance prediction for all VALID detected rows.

    Parameters
    ----------
    analysis_df   : DataFrame from analysis_results.csv
    quality_flags : dict mapping filename -> "VALID" / "QUESTIONABLE" / "INVALID"
    model         : fitted sklearn pipeline from LUTResult.model
    export_dir    : directory to write results CSV
    out_filename  : output CSV filename

    Returns
    -------
    PredictionResult dataclass

    Raises
    ------
    PredictionError on bad inputs or insufficient data.
    """
    required_cols = {"filename", "distanceMeters", "markerWidthPx", "detectionSuccess"}
    missing = required_cols - set(analysis_df.columns)
    if missing:
        raise PredictionError(f"Missing required columns: {', '.join(sorted(missing))}")

    valid_filenames = {k for k, v in quality_flags.items() if v == "VALID"}
    valid_df = analysis_df[
        analysis_df["filename"].isin(valid_filenames)
        & (analysis_df["detectionSuccess"] == True)
        & analysis_df["markerWidthPx"].notna()
        & analysis_df["distanceMeters"].notna()
        & (analysis_df["markerWidthPx"] > 0)
        & (analysis_df["distanceMeters"] > 0)
    ].copy()

    if len(valid_df) == 0:
        raise PredictionError("No VALID rows available for prediction.")

    predicted_distances = []
    for _, row in valid_df.iterrows():
        pred_d = _predict_distance_from_width(model, float(row["markerWidthPx"]))
        predicted_distances.append(pred_d)

    valid_df = valid_df.copy()
    valid_df["predictedDistance"] = predicted_distances
    valid_df["absoluteError"] = np.abs(
        valid_df["distanceMeters"] - valid_df["predictedDistance"]
    )
    valid_df["percentageError"] = (
        valid_df["absoluteError"] / valid_df["distanceMeters"].abs() * 100
    )

    results_cols = [
        "filename", "distanceMeters", "markerWidthPx", "orientation",
        "predictedDistance", "absoluteError", "percentageError",
    ]
    results_cols = [c for c in results_cols if c in valid_df.columns]
    results_df = valid_df[results_cols].copy()

    export_dir.mkdir(parents=True, exist_ok=True)
    results_path = export_dir / out_filename
    results_df.to_csv(results_path, index=False)
    logger.info("Prediction results written to %s", results_path)

    actual = valid_df["distanceMeters"].values
    predicted = np.array(predicted_distances)

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / np.abs(actual))) * 100)
    r2 = float(r2_score(actual, predicted))

    metrics = PredictionMetrics(
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        mape=round(mape, 4),
        r2=round(r2, 4),
        n_samples=len(valid_df),
    )

    return PredictionResult(
        results_df=results_df,
        results_path=results_path,
        metrics=metrics,
    )
