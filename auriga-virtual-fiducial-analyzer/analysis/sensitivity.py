"""
sensitivity.py
--------------
Sensitivity analysis: perturbs markerWidthPx by ±1%, ±3%, ±5%
and re-runs prediction to report how much prediction error grows.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class SensitivityError(Exception):
    pass


@dataclass
class PerturbationResult:
    perturbation_pct: float
    direction: str
    mae: float
    rmse: float
    mean_abs_delta: float


@dataclass
class SensitivityResult:
    perturbations: list
    baseline_mae: float
    baseline_rmse: float
    n_samples: int


def _predict_distances(model, marker_widths: np.ndarray,
                        distance_min: float = 0.01,
                        distance_max: float = 100.0) -> np.ndarray:
    """Vectorised grid-search inversion of the polynomial model."""
    grid = np.linspace(distance_min, distance_max, 3000)
    predicted_widths = model.predict(grid.reshape(-1, 1))
    results = []
    for w in marker_widths:
        idx = np.argmin(np.abs(predicted_widths - w))
        results.append(grid[idx])
    return np.array(results)


def run_sensitivity_analysis(
    analysis_df: pd.DataFrame,
    quality_flags: dict,
    model,
    perturbation_levels: tuple = (1.0, 3.0, 5.0),
) -> SensitivityResult:
    """
    Perturb markerWidthPx by ±p% and re-run prediction; report error growth.

    Parameters
    ----------
    analysis_df         : DataFrame from analysis_results.csv
    quality_flags       : dict mapping filename -> flag string
    model               : fitted sklearn pipeline from LUTResult.model
    perturbation_levels : percentage perturbation levels to test

    Returns
    -------
    SensitivityResult dataclass

    Raises
    ------
    SensitivityError on insufficient data.
    """
    valid_filenames = {fn for fn, flag in quality_flags.items() if flag == "VALID"}
    valid_df = analysis_df[
        analysis_df["filename"].isin(valid_filenames)
        & (analysis_df["detectionSuccess"] == True)
        & analysis_df["markerWidthPx"].notna()
        & analysis_df["distanceMeters"].notna()
        & (analysis_df["markerWidthPx"] > 0)
        & (analysis_df["distanceMeters"] > 0)
    ].copy()

    n = len(valid_df)
    if n < 2:
        raise SensitivityError(f"Need at least 2 VALID samples; got {n}.")

    actual_distances = valid_df["distanceMeters"].values
    baseline_widths  = valid_df["markerWidthPx"].values

    baseline_predicted = _predict_distances(model, baseline_widths)
    baseline_mae  = float(mean_absolute_error(actual_distances, baseline_predicted))
    baseline_rmse = float(np.sqrt(mean_squared_error(actual_distances, baseline_predicted)))

    perturbations = []
    for pct in perturbation_levels:
        for direction, sign in [("positive", +1.0), ("negative", -1.0)]:
            factor = 1.0 + sign * pct / 100.0
            perturbed_widths    = baseline_widths * factor
            perturbed_predicted = _predict_distances(model, perturbed_widths)

            mae  = float(mean_absolute_error(actual_distances, perturbed_predicted))
            rmse = float(np.sqrt(mean_squared_error(actual_distances, perturbed_predicted)))
            mean_abs_delta = float(np.mean(np.abs(perturbed_predicted - baseline_predicted)))

            perturbations.append(PerturbationResult(
                perturbation_pct=pct,
                direction=direction,
                mae=round(mae, 4),
                rmse=round(rmse, 4),
                mean_abs_delta=round(mean_abs_delta, 4),
            ))

    return SensitivityResult(
        perturbations=perturbations,
        baseline_mae=round(baseline_mae, 4),
        baseline_rmse=round(baseline_rmse, 4),
        n_samples=n,
    )
