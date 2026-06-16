"""
recommendation.py
-----------------
Applies threshold rules to prediction metrics and orientation ANOVA to produce
a Go / No-Go verdict with bullet-point rationale.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RecommendationError(Exception):
    pass


@dataclass
class RecommendationResult:
    verdict: str
    rationale: list
    r2: float
    mae: float
    rmse: float
    mape: float
    orientation_anova_p: float
    thresholds: dict


_THRESHOLDS = {
    "r2_go":        0.90,
    "mae_go":       0.15,
    "rmse_go":      0.20,
    "mape_go":      10.0,
    "anova_p_good": 0.05,
}


def _orientation_anova_p(results_df: pd.DataFrame) -> float:
    """
    One-way ANOVA p-value for absoluteError grouped by orientation.
    Returns 1.0 if orientation column missing or insufficient groups.
    """
    if "orientation" not in results_df.columns or "absoluteError" not in results_df.columns:
        return 1.0

    groups = [
        grp["absoluteError"].dropna().values
        for _, grp in results_df.groupby("orientation")
        if len(grp) >= 2
    ]
    if len(groups) < 2:
        return 1.0

    try:
        from scipy import stats
        stat, p = stats.f_oneway(*groups)
        return float(p) if not np.isnan(p) else 1.0
    except Exception:
        return 1.0


def generate_recommendation(
    metrics,
    results_df: pd.DataFrame,
    thresholds: dict = None,
) -> RecommendationResult:
    """
    Generate a Go / No-Go verdict.

    Parameters
    ----------
    metrics    : PredictionMetrics dataclass (has .r2, .mae, .rmse, .mape)
    results_df : DataFrame from PredictionResult.results_df
    thresholds : optional override dict for thresholds

    Returns
    -------
    RecommendationResult dataclass
    """
    t = dict(_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    rationale = []
    go_flags  = []

    r2   = metrics.r2
    mae  = metrics.mae
    rmse = metrics.rmse
    mape = metrics.mape

    if r2 >= t["r2_go"]:
        rationale.append(f"R² = {r2:.4f} ≥ {t['r2_go']} — strong predictive relationship confirmed.")
        go_flags.append(True)
    else:
        rationale.append(f"R² = {r2:.4f} < {t['r2_go']} — predictive relationship is insufficient.")
        go_flags.append(False)

    if mae <= t["mae_go"]:
        rationale.append(f"MAE = {mae:.4f} m ≤ {t['mae_go']} m — acceptable mean absolute error.")
        go_flags.append(True)
    else:
        rationale.append(f"MAE = {mae:.4f} m > {t['mae_go']} m — mean absolute error exceeds threshold.")
        go_flags.append(False)

    if rmse <= t["rmse_go"]:
        rationale.append(f"RMSE = {rmse:.4f} m ≤ {t['rmse_go']} m — acceptable root-mean-square error.")
        go_flags.append(True)
    else:
        rationale.append(f"RMSE = {rmse:.4f} m > {t['rmse_go']} m — RMSE exceeds acceptable threshold.")
        go_flags.append(False)

    if mape <= t["mape_go"]:
        rationale.append(f"MAPE = {mape:.2f}% ≤ {t['mape_go']}% — percentage error is within acceptable range.")
        go_flags.append(True)
    else:
        rationale.append(f"MAPE = {mape:.2f}% > {t['mape_go']}% — percentage error exceeds threshold.")
        go_flags.append(False)

    anova_p = _orientation_anova_p(results_df)
    if anova_p > t["anova_p_good"]:
        rationale.append(
            f"Orientation ANOVA p = {anova_p:.4f} > {t['anova_p_good']} — "
            "prediction errors are consistent across orientations."
        )
        go_flags.append(True)
    else:
        rationale.append(
            f"Orientation ANOVA p = {anova_p:.4f} ≤ {t['anova_p_good']} — "
            "prediction errors differ significantly across orientations."
        )
        go_flags.append(False)

    verdict = "Go" if all(go_flags) else "No-Go"

    if verdict == "Go":
        rationale.append(
            "All thresholds passed. Recommend proceeding to calibration LUT deployment."
        )
    else:
        failed = sum(1 for f in go_flags if not f)
        rationale.append(
            f"{failed} of {len(go_flags)} criteria failed. "
            "Address flagged issues before proceeding to LUT deployment."
        )

    return RecommendationResult(
        verdict=verdict,
        rationale=rationale,
        r2=r2,
        mae=mae,
        rmse=rmse,
        mape=mape,
        orientation_anova_p=round(anova_p, 6),
        thresholds=t,
    )
