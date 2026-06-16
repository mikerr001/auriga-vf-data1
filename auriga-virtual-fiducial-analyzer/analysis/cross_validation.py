"""
cross_validation.py
-------------------
K-fold cross-validation on VALID calibration samples.
Returns per-fold metrics and mean ± std summary.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class CrossValidationError(Exception):
    pass


@dataclass
class FoldMetrics:
    fold: int
    n_train: int
    n_test: int
    mae: float
    rmse: float
    r2: float


@dataclass
class CrossValidationResult:
    k: int
    folds: list
    mean_mae: float
    std_mae: float
    mean_rmse: float
    std_rmse: float
    mean_r2: float
    std_r2: float
    n_samples: int


def run_cross_validation(
    analysis_df: pd.DataFrame,
    quality_flags: dict,
    k: int = 5,
) -> CrossValidationResult:
    """
    Run k-fold cross-validation on VALID samples.

    Parameters
    ----------
    analysis_df   : DataFrame from analysis_results.csv
    quality_flags : dict mapping filename -> flag string
    k             : number of folds (default 5; clamped to min(k, n_samples))

    Returns
    -------
    CrossValidationResult dataclass

    Raises
    ------
    CrossValidationError on insufficient data.
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
    if n < 4:
        raise CrossValidationError(
            f"Need at least 4 VALID samples for cross-validation; got {n}."
        )

    k = min(k, n)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    X = valid_df["distanceMeters"].values.reshape(-1, 1)
    y = valid_df["markerWidthPx"].values

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=True),
            LinearRegression(),
        )
        model.fit(X_train, y_train)
        y_pred_px = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred_px))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_px)))
        r2 = float(r2_score(y_test, y_pred_px)) if len(y_test) >= 2 else float("nan")

        folds.append(FoldMetrics(
            fold=fold_idx,
            n_train=len(train_idx),
            n_test=len(test_idx),
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            r2=round(r2, 4),
        ))

    maes  = [f.mae  for f in folds]
    rmses = [f.rmse for f in folds]
    r2s   = [f.r2   for f in folds if not np.isnan(f.r2)]

    return CrossValidationResult(
        k=k,
        folds=folds,
        mean_mae=round(float(np.mean(maes)), 4),
        std_mae=round(float(np.std(maes)), 4),
        mean_rmse=round(float(np.mean(rmses)), 4),
        std_rmse=round(float(np.std(rmses)), 4),
        mean_r2=round(float(np.mean(r2s)) if r2s else float("nan"), 4),
        std_r2=round(float(np.std(r2s)) if r2s else float("nan"), 4),
        n_samples=n,
    )
