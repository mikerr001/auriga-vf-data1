"""
regression.py
-------------
Regression analysis for Virtual Fiducial scaling hypothesis validation.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)


@dataclass
class RegressionMetrics:
    target: str
    r2: float
    rmse: float
    mae: float
    n_samples: int
    interpretation: str
    model_type: str = "linear"


def _interpret(r2: float) -> str:
    if r2 >= 0.95:
        return "Excellent geometric consistency observed. Virtual Fiducial scaling is strongly supported."
    if r2 >= 0.90:
        return "Strong relationship observed. Suitable for further investigation."
    if r2 >= 0.80:
        return "Moderate relationship observed. Additional calibration data recommended."
    return "Relationship may be insufficient for reliable Virtual Fiducial scaling."


def fit_regression(df: pd.DataFrame, y_col: str) -> Optional[RegressionMetrics]:
    """
    Fit a polynomial (degree-2) regression of *y_col* vs distanceMeters.

    Returns None if there are fewer than 3 valid data points.
    """
    sub = df[["distanceMeters", y_col]].dropna()
    sub = sub[sub["distanceMeters"] > 0]
    sub = sub[sub[y_col] > 0]

    if len(sub) < 3:
        logger.warning("Not enough data to fit regression for %s", y_col)
        return None

    X = sub["distanceMeters"].values.reshape(-1, 1)
    y = sub[y_col].values

    model = make_pipeline(PolynomialFeatures(degree=2, include_bias=True),
                          LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = float(r2_score(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))

    return RegressionMetrics(
        target=y_col,
        r2=r2,
        rmse=rmse,
        mae=mae,
        n_samples=len(sub),
        interpretation=_interpret(r2),
        model_type="polynomial-2",
    )


def run_all_regressions(df: pd.DataFrame) -> dict[str, Optional[RegressionMetrics]]:
    """Run regressions for width, height, and area vs distance."""
    detected = df[df["detectionSuccess"] == True]
    return {
        "markerWidthPx":  fit_regression(detected, "markerWidthPx"),
        "markerHeightPx": fit_regression(detected, "markerHeightPx"),
        "markerAreaPx":   fit_regression(detected, "markerAreaPx"),
    }
