"""
residual.py
-----------
Outlier detection on prediction errors using IQR and Z-score methods.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ResidualError(Exception):
    pass


@dataclass
class OutlierResult:
    n_samples: int
    n_iqr_outliers: int
    n_zscore_outliers: int
    outlier_filenames: list
    iqr_threshold_low: float
    iqr_threshold_high: float
    zscore_threshold: float
    outlier_df: pd.DataFrame


def detect_outliers(
    results_df: pd.DataFrame,
    zscore_threshold: float = 2.5,
) -> OutlierResult:
    """
    Detect outliers in prediction errors using IQR and Z-score.

    Parameters
    ----------
    results_df        : DataFrame from PredictionResult.results_df
    zscore_threshold  : Z-score cutoff (default 2.5)

    Returns
    -------
    OutlierResult dataclass

    Raises
    ------
    ResidualError on missing columns or insufficient data.
    """
    if "absoluteError" not in results_df.columns:
        raise ResidualError("Column 'absoluteError' missing from results DataFrame.")
    if len(results_df) < 3:
        raise ResidualError("Need at least 3 samples for outlier detection.")

    errors = results_df["absoluteError"].dropna().values

    q1, q3 = np.percentile(errors, 25), np.percentile(errors, 75)
    iqr = q3 - q1
    iqr_low  = q1 - 1.5 * iqr
    iqr_high = q3 + 1.5 * iqr

    mean_e = np.mean(errors)
    std_e  = np.std(errors)
    zscores = (errors - mean_e) / (std_e if std_e > 0 else 1.0)

    iqr_mask    = (errors < iqr_low) | (errors > iqr_high)
    zscore_mask = np.abs(zscores) > zscore_threshold
    combined    = iqr_mask | zscore_mask

    outlier_df = results_df.copy()
    outlier_df["isOutlier_IQR"]    = iqr_mask
    outlier_df["isOutlier_ZScore"] = zscore_mask
    outlier_df["isOutlier"]        = combined
    outlier_df["zScore"]           = np.round(zscores, 4)

    outlier_names: list = []
    if "filename" in results_df.columns:
        outlier_names = results_df.loc[combined, "filename"].tolist()

    return OutlierResult(
        n_samples=len(errors),
        n_iqr_outliers=int(iqr_mask.sum()),
        n_zscore_outliers=int(zscore_mask.sum()),
        outlier_filenames=outlier_names,
        iqr_threshold_low=round(float(iqr_low), 6),
        iqr_threshold_high=round(float(iqr_high), 6),
        zscore_threshold=zscore_threshold,
        outlier_df=outlier_df,
    )
