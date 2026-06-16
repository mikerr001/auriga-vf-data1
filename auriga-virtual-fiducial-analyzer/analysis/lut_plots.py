"""
lut_plots.py
------------
Generates four Matplotlib figures for the LUT / prediction results:
1. Actual vs Predicted Distance scatter with identity line
2. Error Distribution histogram with KDE
3. Prediction Error by Orientation box plot
4. Marker Width vs Distance scatter with fitted curve
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PLOT_DPI = 120
ACCENT   = "#4A90D9"
GREEN    = "#7BC67E"
ORANGE   = "#F5A623"
RED      = "#E74C3C"
COLORS   = [ACCENT, GREEN, ORANGE, RED, "#9B59B6", "#1ABC9C"]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(results_df: pd.DataFrame, export_dir: Path) -> str:
    """Scatter plot of actual vs predicted distance with identity line."""
    fname = "lut_actual_vs_predicted.png"
    if results_df.empty or "distanceMeters" not in results_df.columns:
        return fname

    actual    = results_df["distanceMeters"].values
    predicted = results_df["predictedDistance"].values

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted, color=ACCENT, s=60, alpha=0.85, label="Predictions")

    lo = min(actual.min(), predicted.min()) * 0.9
    hi = max(actual.max(), predicted.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1.5, label="Identity line")

    ax.set_xlabel("Actual Distance (m)", fontsize=11)
    ax.set_ylabel("Predicted Distance (m)", fontsize=11)
    ax.set_title("Actual vs Predicted Distance", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, export_dir / fname)
    return fname


def plot_error_distribution(results_df: pd.DataFrame, export_dir: Path) -> str:
    """Histogram of absolute errors with KDE overlay."""
    fname = "lut_error_distribution.png"
    if results_df.empty or "absoluteError" not in results_df.columns:
        return fname

    errors = results_df["absoluteError"].dropna().values

    fig, ax = plt.subplots(figsize=(7, 5))
    n_bins = max(5, min(20, len(errors) // 2))
    ax.hist(errors, bins=n_bins, color=ACCENT, edgecolor="white", alpha=0.75,
            density=True, label="Error histogram")

    if len(errors) >= 3:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(errors)
            x_grid = np.linspace(errors.min(), errors.max(), 200)
            ax.plot(x_grid, kde(x_grid), color=RED, linewidth=2, label="KDE")
        except Exception:
            pass

    ax.axvline(float(np.mean(errors)), color=ORANGE, linestyle="--",
               linewidth=1.5, label=f"Mean={np.mean(errors):.3f}")
    ax.set_xlabel("Absolute Error (m)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Prediction Error Distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, export_dir / fname)
    return fname


def plot_error_by_orientation(results_df: pd.DataFrame, export_dir: Path) -> str:
    """Box plot of absolute prediction error grouped by orientation."""
    fname = "lut_error_by_orientation.png"
    if results_df.empty or "absoluteError" not in results_df.columns:
        return fname
    if "orientation" not in results_df.columns:
        return fname

    orientations = sorted(results_df["orientation"].dropna().unique())
    data = [
        results_df[results_df["orientation"] == o]["absoluteError"].dropna().values
        for o in orientations
    ]
    data = [d for d in data if len(d) > 0]
    if not data:
        return fname

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    ax.set_xticks(range(1, len(orientations) + 1))
    ax.set_xticklabels(orientations)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xlabel("Orientation", fontsize=11)
    ax.set_ylabel("Absolute Error (m)", fontsize=11)
    ax.set_title("Prediction Error by Orientation", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, export_dir / fname)
    return fname


def plot_marker_width_vs_distance(
    analysis_df: pd.DataFrame,
    model,
    export_dir: Path,
) -> str:
    """Scatter of markerWidthPx vs distanceMeters with fitted curve."""
    fname = "lut_marker_width_vs_distance.png"
    detected = analysis_df[
        (analysis_df["detectionSuccess"] == True)
        & analysis_df["markerWidthPx"].notna()
        & analysis_df["distanceMeters"].notna()
    ].copy()

    if detected.empty:
        return fname

    fig, ax = plt.subplots(figsize=(8, 5))
    orientations = sorted(detected["orientation"].dropna().unique()) if "orientation" in detected.columns else []
    if orientations:
        for i, ori in enumerate(orientations):
            sub = detected[detected["orientation"] == ori]
            ax.scatter(sub["distanceMeters"], sub["markerWidthPx"],
                       label=ori, color=COLORS[i % len(COLORS)], s=55, alpha=0.85)
    else:
        ax.scatter(detected["distanceMeters"], detected["markerWidthPx"],
                   color=ACCENT, s=55, alpha=0.85, label="Samples")

    if model is not None:
        try:
            x_range = np.linspace(
                detected["distanceMeters"].min(),
                detected["distanceMeters"].max(),
                200,
            )
            y_fit = model.predict(x_range.reshape(-1, 1))
            ax.plot(x_range, y_fit, "--", color="grey", linewidth=2, label="Fitted curve")
        except Exception:
            pass

    ax.set_xlabel("Distance (m)", fontsize=11)
    ax.set_ylabel("Marker Width (px)", fontsize=11)
    ax.set_title("Marker Width vs Distance (LUT Calibration)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, export_dir / fname)
    return fname


def generate_lut_plots(
    results_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    model,
    export_dir: Path,
) -> dict:
    """
    Generate all four LUT visualisation plots.

    Returns a dict of plot_key -> filename.
    """
    plots = {}
    plots["actual_vs_predicted"]   = plot_actual_vs_predicted(results_df, export_dir)
    plots["error_distribution"]    = plot_error_distribution(results_df, export_dir)
    plots["error_by_orientation"]  = plot_error_by_orientation(results_df, export_dir)
    plots["marker_width_distance"] = plot_marker_width_vs_distance(analysis_df, model, export_dir)
    return plots
