"""
plots.py
--------
Generates all visualisation plots for the analysis dashboard.
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
ACCENT = "#4A90D9"
COLORS = ["#4A90D9", "#7BC67E", "#F5A623", "#E74C3C"]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_distance_vs_metric(df: pd.DataFrame, y_col: str,
                            y_label: str, title: str, out_path: Path) -> None:
    """Scatter plot of distanceMeters vs *y_col* coloured by orientation."""
    detected = df[df["detectionSuccess"] == True].copy()
    if detected.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    orientations = sorted(detected["orientation"].dropna().unique())
    for i, ori in enumerate(orientations):
        sub = detected[detected["orientation"] == ori]
        ax.scatter(sub["distanceMeters"], sub[y_col],
                   label=ori, color=COLORS[i % len(COLORS)], s=60, alpha=0.85)

    # best-fit line
    x = detected["distanceMeters"].values
    y = detected[y_col].values
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 2)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, np.polyval(coeffs, x_line),
                "--", color="grey", linewidth=1.5, label="Poly fit")

    ax.set_xlabel("Distance (m)", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def plot_orientation_robustness(df: pd.DataFrame, out_path: Path) -> None:
    """Box plot of markerWidthPx grouped by orientation."""
    detected = df[df["detectionSuccess"] == True].copy()
    if detected.empty:
        return

    orientations = sorted(detected["orientation"].dropna().unique())
    data = [detected[detected["orientation"] == o]["markerWidthPx"].dropna().values
            for o in orientations]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    ax.set_xticks(range(1, len(orientations) + 1))
    ax.set_xticklabels(orientations)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xlabel("Orientation", fontsize=11)
    ax.set_ylabel("Marker Width (px)", fontsize=11)
    ax.set_title("Orientation Robustness — Marker Width Distribution", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_detection_success_rate(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of detection success rate per orientation."""
    orientations = sorted(df["orientation"].dropna().unique())
    rates = []
    for o in orientations:
        sub = df[df["orientation"] == o]
        rate = sub["detectionSuccess"].sum() / len(sub) * 100 if len(sub) else 0
        rates.append(rate)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(orientations, rates,
                  color=COLORS[:len(orientations)], edgecolor="white", width=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 115)
    ax.set_xlabel("Orientation", fontsize=11)
    ax.set_ylabel("Detection Rate (%)", fontsize=11)
    ax.set_title("Detection Success Rate by Orientation", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_distance_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Histogram of distance values in the dataset."""
    fig, ax = plt.subplots(figsize=(7, 4))
    distances = df["distanceMeters"].dropna()
    ax.hist(distances, bins=10, color=ACCENT, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Distance (m)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Capture Distance Distribution", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)


def generate_all_plots(df: pd.DataFrame, export_dir: Path) -> dict[str, str]:
    """
    Generate all required plots, saving them to *export_dir*.
    Returns a dict of plot_key -> filename.
    """
    plots: dict[str, str] = {}

    def _p(name: str) -> Path:
        return export_dir / name

    plot_distance_vs_metric(df, "markerWidthPx", "Marker Width (px)",
                            "Distance vs Marker Width", _p("plot_dist_width.png"))
    plots["dist_width"] = "plot_dist_width.png"

    plot_distance_vs_metric(df, "markerHeightPx", "Marker Height (px)",
                            "Distance vs Marker Height", _p("plot_dist_height.png"))
    plots["dist_height"] = "plot_dist_height.png"

    plot_distance_vs_metric(df, "markerAreaPx", "Marker Area (px²)",
                            "Distance vs Marker Area", _p("plot_dist_area.png"))
    plots["dist_area"] = "plot_dist_area.png"

    plot_orientation_robustness(df, _p("plot_orientation_robustness.png"))
    plots["orientation"] = "plot_orientation_robustness.png"

    plot_detection_success_rate(df, _p("plot_detection_rate.png"))
    plots["detection_rate"] = "plot_detection_rate.png"

    plot_distance_distribution(df, _p("plot_distance_dist.png"))
    plots["distance_dist"] = "plot_distance_dist.png"

    return plots
