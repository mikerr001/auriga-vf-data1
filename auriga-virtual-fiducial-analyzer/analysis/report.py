"""
report.py
---------
PDF report generation for the Virtual Fiducial analysis.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)

from .regression import RegressionMetrics

logger = logging.getLogger(__name__)

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm
ACCENT_HEX = colors.HexColor("#4A90D9")


def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("AurigaTitle",
                         parent=s["Title"],
                         fontSize=20, textColor=ACCENT_HEX,
                         spaceAfter=6))
    s.add(ParagraphStyle("AurigaH1",
                         parent=s["Heading1"],
                         fontSize=14, textColor=ACCENT_HEX,
                         spaceBefore=14, spaceAfter=4))
    s.add(ParagraphStyle("AurigaH2",
                         parent=s["Heading2"],
                         fontSize=11, textColor=colors.HexColor("#333333"),
                         spaceBefore=8, spaceAfter=2))
    s.add(ParagraphStyle("AurigaBody",
                         parent=s["Normal"],
                         fontSize=10, leading=15))
    return s


def _table_style(header_bg=ACCENT_HEX):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#F7F9FC"), colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])


def generate_report(df: pd.DataFrame,
                    regressions: dict[str, Optional[RegressionMetrics]],
                    plot_paths: dict[str, str],
                    export_dir: Path,
                    out_filename: str = "virtual_fiducial_analysis_report.pdf") -> Path:
    """
    Build the full PDF report and save it to *export_dir/out_filename*.
    Returns the path to the generated file.
    """
    out_path = export_dir / out_filename
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN)
    S = _styles()
    story = []

    # ------------------------------------------------------------------ cover
    story.append(Paragraph("Auriga Virtual Fiducial Analyzer", S["AurigaTitle"]))
    story.append(Paragraph("Virtual Fiducial Hypothesis — Validation Report", S["AurigaH2"]))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                           S["AurigaBody"]))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT_HEX, spaceAfter=12))

    # -------------------------------------------------------- executive summary
    story.append(Paragraph("1. Executive Summary", S["AurigaH1"]))
    total = len(df)
    detected = int(df["detectionSuccess"].sum())
    rate = detected / total * 100 if total else 0
    story.append(Paragraph(
        f"This report presents the results of an automated geometric analysis "
        f"performed on a calibration dataset of <b>{total}</b> images. "
        f"ArUco markers were successfully detected in <b>{detected}</b> images "
        f"({rate:.1f}% detection rate). "
        f"Regression analysis was applied to evaluate whether Virtual Fiducial "
        f"scaling produces a stable geometric relationship with distance.",
        S["AurigaBody"]))
    story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------------ dataset statistics
    story.append(Paragraph("2. Dataset Statistics", S["AurigaH1"]))
    dist_counts = df["distanceMeters"].value_counts().sort_index()
    ori_counts = df["orientation"].value_counts()
    stats_data = [["Metric", "Value"],
                  ["Total images",     str(total)],
                  ["Successful detections", str(detected)],
                  ["Detection rate",   f"{rate:.1f}%"],
                  ["Unique distances", str(df["distanceMeters"].nunique())],
                  ["Unique orientations", str(df["orientation"].nunique())],
                  ["Camera devices",   str(df["deviceName"].nunique())],
                  ]
    t = Table(stats_data, colWidths=[9 * cm, 7 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    # distance dist plot
    ddist = plot_paths.get("distance_dist")
    if ddist and (export_dir / ddist).exists():
        story.append(RLImage(str(export_dir / ddist), width=14 * cm, height=8 * cm))
    story.append(Spacer(1, 0.4 * cm))

    # ----------------------------------------------------- detection statistics
    story.append(Paragraph("3. Detection Statistics by Orientation", S["AurigaH1"]))
    detected_df = df[df["detectionSuccess"] == True]
    det_data = [["Orientation", "Total", "Detected", "Rate",
                 "Avg Width (px)", "Avg Height (px)"]]
    for ori in sorted(df["orientation"].dropna().unique()):
        sub_all = df[df["orientation"] == ori]
        sub_det = detected_df[detected_df["orientation"] == ori]
        n = len(sub_all)
        nd = len(sub_det)
        r = f"{nd/n*100:.0f}%" if n else "—"
        aw = f"{sub_det['markerWidthPx'].mean():.1f}" if nd else "—"
        ah = f"{sub_det['markerHeightPx'].mean():.1f}" if nd else "—"
        det_data.append([ori, str(n), str(nd), r, aw, ah])
    t2 = Table(det_data, colWidths=[3.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm, 3*cm])
    t2.setStyle(_table_style())
    story.append(t2)
    story.append(Spacer(1, 0.4 * cm))

    drate = plot_paths.get("detection_rate")
    if drate and (export_dir / drate).exists():
        story.append(RLImage(str(export_dir / drate), width=14 * cm, height=8 * cm))

    story.append(PageBreak())

    # ---------------------------------------------------------- scatter plots
    story.append(Paragraph("4. Geometric Relationship Plots", S["AurigaH1"]))
    for key, label in [("dist_width", "Distance vs Marker Width"),
                       ("dist_height", "Distance vs Marker Height"),
                       ("dist_area", "Distance vs Marker Area"),
                       ("orientation", "Orientation Robustness")]:
        pname = plot_paths.get(key)
        if pname and (export_dir / pname).exists():
            story.append(Paragraph(label, S["AurigaH2"]))
            story.append(RLImage(str(export_dir / pname), width=14*cm, height=8.5*cm))
            story.append(Spacer(1, 0.3*cm))

    story.append(PageBreak())

    # --------------------------------------------------- regression analysis
    story.append(Paragraph("5. Regression Analysis", S["AurigaH1"]))
    story.append(Paragraph(
        "A polynomial (degree-2) regression was fitted to each geometric metric "
        "as a function of capture distance. R², RMSE, and MAE are reported below.",
        S["AurigaBody"]))
    story.append(Spacer(1, 0.3*cm))

    reg_data = [["Metric", "Model", "R²", "RMSE", "MAE", "Samples"]]
    labels = {"markerWidthPx": "Marker Width",
              "markerHeightPx": "Marker Height",
              "markerAreaPx": "Marker Area"}
    for col, met in regressions.items():
        if met:
            reg_data.append([labels.get(col, col), met.model_type,
                             f"{met.r2:.4f}", f"{met.rmse:.2f}",
                             f"{met.mae:.2f}", str(met.n_samples)])
        else:
            reg_data.append([labels.get(col, col), "—", "—", "—", "—", "—"])
    t3 = Table(reg_data, colWidths=[4*cm, 3.5*cm, 2*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    t3.setStyle(_table_style())
    story.append(t3)
    story.append(Spacer(1, 0.5*cm))

    # --------------------------------------------------------- interpretation
    story.append(Paragraph("6. Interpretation of Findings", S["AurigaH1"]))
    for col, met in regressions.items():
        if met:
            story.append(Paragraph(f"<b>{labels.get(col, col)}:</b> {met.interpretation}",
                                   S["AurigaBody"]))
            story.append(Spacer(1, 0.2*cm))

    # --------------------------------------------------------- recommendations
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("7. Recommendations", S["AurigaH1"]))
    recs = [
        "Increase dataset size across all distances and orientations for stronger statistical confidence.",
        "Ensure consistent lighting conditions across all calibration sessions.",
        "Collect data at additional intermediate distances to refine the polynomial fit.",
        "If R² ≥ 0.95 across all metrics, proceed to calibration lookup table construction.",
        "Consider multi-camera validation to test device-independence of the Virtual Fiducial approach.",
    ]
    for i, rec in enumerate(recs, 1):
        story.append(Paragraph(f"{i}. {rec}", S["AurigaBody"]))
        story.append(Spacer(1, 0.1*cm))

    doc.build(story)
    logger.info("PDF report saved to %s", out_path)
    return out_path
