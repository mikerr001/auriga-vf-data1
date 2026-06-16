"""
lut_report.py
-------------
Generates lut_validation_report.pdf using ReportLab.
Embeds all visualisations, metric tables, verdict, and research-debt section.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)

logger = logging.getLogger(__name__)

PAGE_W, PAGE_H = A4
MARGIN     = 2 * cm
ACCENT_HEX = colors.HexColor("#4A90D9")
GO_HEX     = colors.HexColor("#2e7d32")
NOGO_HEX   = colors.HexColor("#c62828")


def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("AurigaTitle",
                         parent=s["Title"],
                         fontSize=20, textColor=ACCENT_HEX, spaceAfter=6))
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
    s.add(ParagraphStyle("AurigaGo",
                         parent=s["Normal"],
                         fontSize=16, textColor=GO_HEX,
                         fontName="Helvetica-Bold", spaceAfter=4))
    s.add(ParagraphStyle("AurigaNoGo",
                         parent=s["Normal"],
                         fontSize=16, textColor=NOGO_HEX,
                         fontName="Helvetica-Bold", spaceAfter=4))
    return s


def _table_style(header_bg=ACCENT_HEX):
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.HexColor("#F7F9FC"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])


def generate_lut_report(
    analysis_df: pd.DataFrame,
    results_df: pd.DataFrame,
    quality_flags: dict,
    metrics,
    cv_result,
    sensitivity_result,
    recommendation,
    lut_plots: dict,
    export_dir: Path,
    out_filename: str = "lut_validation_report.pdf",
) -> Path:
    """
    Build the LUT Validation PDF report.

    Parameters
    ----------
    analysis_df        : original analysis DataFrame
    results_df         : prediction results DataFrame
    quality_flags      : dict filename -> flag
    metrics            : PredictionMetrics
    cv_result          : CrossValidationResult
    sensitivity_result : SensitivityResult
    recommendation     : RecommendationResult
    lut_plots          : dict plot_key -> filename
    export_dir         : directory for output file
    out_filename       : output PDF filename

    Returns
    -------
    Path to the generated PDF.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / out_filename

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )
    S = _styles()
    story = []

    # ---------------------------------------------------------------- cover
    story.append(Paragraph("Auriga Virtual Fiducial Analyzer", S["AurigaTitle"]))
    story.append(Paragraph("Phase 3 — LUT Validation Report", S["AurigaH2"]))
    story.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        S["AurigaBody"],
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT_HEX, spaceAfter=12))

    # ------------------------------------------------- executive summary
    story.append(Paragraph("1. Executive Summary", S["AurigaH1"]))
    verdict_style = S["AurigaGo"] if recommendation.verdict == "Go" else S["AurigaNoGo"]
    story.append(Paragraph(f"Verdict: {recommendation.verdict}", verdict_style))

    total   = len(analysis_df)
    n_valid = sum(1 for v in quality_flags.values() if v == "VALID")
    n_qust  = sum(1 for v in quality_flags.values() if v == "QUESTIONABLE")
    n_inv   = sum(1 for v in quality_flags.values() if v == "INVALID")

    story.append(Paragraph(
        f"This report covers {total} calibration samples. "
        f"After quality review, {n_valid} were marked VALID, "
        f"{n_qust} QUESTIONABLE, and {n_inv} INVALID. "
        f"Prediction metrics: R²={metrics.r2}, MAE={metrics.mae} m, "
        f"RMSE={metrics.rmse} m, MAPE={metrics.mape}%.",
        S["AurigaBody"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    # --------------------------------------------------- dataset statistics
    story.append(Paragraph("2. Dataset Statistics", S["AurigaH1"]))
    n_detected = int(analysis_df["detectionSuccess"].sum()) if "detectionSuccess" in analysis_df.columns else 0
    det_rate = n_detected / total * 100 if total else 0
    stats_data = [
        ["Metric", "Value"],
        ["Total samples",         str(total)],
        ["Successful detections", str(n_detected)],
        ["Detection rate",        f"{det_rate:.1f}%"],
        ["Unique distances",      str(analysis_df["distanceMeters"].nunique()) if "distanceMeters" in analysis_df.columns else "—"],
        ["Unique orientations",   str(analysis_df["orientation"].nunique()) if "orientation" in analysis_df.columns else "—"],
    ]
    t = Table(stats_data, colWidths=[9 * cm, 7 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------ quality flag summary
    story.append(Paragraph("3. Quality Flag Summary", S["AurigaH1"]))
    flag_data = [
        ["Flag", "Count", "Percentage"],
        ["VALID",        str(n_valid), f"{n_valid/total*100:.1f}%" if total else "—"],
        ["QUESTIONABLE", str(n_qust),  f"{n_qust/total*100:.1f}%"  if total else "—"],
        ["INVALID",      str(n_inv),   f"{n_inv/total*100:.1f}%"   if total else "—"],
    ]
    t2 = Table(flag_data, colWidths=[6 * cm, 5 * cm, 5 * cm])
    t2.setStyle(_table_style())
    story.append(t2)
    story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------- prediction metrics
    story.append(Paragraph("4. Prediction Metrics", S["AurigaH1"]))
    metrics_data = [
        ["Metric", "Value", "Threshold", "Status"],
        ["R²",    f"{metrics.r2:.4f}",   "≥ 0.90",  "✓" if metrics.r2 >= 0.90 else "✗"],
        ["MAE",   f"{metrics.mae:.4f} m","≤ 0.15 m","✓" if metrics.mae <= 0.15 else "✗"],
        ["RMSE",  f"{metrics.rmse:.4f} m","≤ 0.20 m","✓" if metrics.rmse <= 0.20 else "✗"],
        ["MAPE",  f"{metrics.mape:.2f}%","≤ 10%",   "✓" if metrics.mape <= 10.0 else "✗"],
        ["N samples", str(metrics.n_samples), "—", "—"],
    ]
    t3 = Table(metrics_data, colWidths=[4 * cm, 4 * cm, 4 * cm, 4 * cm])
    t3.setStyle(_table_style())
    story.append(t3)
    story.append(Spacer(1, 0.4 * cm))

    # -------------------------------------------------- visualisations
    story.append(Paragraph("5. Visualisations", S["AurigaH1"]))
    plot_labels = {
        "actual_vs_predicted":   "Actual vs Predicted Distance",
        "error_distribution":    "Error Distribution",
        "error_by_orientation":  "Prediction Error by Orientation",
        "marker_width_distance": "Marker Width vs Distance",
    }
    for key, label in plot_labels.items():
        fname = lut_plots.get(key)
        if fname:
            p = export_dir / fname
            if p.exists():
                story.append(Paragraph(label, S["AurigaH2"]))
                story.append(RLImage(str(p), width=14 * cm, height=8 * cm))
                story.append(Spacer(1, 0.3 * cm))

    story.append(PageBreak())

    # -------------------------------------------------- outlier analysis
    story.append(Paragraph("6. Outlier Analysis", S["AurigaH1"]))
    if not results_df.empty and "absoluteError" in results_df.columns:
        try:
            from .residual import detect_outliers
            outlier_result = detect_outliers(results_df)
            story.append(Paragraph(
                f"IQR outliers: {outlier_result.n_iqr_outliers} | "
                f"Z-score outliers (threshold={outlier_result.zscore_threshold}): "
                f"{outlier_result.n_zscore_outliers}",
                S["AurigaBody"],
            ))
            if outlier_result.outlier_filenames:
                story.append(Paragraph(
                    "Outlier filenames: " + ", ".join(outlier_result.outlier_filenames),
                    S["AurigaBody"],
                ))
        except Exception as exc:
            story.append(Paragraph(f"Outlier analysis unavailable: {exc}", S["AurigaBody"]))
    else:
        story.append(Paragraph("Insufficient data for outlier analysis.", S["AurigaBody"]))
    story.append(Spacer(1, 0.4 * cm))

    # ------------------------------------------------ cross-validation
    story.append(Paragraph("7. Cross-Validation Summary", S["AurigaH1"]))
    if cv_result is not None:
        story.append(Paragraph(
            f"{cv_result.k}-fold cross-validation on {cv_result.n_samples} VALID samples:",
            S["AurigaBody"],
        ))
        cv_data = [["Fold", "Train N", "Test N", "MAE", "RMSE", "R²"]]
        for fold in cv_result.folds:
            cv_data.append([
                str(fold.fold), str(fold.n_train), str(fold.n_test),
                f"{fold.mae:.4f}", f"{fold.rmse:.4f}", f"{fold.r2:.4f}",
            ])
        cv_data.append([
            "Mean±Std", "—", "—",
            f"{cv_result.mean_mae:.4f}±{cv_result.std_mae:.4f}",
            f"{cv_result.mean_rmse:.4f}±{cv_result.std_rmse:.4f}",
            f"{cv_result.mean_r2:.4f}±{cv_result.std_r2:.4f}",
        ])
        t4 = Table(cv_data, colWidths=[2*cm, 2.5*cm, 2.5*cm, 3*cm, 3*cm, 3*cm])
        t4.setStyle(_table_style())
        story.append(t4)
    else:
        story.append(Paragraph("Cross-validation was not available.", S["AurigaBody"]))
    story.append(Spacer(1, 0.4 * cm))

    # ----------------------------------------------- sensitivity analysis
    story.append(Paragraph("8. Sensitivity Analysis", S["AurigaH1"]))
    if sensitivity_result is not None:
        story.append(Paragraph(
            f"Baseline MAE={sensitivity_result.baseline_mae} m, "
            f"RMSE={sensitivity_result.baseline_rmse} m. "
            f"Results for markerWidthPx perturbations:",
            S["AurigaBody"],
        ))
        sens_data = [["Perturbation", "Direction", "MAE (m)", "RMSE (m)", "Mean Δ Distance (m)"]]
        for p in sensitivity_result.perturbations:
            sens_data.append([
                f"{p.perturbation_pct}%", p.direction,
                f"{p.mae:.4f}", f"{p.rmse:.4f}", f"{p.mean_abs_delta:.4f}",
            ])
        t5 = Table(sens_data, colWidths=[3*cm, 3*cm, 3*cm, 3*cm, 4*cm])
        t5.setStyle(_table_style())
        story.append(t5)
    else:
        story.append(Paragraph("Sensitivity analysis was not available.", S["AurigaBody"]))
    story.append(Spacer(1, 0.4 * cm))

    # -------------------------------------------- recommendations / verdict
    story.append(Paragraph("9. Recommendation Engine", S["AurigaH1"]))
    story.append(Paragraph(f"Verdict: {recommendation.verdict}", verdict_style))
    for item in recommendation.rationale:
        story.append(Paragraph(f"• {item}", S["AurigaBody"]))
        story.append(Spacer(1, 0.1 * cm))
    story.append(Spacer(1, 0.3 * cm))

    # -------------------------------------------- research debt
    story.append(Paragraph("10. Research Debt Updates", S["AurigaH1"]))
    debt_items = [
        "RD-001: Expand calibration dataset to ≥50 samples per distance tier.",
        "RD-002: Validate LUT across multiple camera devices and lighting conditions.",
        "RD-003: Implement temperature/humidity correction factors for outdoor deployments.",
        "RD-004: Assess LUT stability across firmware versions and camera settings.",
        "RD-005: Investigate non-planar marker placement effects on width measurements.",
    ]
    for item in debt_items:
        story.append(Paragraph(f"• {item}", S["AurigaBody"]))
        story.append(Spacer(1, 0.1 * cm))

    doc.build(story)
    logger.info("LUT validation report saved to %s", out_path)
    return out_path
