# Auriga Virtual Fiducial Analyzer

A production-quality research web application for validating the **Virtual Fiducial Hypothesis** that underpins Auriga's monocular navigation engine.

---

## Scientific Purpose

Auriga's navigation system hypothesises that objects resting on the ground can have their distance estimated by scaling a virtual fiducial marker until it aligns with the base of the target object. The required scale factor encodes distance.

This application validates that hypothesis across three phases:

1. **Phase 1 — Detection**: ArUco marker detection in calibration images.
2. **Phase 2 — Regression Analysis**: Polynomial regression to quantify the relationship between marker appearance and capture distance.
3. **Phase 3 — LUT Generation & Distance Prediction**: Quality-flagged calibration lookup table, inverse-model distance prediction, cross-validation, sensitivity analysis, and a Go/No-Go recommendation engine.

---

## Features

### Phase 1 & 2 (Analysis)
- Upload a ZIP of calibration images and a metadata CSV.
- Automatic ArUco marker detection (DICT_4X4_50).
- Six visualisation plots (distance vs width/height/area, orientation robustness, detection rate, distance distribution).
- Polynomial regression with R², RMSE, and MAE.
- Automatic research interpretation of findings.
- Downloadable `analysis_results.csv` and `virtual_fiducial_analysis_report.pdf`.

### Phase 3 (LUT & Prediction)
- **Quality Review**: Flag each sample as VALID / QUESTIONABLE / INVALID via an interactive table. Bulk-flag controls included.
- **LUT Generation**: Builds `virtual_fiducial_lut.csv` from VALID rows only using a polynomial inverse model.
- **Distance Prediction**: Runs per-sample actualDistance / predictedDistance / absoluteError / percentageError, writes `prediction_results.csv`.
- **Accuracy Metrics**: MAE, RMSE, MAPE, and R² reported on a stat-card dashboard.
- **Visualisations**: Four plots — Actual vs Predicted, Error Distribution (with KDE), Error by Orientation, Marker Width vs Distance.
- **Cross-Validation**: k-fold CV with per-fold and mean ± std metrics.
- **Sensitivity Analysis**: Error growth under ±1%, ±3%, ±5% perturbation of `markerWidthPx`.
- **Recommendation Engine**: Go / No-Go verdict with bullet-point rationale based on R², MAE, RMSE, MAPE, and orientation ANOVA.
- **LUT Validation Report**: Generates `lut_validation_report.pdf` containing Executive Summary, Dataset Statistics, Quality Flag Summary, Prediction Metrics, all visualisations, Outlier Analysis, Cross-Validation, Sensitivity Analysis, Recommendation, and Research Debt section.

---

## Installation

```bash
cd auriga-virtual-fiducial-analyzer
pip install -r requirements.txt
```

---

## Running Locally

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Phase 3 Workflow

After completing a Phase 1/2 analysis on the dashboard:

1. Click **"Review Data Quality →"** at the bottom of the dashboard.
2. On the Quality Review page, toggle each sample to **VALID**, **QUESTIONABLE**, or **INVALID**. Use the bulk-flag buttons for efficiency.
3. Click **"Generate LUT & Run Predictions"**. The app builds the LUT from VALID rows and runs the full prediction pipeline.
4. On the **Prediction Results** page:
   - Review accuracy metrics (R², MAE, RMSE, MAPE).
   - Inspect the four visualisation plots.
   - Check the cross-validation summary and sensitivity analysis table.
   - Read the Recommendation Engine verdict (Go / No-Go).
   - Click **"Generate Full PDF Report"** to download `lut_validation_report.pdf`.

---

## Phase 3 Output Files

| File | Location | Description |
|------|----------|-------------|
| `virtual_fiducial_lut.csv` | `temp/<session>/exports/` | Calibration lookup table (distance → predicted marker width) |
| `prediction_results.csv`   | `temp/<session>/exports/` | Per-sample actual vs predicted distances with error columns |
| `lut_validation_report.pdf`| `temp/<session>/exports/` | Full Phase 3 validation PDF report |
| `lut_actual_vs_predicted.png` | exports | Actual vs Predicted Distance scatter |
| `lut_error_distribution.png`  | exports | Error Distribution histogram with KDE |
| `lut_error_by_orientation.png`| exports | Prediction Error by Orientation box plot |
| `lut_marker_width_vs_distance.png` | exports | Marker Width vs Distance fitted curve |

---

## Expected CSV Format

| Column | Description |
|---|---|
| `filename` | Image filename (must match a file in the ZIP) |
| `objectName` | Name of the target object (e.g. `aruco`) |
| `distanceMeters` | Distance from camera to target in metres |
| `cameraHeightCm` | Camera height above ground in cm |
| `deviceName` | Camera device identifier |
| `orientation` | Shot orientation: Center / Left / Right / Down |
| `notes` | Free-text observation notes (optional) |
| `capturedAt` | ISO 8601 timestamp (optional) |
| `sourceType` | Capture source, e.g. `webcam` (optional) |

**Example:**
```csv
filename,objectName,distanceMeters,cameraHeightCm,deviceName,orientation,notes,capturedAt,sourceType
image001.jpg,aruco,0.5,50,Camera-1,Center,note here,2026-06-16T08:00:00Z,webcam
image002.jpg,aruco,1.0,50,Camera-1,Left,,2026-06-16T08:05:00Z,webcam
```

---

## Running Tests

```bash
cd auriga-virtual-fiducial-analyzer
python -m pytest tests/ -v
```

---

## Project Structure

```
auriga-virtual-fiducial-analyzer/
├── app.py                         # Flask application (Phase 1/2/3 routes)
├── requirements.txt
├── README.md
├── analysis/
│   ├── pipeline.py                # Orchestrates the full detection/regression analysis
│   ├── detector.py                # ArUco detection
│   ├── regression.py              # Polynomial regression
│   ├── plots.py                   # Phase 1/2 Matplotlib visualisations
│   ├── report.py                  # Phase 1/2 PDF report generation
│   ├── lut.py                     # Phase 3: LUT generation from VALID rows
│   ├── prediction.py              # Phase 3: Inverse-model distance prediction
│   ├── cross_validation.py        # Phase 3: k-fold cross-validation
│   ├── residual.py                # Phase 3: Outlier detection (IQR + Z-score)
│   ├── sensitivity.py             # Phase 3: Sensitivity analysis
│   ├── recommendation.py          # Phase 3: Go/No-Go recommendation engine
│   ├── lut_plots.py               # Phase 3: Four prediction visualisation plots
│   └── lut_report.py              # Phase 3: LUT validation PDF report
├── templates/
│   ├── base.html
│   ├── index.html                 # Upload page
│   ├── dashboard.html             # Results dashboard (includes Phase 3 CTA)
│   ├── review.html                # Phase 3: Quality review table
│   ├── prediction_results.html    # Phase 3: Prediction results page
│   └── error.html
├── tests/
│   ├── test_analysis.py           # Phase 1/2 tests
│   └── test_lut_prediction.py     # Phase 3 tests
├── uploads/                       # (runtime — not committed)
├── temp/                          # (runtime — not committed)
└── exports/                       # (runtime — not committed)
```

---

## Part of the Auriga Navigation Project

This tool is a standalone research instrument within the broader Auriga monocular navigation engine development pipeline.
