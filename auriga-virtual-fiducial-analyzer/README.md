# Auriga Virtual Fiducial Analyzer

A production-quality research web application for validating the **Virtual Fiducial Hypothesis** that underpins Auriga's monocular navigation engine.

---

## Scientific Purpose

Auriga's navigation system hypothesises that objects resting on the ground can have their distance estimated by scaling a virtual fiducial marker until it aligns with the base of the target object. The required scale factor encodes distance.

This application validates that hypothesis by:

1. Detecting ArUco markers in calibration images.
2. Measuring the geometric relationship between marker appearance and capture distance.
3. Fitting polynomial regression models to quantify that relationship.
4. Interpreting whether the relationship is sufficiently stable for Virtual Fiducial scaling.

---

## Features

- Upload a ZIP of calibration images and a metadata CSV.
- Automatic ArUco marker detection (DICT_4X4_50).
- Six visualisation plots (distance vs width/height/area, orientation robustness, detection rate, distance distribution).
- Polynomial regression with R², RMSE, and MAE.
- Automatic research interpretation of findings.
- Downloadable `analysis_results.csv` and `virtual_fiducial_analysis_report.pdf`.

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
├── app.py                    # Flask application
├── requirements.txt
├── README.md
├── analysis/
│   ├── pipeline.py           # Orchestrates the full analysis
│   ├── detector.py           # ArUco detection
│   ├── regression.py         # Polynomial regression
│   ├── plots.py              # Matplotlib visualisations
│   └── report.py             # PDF report generation
├── templates/
│   ├── base.html
│   ├── index.html            # Upload page
│   ├── dashboard.html        # Results dashboard
│   └── error.html
├── tests/
│   └── test_analysis.py
├── uploads/                  # (runtime — not committed)
├── temp/                     # (runtime — not committed)
└── exports/                  # (runtime — not committed)
```

---

## Screenshots

*(Add screenshots of the upload page and dashboard here)*

---

## Part of the Auriga Navigation Project

This tool is a standalone research instrument within the broader Auriga monocular navigation engine development pipeline.
