# Auriga VF Engine

Calibration dataset and source repository for the Auriga Visual Feature Engine.

## Directory Structure

```
auriga-vf-engine/
├── datasets/
│   └── calibration/
│       ├── images/          # Recovered JPG calibration frames
│       └── metadata.csv     # Per-image capture metadata
├── notebooks/               # Jupyter / analysis notebooks
└── src/                     # Engine source code
```

## Dataset — `datasets/calibration/`

| Field | Description |
|---|---|
| `id` | Unique capture UUID |
| `filename` | Corresponding JPG filename in `images/` |
| `objectName` | Target object photographed (e.g. `aruco`) |
| `distanceMeters` | Camera-to-target distance in metres |
| `cameraHeightCm` | Camera height above ground in cm |
| `deviceName` | Camera device identifier |
| `orientation` | Shot orientation (Center / Left / Right / Down) |
| `notes` | Free-text observation notes |
| `capturedAt` | ISO 8601 capture timestamp |
| `sourceType` | Capture source (e.g. `webcam`) |

**16 calibration images** are stored in `datasets/calibration/images/`, each named by its UUID matching the `id` column in `metadata.csv`.
