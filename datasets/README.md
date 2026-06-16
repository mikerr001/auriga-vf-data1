# Auriga Virtual Fiducial Engine — Calibration Dataset

This dataset was collected using the Auriga-VF-Collector internal research tool.

## Structure

```
datasets/
├── images/          # Raw captured images (JPEG/PNG/WebP)
├── metadata.csv     # Per-image metadata
└── collection_report.json  # Aggregated statistics
```

## metadata.csv fields

| Field | Description |
|-------|-------------|
| id | Unique capture UUID |
| filename | Image filename in images/ directory |
| objectName | Name of the object being captured |
| distanceMeters | Ground-truth distance in meters |
| cameraHeightCm | Camera height above floor in centimeters |
| deviceName | Device used for capture |
| orientation | Camera orientation (Center/Left/Right/Down) |
| notes | Optional freeform notes |
| capturedAt | ISO 8601 timestamp |

## Example metadata.csv

```csv
id,filename,objectName,distanceMeters,cameraHeightCm,deviceName,orientation,notes,capturedAt
a1b2c3d4-...,a1b2c3d4-....jpg,RedCube,1.5,120,iPhone 14,Center,Good lighting,2026-06-16T10:00:00.000Z
```

## License

Internal use only — Auriga research team.
