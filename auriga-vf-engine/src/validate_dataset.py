"""
validate_dataset.py
-------------------
Validates auriga-vf-engine/datasets/calibration/metadata.csv against
the images/ folder sitting next to it.

Checks:
  1. Every row in metadata.csv has a matching JPG in images/
  2. Every JPG in images/ has a corresponding row in metadata.csv
  3. Required columns are present and non-empty
  4. Numeric fields (distanceMeters, cameraHeightCm) are valid numbers > 0

Usage:
  python auriga-vf-engine/src/validate_dataset.py
"""

import csv
import os
import sys

CALIBRATION_DIR = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "calibration"
)
METADATA_PATH = os.path.join(CALIBRATION_DIR, "metadata.csv")
IMAGES_DIR = os.path.join(CALIBRATION_DIR, "images")

REQUIRED_COLUMNS = [
    "id", "filename", "objectName", "distanceMeters",
    "cameraHeightCm", "deviceName", "orientation", "capturedAt", "sourceType",
]

NUMERIC_POSITIVE_FIELDS = ["distanceMeters", "cameraHeightCm"]


def load_metadata(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []
    return rows, columns


def images_on_disk(images_dir):
    return {
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg"))
    }


def validate():
    errors = []
    warnings = []

    # --- load ---
    if not os.path.isfile(METADATA_PATH):
        print(f"[FATAL] metadata.csv not found at: {METADATA_PATH}")
        sys.exit(1)

    if not os.path.isdir(IMAGES_DIR):
        print(f"[FATAL] images/ directory not found at: {IMAGES_DIR}")
        sys.exit(1)

    rows, columns = load_metadata(METADATA_PATH)
    disk_images = images_on_disk(IMAGES_DIR)

    print(f"Loaded {len(rows)} rows from metadata.csv")
    print(f"Found  {len(disk_images)} JPGs in images/\n")

    # --- check required columns ---
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    csv_filenames = set()

    for i, row in enumerate(rows, start=2):  # row 1 is header
        row_id = row.get("id", f"<row {i}>")

        # required fields non-empty
        for col in REQUIRED_COLUMNS:
            if col in columns and not str(row.get(col, "")).strip():
                errors.append(f"Row {i} [{row_id}]: '{col}' is empty")

        # numeric positive checks
        for field in NUMERIC_POSITIVE_FIELDS:
            val = row.get(field, "")
            try:
                num = float(val)
                if num <= 0:
                    errors.append(
                        f"Row {i} [{row_id}]: '{field}' must be > 0, got {val}"
                    )
            except (ValueError, TypeError):
                if field in columns:
                    errors.append(
                        f"Row {i} [{row_id}]: '{field}' is not a number: '{val}'"
                    )

        # track CSV filenames for cross-check
        filename = str(row.get("filename", "")).strip()
        if filename:
            csv_filenames.add(filename)

    # --- cross-check: CSV rows vs disk ---
    missing_on_disk = csv_filenames - disk_images
    for f in sorted(missing_on_disk):
        errors.append(f"Image referenced in CSV but missing from images/: {f}")

    orphan_on_disk = disk_images - csv_filenames
    for f in sorted(orphan_on_disk):
        warnings.append(f"Image exists on disk but has no CSV row: {f}")

    # --- report ---
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ⚠  {w}")
        print()

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  ✗  {e}")
        print(f"\nValidation FAILED — {len(errors)} error(s), {len(warnings)} warning(s).")
        sys.exit(1)
    else:
        print(f"Validation PASSED — {len(rows)} rows, {len(disk_images)} images, "
              f"{len(warnings)} warning(s).")
        sys.exit(0)


if __name__ == "__main__":
    validate()
