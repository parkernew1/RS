#!/usr/bin/env python3
"""
Convert profile RT Dose DICOM files into CSV files.

1. Put your DICOM folder path in DICOM_FOLDER.
2. Put your desired CSV output folder path in CSV_OUTPUT_FOLDER.
3. Run the script.

For each .dcm file, the script samples a horizontal dose profile at a chosen
depth below the phantom surface. By default, that depth is 3 mm.
"""

from pathlib import Path
import csv

import numpy as np
import pydicom
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

# Folder containing the profile DICOM dose files.
DICOM_FOLDER = Path('/Users/parkernew/Code/work/RS Project/Final/Profile/12MeV/dcm')

# Folder where the CSV files should be written.
CSV_OUTPUT_FOLDER = Path('/Users/parkernew/Code/work/RS Project/Final/Profile/12MeV/dcm-CSV')

# Depth of the profile below the phantom surface.
# Example: 3.0 means "sample the profile at 3 mm depth."
PROFILE_DEPTH_MM = 3.0

# The y position of the phantom surface in the DICOM patient coordinate system.
# The profile is sampled at SURFACE_Y_MM + PROFILE_DEPTH_MM.
SURFACE_Y_MM = -377.1

# Central-axis location in the DICOM patient coordinate system.
# The output X values are relative to CENTRAL_AXIS_X_MM, so the central axis is
# written as X = 0 mm in the CSV.
CENTRAL_AXIS_X_MM = 0.6
CENTRAL_AXIS_Z_MM = 0.0

# The profile runs from -PROFILE_HALF_WIDTH_MM to +PROFILE_HALF_WIDTH_MM.
PROFILE_HALF_WIDTH_MM = 100.0

# 2001 points gives one dose sample every 0.1 mm across a 200 mm profile.
NUMBER_OF_PROFILE_POINTS = 2001


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_dicom_files(folder):
    """Return all DICOM files in a folder."""
    dicom_files = []
    for pattern in ("*.dcm", "*.DCM"):
        dicom_files.extend(folder.glob(pattern))
    return sorted(dicom_files)


def read_dicom_dose_grid(dicom_path):
    """Read one DICOM dose file and return its coordinates and dose grid."""
    dicom = pydicom.dcmread(dicom_path)

    # DICOM stores dose as integer pixel values. DoseGridScaling converts those
    # integers into dose in Gy.
    dose_grid_gy = dicom.pixel_array.astype(float) * float(dicom.DoseGridScaling)

    # ImagePositionPatient is the x, y, z position of the first dose-grid point.
    first_x_mm = float(dicom.ImagePositionPatient[0])
    first_y_mm = float(dicom.ImagePositionPatient[1])
    first_z_mm = float(dicom.ImagePositionPatient[2])

    # PixelSpacing contains [row spacing, column spacing]. In this dose grid,
    # rows move in y and columns move in x.
    y_spacing_mm = float(dicom.PixelSpacing[0])
    x_spacing_mm = float(dicom.PixelSpacing[1])

    # The dose grid is indexed as [z, y, x].
    z_positions_mm = first_z_mm + np.asarray(dicom.GridFrameOffsetVector, dtype=float)
    y_positions_mm = first_y_mm + y_spacing_mm * np.arange(dose_grid_gy.shape[1])
    x_positions_mm = first_x_mm + x_spacing_mm * np.arange(dose_grid_gy.shape[2])

    return x_positions_mm, y_positions_mm, z_positions_mm, dose_grid_gy


def calculate_profile(dicom_path):
    """Sample one horizontal dose profile at PROFILE_DEPTH_MM."""
    x_positions_mm, y_positions_mm, z_positions_mm, dose_grid_gy = read_dicom_dose_grid(
        dicom_path
    )

    # This object lets us ask for dose values between the stored grid points.
    dose_at_point = RegularGridInterpolator(
        (z_positions_mm, y_positions_mm, x_positions_mm),
        dose_grid_gy,
        bounds_error=False,
        fill_value=np.nan,
    )

    relative_x_mm = np.linspace(
        -PROFILE_HALF_WIDTH_MM,
        PROFILE_HALF_WIDTH_MM,
        NUMBER_OF_PROFILE_POINTS,
    )

    x_sample_positions_mm = CENTRAL_AXIS_X_MM + relative_x_mm
    y_sample_positions_mm = np.full_like(
        relative_x_mm,
        SURFACE_Y_MM + PROFILE_DEPTH_MM,
    )
    z_sample_positions_mm = np.full_like(relative_x_mm, CENTRAL_AXIS_Z_MM)

    # RegularGridInterpolator expects points in the same order as the dose grid:
    # z first, then y, then x.
    sample_points = np.column_stack(
        [z_sample_positions_mm, y_sample_positions_mm, x_sample_positions_mm]
    )
    dose_gy = dose_at_point(sample_points)

    return relative_x_mm, dose_gy


def write_profile_csv(csv_path, x_mm, dose_gy):
    """Write one profile curve to a CSV file."""
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["X [mm]", "Dose [Gy]"])

        for x_value, dose_value in zip(x_mm, dose_gy):
            writer.writerow([f"{x_value:.10g}", f"{dose_value:.10g}"])


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main():
    """Convert every .dcm file in DICOM_FOLDER."""
    dicom_files = find_dicom_files(DICOM_FOLDER)

    if not dicom_files:
        print(f"No .dcm files were found in: {DICOM_FOLDER}")
        return

    CSV_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("Profile DICOM to CSV conversion")
    print(f"DICOM folder:  {DICOM_FOLDER}")
    print(f"CSV folder:    {CSV_OUTPUT_FOLDER}")
    print(f"Profile depth: {PROFILE_DEPTH_MM:g} mm")
    print(f"Surface y:     {SURFACE_Y_MM:g} mm")
    print()

    for dicom_path in dicom_files:
        x_mm, dose_gy = calculate_profile(dicom_path)

        csv_path = CSV_OUTPUT_FOLDER / f"{dicom_path.stem}.csv"
        write_profile_csv(csv_path, x_mm, dose_gy)

        maximum_dose = np.nanmax(dose_gy)
        print(
            f"{dicom_path.name} -> {csv_path.name} "
            f"(maximum profile dose {maximum_dose:.6g} Gy)"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
