"""
Convert PDD RT Dose DICOM files into CSV files.

1. Put your DICOM folder path in DICOM_FOLDER.
2. Put your desired CSV output folder path in CSV_OUTPUT_FOLDER.
3. Run the script.

For each .dcm file, the script samples dose down the central axis and writes
one CSV with the same name as the DICOM file.
"""

from pathlib import Path
import csv

import numpy as np
import pydicom
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

# Folder containing the PDD DICOM dose files.
DICOM_FOLDER = Path('/Users/parkernew/Code/work/RS Project/Final/PDD/12MeV/dcm')

# Folder where the CSV files should be written.
CSV_OUTPUT_FOLDER = Path('/Users/parkernew/Code/work/RS Project/Final/PDD/12MeV/dcm-CSV')

# The PDD line starts at the phantom surface and goes deeper into the phantom.
# These values come from the current RayStation PDD export geometry.
# -377.1 for 100 SSD, -427.1 for 105 SSD, -477.1 for 110 SSD - WRONG
SURFACE_Y_MM = -377.1
DEEPEST_DEPTH_MM = 100.0

# Central-axis location in the DICOM patient coordinate system.
CENTRAL_AXIS_X_MM = 0.6
CENTRAL_AXIS_Z_MM = 0.0

# 1051 points gives one dose sample every 0.1 mm from 0 to 105 mm.
NUMBER_OF_DEPTH_POINTS = 1051


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def read_dicom_dose_grid(dicom_path):
    """Read one DICOM dose file and return its x, y, z positions and dose grid."""
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


def calculate_pdd(dicom_path):
    """Sample dose from the surface down the central axis of one DICOM file."""
    x_positions_mm, y_positions_mm, z_positions_mm, dose_grid_gy = read_dicom_dose_grid(dicom_path)

    # This object lets us ask for dose values between the stored grid points.
    dose_at_point = RegularGridInterpolator(
        (z_positions_mm, y_positions_mm, x_positions_mm),
        dose_grid_gy,
        bounds_error=False,
        fill_value=np.nan,
    )

    depth_mm = np.linspace(0.0, DEEPEST_DEPTH_MM, NUMBER_OF_DEPTH_POINTS)

    # Depth increases by moving in the patient y direction from the surface.
    y_sample_positions_mm = SURFACE_Y_MM + depth_mm
    x_sample_positions_mm = np.full_like(depth_mm, CENTRAL_AXIS_X_MM)
    z_sample_positions_mm = np.full_like(depth_mm, CENTRAL_AXIS_Z_MM)

    # RegularGridInterpolator expects points in the same order as the dose grid:
    # z first, then y, then x.
    sample_points = np.column_stack(
        [z_sample_positions_mm, y_sample_positions_mm, x_sample_positions_mm]
    )
    dose_gy = dose_at_point(sample_points)

    maximum_dose_gy = np.nanmax(dose_gy)
    pdd_percent = dose_gy / maximum_dose_gy * 100.0

    return depth_mm, dose_gy, pdd_percent


def write_pdd_csv(csv_path, depth_mm, dose_gy, pdd_percent):
    """Write one PDD curve to a CSV file."""
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["Depth [mm]", "Dose [Gy]", "PDD [%]"])

        for depth, dose, pdd in zip(depth_mm, dose_gy, pdd_percent):
            writer.writerow([f"{depth:.10g}", f"{dose:.10g}", f"{pdd:.10g}"])


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main():
    """Convert every .dcm file in DICOM_FOLDER."""
    dicom_files = sorted(DICOM_FOLDER.glob("*.dcm"))

    if not dicom_files:
        print(f"No .dcm files were found in: {DICOM_FOLDER}")
        return

    CSV_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("PDD DICOM to CSV conversion")
    print(f"DICOM folder: {DICOM_FOLDER}")
    print(f"CSV folder:   {CSV_OUTPUT_FOLDER}")
    print()

    for dicom_path in dicom_files:
        depth_mm, dose_gy, pdd_percent = calculate_pdd(dicom_path)

        csv_path = CSV_OUTPUT_FOLDER / f"{dicom_path.stem}.csv"
        write_pdd_csv(csv_path, depth_mm, dose_gy, pdd_percent)

        maximum_dose_index = int(np.nanargmax(dose_gy))
        maximum_dose = dose_gy[maximum_dose_index]
        depth_of_maximum_dose = depth_mm[maximum_dose_index]

        print(
            f"{dicom_path.name} -> {csv_path.name} "
            f"(maximum dose {maximum_dose:.6g} Gy at {depth_of_maximum_dose:.1f} mm)"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
