#!/usr/bin/env python3
"""
Convert Octavius MCC profile files into central-axis profile CSV files.

1. Read one folder of Octavius .mcc files.
2. Parse each MCC file into detector rows.
3. Rebuild a rectangular dose map while respecting the staggered detector rows.
4. Pull out the cross-plane profile closest to the central in-plane axis.
5. Write one CSV per MCC file, using the same stem as the MCC file.

The user-editable settings are grouped at the top of the file.
"""

from pathlib import Path
import csv
import re

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

# Folder containing the Octavius MCC files for one energy.
MCC_FOLDER = Path('/Users/parkernew/Code/work/RS Project/Final/Profile/6MeV/mcc')

# Folder where the CSV files should be written.
CSV_OUTPUT_FOLDER = Path('/Users/parkernew/Code/work/RS Project/Final/Profile/6MeV/mcc-CSV')

# If this is None, every MCC file in MCC_FOLDER is converted.
# To convert only one file, enter its full path here.
SINGLE_MCC_FILE = None
# Example:
# SINGLE_MCC_FILE = Path('/Users/parkernew/Code/work/RS Project/Final/Profile/6MeV/mcc/6MeV_CC_7point5.mcc')

# The central-axis profile is taken from the detector row closest to this
# in-plane y position. For most centered profiles, keep this at 0.0 mm.
CENTRAL_INPLANE_Y_MM = 0.0

# Set this to True if you want a matplotlib window showing the reconstructed
# 2D dose map for each MCC file.
DISPLAY_DOSE_MAP = True

# Set this to True if you want a PNG dose-map image saved beside each CSV.
SAVE_DOSE_MAP_PNG = True


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

class DetectorRow:
    """One BEGIN_SCAN block from an MCC file."""

    def __init__(self, scan_number, inplane_y_mm, crossplane_x_mm, dose_gy):
        self.scan_number = scan_number
        self.inplane_y_mm = inplane_y_mm
        self.crossplane_x_mm = crossplane_x_mm
        self.dose_gy = dose_gy


# ---------------------------------------------------------------------------
# Reading MCC files
# ---------------------------------------------------------------------------

NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def find_mcc_files(folder):
    """Return all MCC files in a folder."""
    mcc_files = []
    for pattern in ("*.mcc", "*.MCC"):
        mcc_files.extend(folder.glob(pattern))
    return sorted(mcc_files)


def read_mcc_file(mcc_path):
    """Read one Octavius MCC file into detector rows."""
    detector_rows = []

    inside_scan = False
    inside_data = False
    scan_number = None
    inplane_y_mm = None
    crossplane_x_values = []
    dose_values = []

    with mcc_path.open("r", encoding="utf-8", errors="ignore") as mcc_file:
        for raw_line in mcc_file:
            line = raw_line.strip()
            if not line:
                continue

            begin_scan_match = re.fullmatch(r"BEGIN_SCAN\s+(\d+)", line)
            if begin_scan_match:
                inside_scan = True
                inside_data = False
                scan_number = int(begin_scan_match.group(1))
                inplane_y_mm = None
                crossplane_x_values = []
                dose_values = []
                continue

            if not inside_scan:
                continue

            if line.startswith("SCAN_OFFAXIS_INPLANE="):
                inplane_y_mm = float(line.split("=", 1)[1])
                continue

            if line == "BEGIN_DATA":
                inside_data = True
                continue

            if line == "END_DATA":
                inside_data = False
                continue

            if re.fullmatch(r"END_SCAN\s+\d+", line):
                if scan_number is None:
                    raise ValueError(f"{mcc_path.name}: END_SCAN found without BEGIN_SCAN.")
                if inplane_y_mm is None:
                    raise ValueError(f"{mcc_path.name}: scan {scan_number} is missing SCAN_OFFAXIS_INPLANE.")
                if not crossplane_x_values:
                    raise ValueError(f"{mcc_path.name}: scan {scan_number} has no dose data.")

                x_array = np.asarray(crossplane_x_values, dtype=float)
                dose_array = np.asarray(dose_values, dtype=float)
                sort_order = np.argsort(x_array)

                detector_rows.append(
                    DetectorRow(
                        scan_number=scan_number,
                        inplane_y_mm=float(inplane_y_mm),
                        crossplane_x_mm=x_array[sort_order],
                        dose_gy=dose_array[sort_order],
                    )
                )

                inside_scan = False
                inside_data = False
                scan_number = None
                inplane_y_mm = None
                crossplane_x_values = []
                dose_values = []
                continue

            if inside_data:
                data_without_comment = line.split("#", 1)[0].strip()
                numbers = NUMBER_PATTERN.findall(data_without_comment)
                if len(numbers) >= 2:
                    crossplane_x_values.append(float(numbers[0]))
                    dose_values.append(float(numbers[1]))

    if not detector_rows:
        raise ValueError(f"No detector rows were found in {mcc_path}.")

    detector_rows.sort(key=lambda row: row.inplane_y_mm)
    return detector_rows


# ---------------------------------------------------------------------------
# Rebuilding the offset detector grid
# ---------------------------------------------------------------------------

def estimate_common_x_spacing(detector_rows):
    """Estimate the smallest real x spacing present across all detector rows."""
    all_x_positions = np.concatenate([row.crossplane_x_mm for row in detector_rows])
    unique_x_positions = np.unique(np.round(all_x_positions, decimals=8))
    unique_x_positions.sort()

    spacings = np.diff(unique_x_positions)
    spacings = spacings[spacings > 1e-8]
    if len(spacings) == 0:
        raise ValueError("Could not estimate the common x spacing.")

    return float(np.min(spacings))


def build_aligned_dose_map(detector_rows):
    """
    Rebuild a rectangular x-y dose map from the staggered Octavius detector rows.

    Neighboring Octavius rows are offset in x, so some x positions are measured
    on one row but not on the next. The script keeps direct measurements where
    they exist. Missing points are usually filled by vertical interpolation from
    nearby rows that measured the same x coordinate. At the outer edges, where
    a vertical bracket is not available, the script uses same-row horizontal
    interpolation as a fallback.
    """
    y_positions_mm = np.asarray([row.inplane_y_mm for row in detector_rows], dtype=float)

    common_x_min = max(float(np.min(row.crossplane_x_mm)) for row in detector_rows)
    common_x_max = min(float(np.max(row.crossplane_x_mm)) for row in detector_rows)
    if common_x_max <= common_x_min:
        raise ValueError("Detector rows do not share a common x range.")

    x_spacing_mm = estimate_common_x_spacing(detector_rows)
    number_of_steps = int(np.floor((common_x_max - common_x_min) / x_spacing_mm + 1e-9))
    x_positions_mm = common_x_min + x_spacing_mm * np.arange(number_of_steps + 1, dtype=float)

    dose_map_gy = np.full((len(detector_rows), len(x_positions_mm)), np.nan, dtype=float)
    directly_measured = np.zeros(dose_map_gy.shape, dtype=bool)
    interpolation_method = np.full(dose_map_gy.shape, "unfilled", dtype="U32")

    measured_by_row = []
    for row in detector_rows:
        row_measurements = {
            round(float(x), 8): float(dose)
            for x, dose in zip(row.crossplane_x_mm, row.dose_gy)
        }
        measured_by_row.append(row_measurements)

    x_keys = [round(float(x), 8) for x in x_positions_mm]

    for row_index, row_measurements in enumerate(measured_by_row):
        for column_index, x_key in enumerate(x_keys):
            if x_key in row_measurements:
                dose_map_gy[row_index, column_index] = row_measurements[x_key]
                directly_measured[row_index, column_index] = True
                interpolation_method[row_index, column_index] = "direct"

    for row_index, row in enumerate(detector_rows):
        for column_index, x_key in enumerate(x_keys):
            if np.isfinite(dose_map_gy[row_index, column_index]):
                continue

            lower_row_index = None
            upper_row_index = None

            for candidate_index in range(row_index - 1, -1, -1):
                if x_key in measured_by_row[candidate_index]:
                    lower_row_index = candidate_index
                    break

            for candidate_index in range(row_index + 1, len(detector_rows)):
                if x_key in measured_by_row[candidate_index]:
                    upper_row_index = candidate_index
                    break

            if lower_row_index is not None and upper_row_index is not None:
                lower_y = y_positions_mm[lower_row_index]
                upper_y = y_positions_mm[upper_row_index]
                fraction = (y_positions_mm[row_index] - lower_y) / (upper_y - lower_y)

                lower_dose = measured_by_row[lower_row_index][x_key]
                upper_dose = measured_by_row[upper_row_index][x_key]
                dose_map_gy[row_index, column_index] = lower_dose + fraction * (upper_dose - lower_dose)
                interpolation_method[row_index, column_index] = "vertical_offset_rows"
                continue

            dose_map_gy[row_index, column_index] = np.interp(
                x_positions_mm[column_index],
                row.crossplane_x_mm,
                row.dose_gy,
            )
            interpolation_method[row_index, column_index] = "horizontal_edge_fallback"

    return x_positions_mm, y_positions_mm, dose_map_gy, directly_measured, interpolation_method


# ---------------------------------------------------------------------------
# Central-axis profile and output
# ---------------------------------------------------------------------------

def central_profile_from_dose_map(x_positions_mm, y_positions_mm, dose_map_gy, directly_measured, interpolation_method):
    """Return the cross-plane profile from the row closest to CENTRAL_INPLANE_Y_MM."""
    central_row_index = int(np.argmin(np.abs(y_positions_mm - CENTRAL_INPLANE_Y_MM)))

    return (
        float(y_positions_mm[central_row_index]),
        x_positions_mm,
        dose_map_gy[central_row_index, :],
        directly_measured[central_row_index, :],
        interpolation_method[central_row_index, :],
    )


def write_profile_csv(csv_path, central_y_mm, x_mm, dose_gy, directly_measured, interpolation_method):
    """Write one central-axis profile CSV."""
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow([
            "X [mm]",
            "Dose [Gy]",
            "Central Inplane Y [mm]",
            "Directly Measured",
            "Interpolation Method",
        ])

        for x_value, dose_value, measured, method in zip(
            x_mm,
            dose_gy,
            directly_measured,
            interpolation_method,
        ):
            writer.writerow([
                f"{x_value:.10g}",
                f"{dose_value:.10g}",
                f"{central_y_mm:.10g}",
                int(bool(measured)),
                method,
            ])


def display_dose_map(mcc_path, x_positions_mm, y_positions_mm, dose_map_gy, central_y_mm):
    """Show the reconstructed dose map with matplotlib."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(
        dose_map_gy,
        extent=[x_positions_mm[0], x_positions_mm[-1], y_positions_mm[0], y_positions_mm[-1]],
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )
    ax.axhline(central_y_mm, color="white", linewidth=1.2, linestyle="--", label="CSV profile row")
    ax.axvline(0.0, color="white", linewidth=1.2, linestyle=":", label="x = 0 mm")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Dose [Gy]")
    ax.set_title(f"{mcc_path.stem}: reconstructed Octavius dose map")
    ax.set_xlabel("Cross-plane X [mm]")
    ax.set_ylabel("In-plane Y [mm]")
    ax.legend(loc="best")
    fig.tight_layout()

    return fig


def convert_one_mcc_file(mcc_path):
    """Convert one MCC file and return the output CSV path."""
    detector_rows = read_mcc_file(mcc_path)
    x_positions_mm, y_positions_mm, dose_map_gy, directly_measured, interpolation_method = (
        build_aligned_dose_map(detector_rows)
    )

    central_y_mm, profile_x_mm, profile_dose_gy, profile_measured, profile_method = (
        central_profile_from_dose_map(
            x_positions_mm,
            y_positions_mm,
            dose_map_gy,
            directly_measured,
            interpolation_method,
        )
    )

    csv_path = CSV_OUTPUT_FOLDER / f"{mcc_path.stem}.csv"
    write_profile_csv(
        csv_path,
        central_y_mm,
        profile_x_mm,
        profile_dose_gy,
        profile_measured,
        profile_method,
    )

    if DISPLAY_DOSE_MAP or SAVE_DOSE_MAP_PNG:
        fig = display_dose_map(mcc_path, x_positions_mm, y_positions_mm, dose_map_gy, central_y_mm)

        if SAVE_DOSE_MAP_PNG:
            png_path = CSV_OUTPUT_FOLDER / f"{mcc_path.stem}_dose_map.png"
            fig.savefig(png_path, dpi=200)
            print(f"  Saved dose map: {png_path.name}")

        if DISPLAY_DOSE_MAP:
            plt.show()
        else:
            plt.close(fig)

    maximum_dose_index = int(np.nanargmax(profile_dose_gy))
    maximum_dose = profile_dose_gy[maximum_dose_index]
    maximum_dose_x = profile_x_mm[maximum_dose_index]

    print(
        f"{mcc_path.name} -> {csv_path.name} "
        f"(central y {central_y_mm:g} mm, max {maximum_dose:.6g} Gy at x {maximum_dose_x:g} mm)"
    )

    return csv_path


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main():
    """Convert the selected MCC file or every MCC file in MCC_FOLDER."""
    if SINGLE_MCC_FILE is None:
        mcc_files = find_mcc_files(MCC_FOLDER)
    else:
        mcc_files = [Path(SINGLE_MCC_FILE)]

    if not mcc_files:
        print(f"No MCC files were found in: {MCC_FOLDER}")
        return

    CSV_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("Octavius MCC to central-axis CSV conversion")
    print(f"MCC folder:          {MCC_FOLDER}")
    print(f"CSV output folder:   {CSV_OUTPUT_FOLDER}")
    print(f"Central in-plane y:  {CENTRAL_INPLANE_Y_MM:g} mm")
    print(f"Display dose map:    {DISPLAY_DOSE_MAP}")
    print(f"Save dose map PNG:   {SAVE_DOSE_MAP_PNG}")
    print()

    for mcc_path in mcc_files:
        convert_one_mcc_file(mcc_path)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
