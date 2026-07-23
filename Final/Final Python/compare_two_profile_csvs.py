#!/usr/bin/env python3
"""
Compare one film profile CSV against one DICOM/RayStation profile CSV.

This script reads two profile CSV files, prints agreement metrics, and plots
the profiles together.
the most important user settings are grouped near the top of the file.

The easiest way to use it is to edit the user settings below, then run:

    python compare_two_profile_csvs.py

You can also pass paths from the command line:

    python compare_two_profile_csvs.py path/to/film.csv path/to/dicom.csv
"""

from pathlib import Path
import argparse
import csv
import math

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

# Enter the film profile CSV path here.
FILM_PROFILE_CSV_PATH = Path(
    '/Users/parkernew/Code/work/RS Project/Final/Profile/6MeV/Scans-CSV/Profile_6MeV_10_105_4_Lead_3mm.csv'
)

# Enter the DICOM/RayStation profile CSV path here.
DICOM_PROFILE_CSV_PATH = Path(
    '/Users/parkernew/Code/work/RS Project/Final/Profile/6MeV/dcm-CSV/Profile_6MeV_10_105_4_Lead_3mm.csv'
)

# Enter the folder where saved plots and metric text files should go.
# SAVE_RESULTS is False by default, so this script will not save anything unless
# you intentionally change SAVE_RESULTS to True.
RESULTS_FOLDER = Path(
    "/Users/parkernew/Code/work/RS Project/Final/Profile/6MeV/Results"
)
SAVE_RESULTS = True

# Profiles can point in opposite left/right directions depending on scanning
# and RayStation export orientation. Change one of these to True if a profile
# appears mirrored.
FLIP_FILM_X = False
FLIP_DICOM_X = True

# If True, each profile is shifted so its field center is at x = 0 mm before
# comparison. This is usually helpful because film x_mm starts at the ROI edge,
# while DICOM x_mm is already centered on the central axis.
ALIGN_PROFILES_BY_FIELD_CENTER = True

# DICOM normalization point.
# The DICOM dose at this x position becomes 100%.
# For central-axis normalization, keep this at 0.0 mm.
DICOM_NORMALIZATION_X_MM = 0.0

# Ignore the far tails if desired.
# Example: -80.0 and 80.0 compare only the central 160 mm after any flipping
# and field-center alignment. Use None to keep the full overlapping range.
COMPARISON_MIN_X_MM = -60.0
COMPARISON_MAX_X_MM = None


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

class ProfileCurve:
    """One profile curve loaded from a CSV file."""

    def __init__(self, label, path, x_mm, dose_gy, relative_percent):
        self.label = label
        self.path = path
        self.x_mm = x_mm
        self.dose_gy = dose_gy
        self.relative_percent = relative_percent


# ---------------------------------------------------------------------------
# Reading CSV files
# ---------------------------------------------------------------------------

def normalized_column_name(column_name):
    """Make CSV column matching tolerant of spaces, units, and capitalization."""
    return (
        column_name.lower()
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "percent")
        .replace("/", "")
        .replace("_", "")
        .replace(" ", "")
    )


def find_column(fieldnames, possible_names, required=True):
    """Find one CSV column from a list of acceptable names."""
    normalized_to_original = {
        normalized_column_name(name): name
        for name in fieldnames
    }

    for possible_name in possible_names:
        normalized = normalized_column_name(possible_name)
        if normalized in normalized_to_original:
            return normalized_to_original[normalized]

    if required:
        raise ValueError(
            "Could not find one of these columns in the CSV: "
            + ", ".join(possible_names)
        )

    return None


def read_numeric_columns(path, label, column_options):
    """Read selected numeric columns from one CSV file."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV does not exist: {path}")

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not have a CSV header row.")

        columns = {
            key: find_column(reader.fieldnames, names, required=required)
            for key, names, required in column_options
        }

        values = {key: [] for key in columns}
        for row_number, row in enumerate(reader, start=2):
            try:
                for key, column_name in columns.items():
                    if column_name is None:
                        continue
                    values[key].append(float(row[column_name]))
            except ValueError as error:
                raise ValueError(
                    f"Could not read numeric values from {path} on row {row_number}."
                ) from error

    return {
        key: np.asarray(column_values, dtype=float)
        for key, column_values in values.items()
    }


def sorted_unique(x_mm, values):
    """Sort by x and average duplicate x positions if present."""
    finite = np.isfinite(x_mm) & np.isfinite(values)
    x_mm = x_mm[finite]
    values = values[finite]

    if len(x_mm) == 0:
        raise ValueError("A profile CSV does not contain any valid numeric points.")

    order = np.argsort(x_mm)
    x_mm = x_mm[order]
    values = values[order]

    unique_x, inverse = np.unique(x_mm, return_inverse=True)
    if len(unique_x) == len(x_mm):
        return x_mm, values

    averaged_values = np.zeros_like(unique_x, dtype=float)
    for index in range(len(unique_x)):
        averaged_values[index] = float(np.mean(values[inverse == index]))

    return unique_x, averaged_values


def read_film_profile_csv(path):
    """Read a film profile CSV written by convert_film_profile_to_csv.py."""
    columns = read_numeric_columns(
        path=path,
        label="Film",
        column_options=[
            ("x_mm", ["x_mm", "X [mm]", "X mm"], True),
            ("relative_percent", ["smoothed_percent", "normalized_percent"], True),
            ("dose_gy", ["dose_gy", "Dose [Gy]", "Dose Gy"], False),
        ],
    )

    x_mm, relative_percent = sorted_unique(
        columns["x_mm"],
        columns["relative_percent"],
    )

    dose_gy = columns["dose_gy"]
    if len(dose_gy) == len(columns["x_mm"]):
        dose_x, dose_gy = sorted_unique(columns["x_mm"], dose_gy)
        dose_gy = np.interp(x_mm, dose_x, dose_gy)
    else:
        dose_gy = np.full_like(x_mm, np.nan, dtype=float)

    return ProfileCurve(
        label="Film",
        path=Path(path).expanduser(),
        x_mm=x_mm,
        dose_gy=dose_gy,
        relative_percent=relative_percent,
    )


def read_dicom_profile_csv(path):
    """Read a DICOM profile CSV and normalize dose to relative percent."""
    columns = read_numeric_columns(
        path=path,
        label="DICOM",
        column_options=[
            ("x_mm", ["X [mm]", "X mm", "x_mm"], True),
            ("dose_gy", ["Dose [Gy]", "Dose Gy", "dose_gy"], True),
        ],
    )

    x_mm, dose_gy = sorted_unique(columns["x_mm"], columns["dose_gy"])
    relative_percent = normalize_dicom_dose_at_x_zero(x_mm, dose_gy)

    return ProfileCurve(
        label="DICOM",
        path=Path(path).expanduser(),
        x_mm=x_mm,
        dose_gy=dose_gy,
        relative_percent=relative_percent,
    )


# ---------------------------------------------------------------------------
# Profile calculations
# ---------------------------------------------------------------------------

def crossing_one_side(x_mm, y_percent, level, side):
    """Find one interpolated crossing for a percent level."""
    x = np.asarray(x_mm, dtype=float)
    y = np.asarray(y_percent, dtype=float)
    center_index = int(np.nanargmax(y))

    if side == "left":
        search_indices = range(center_index, 0, -1)
        for index in search_indices:
            y1 = y[index]
            y0 = y[index - 1]
            if (y1 - level) * (y0 - level) <= 0 and y1 != y0:
                fraction = (level - y0) / (y1 - y0)
                return float(x[index - 1] + fraction * (x[index] - x[index - 1]))

    elif side == "right":
        search_indices = range(center_index, len(y) - 1)
        for index in search_indices:
            y0 = y[index]
            y1 = y[index + 1]
            if (y0 - level) * (y1 - level) <= 0 and y1 != y0:
                fraction = (level - y0) / (y1 - y0)
                return float(x[index] + fraction * (x[index + 1] - x[index]))

    else:
        raise ValueError('side must be "left" or "right".')

    return None


def crossing_pair(x_mm, y_percent, level):
    """Find left and right crossings for one percent level."""
    left = crossing_one_side(x_mm, y_percent, level, "left")
    right = crossing_one_side(x_mm, y_percent, level, "right")
    return left, right


def field_center(x_mm, relative_percent):
    """Estimate the field center from profile crossings."""
    for level in [50.0, 20.0, 80.0]:
        left, right = crossing_pair(x_mm, relative_percent, level)
        if left is not None and right is not None:
            return float((left + right) / 2.0)

    max_index = int(np.nanargmax(relative_percent))
    return float(x_mm[max_index])


def normalize_dicom_dose_at_x_zero(x_mm, dose_gy):
    """Normalize DICOM dose so the dose at DICOM_NORMALIZATION_X_MM is 100%."""
    minimum_x = float(np.min(x_mm))
    maximum_x = float(np.max(x_mm))
    if not (minimum_x <= DICOM_NORMALIZATION_X_MM <= maximum_x):
        raise ValueError(
            f"DICOM_NORMALIZATION_X_MM={DICOM_NORMALIZATION_X_MM:g} mm is outside "
            f"the DICOM profile range {minimum_x:g} to {maximum_x:g} mm."
        )

    normalization_dose = float(np.interp(DICOM_NORMALIZATION_X_MM, x_mm, dose_gy))
    if not math.isfinite(normalization_dose) or normalization_dose <= 0:
        raise ValueError("DICOM profile has no positive dose values.")

    return dose_gy / normalization_dose * 100.0


def apply_orientation_and_alignment(film, dicom):
    """Apply user-selected flipping and optional field-center alignment."""
    film_x = film.x_mm.copy()
    dicom_x = dicom.x_mm.copy()

    if FLIP_FILM_X:
        film_x = -film_x
    if FLIP_DICOM_X:
        dicom_x = -dicom_x

    film_x, film_percent = sorted_unique(film_x, film.relative_percent)
    dicom_x, dicom_percent = sorted_unique(dicom_x, dicom.relative_percent)

    if np.any(np.isfinite(film.dose_gy)):
        film_dose_x = -film.x_mm if FLIP_FILM_X else film.x_mm
        film_dose_x, film_dose_gy = sorted_unique(film_dose_x, film.dose_gy)
        film_dose_gy = np.interp(film_x, film_dose_x, film_dose_gy)
    else:
        film_dose_gy = np.full_like(film_x, np.nan)

    if np.any(np.isfinite(dicom.dose_gy)):
        dicom_dose_x = -dicom.x_mm if FLIP_DICOM_X else dicom.x_mm
        dicom_dose_x, dicom_dose_gy = sorted_unique(dicom_dose_x, dicom.dose_gy)
        dicom_dose_gy = np.interp(dicom_x, dicom_dose_x, dicom_dose_gy)
    else:
        dicom_dose_gy = np.full_like(dicom_x, np.nan)

    film_center_before_alignment = field_center(film_x, film_percent)
    dicom_center_before_alignment = field_center(dicom_x, dicom_percent)

    if ALIGN_PROFILES_BY_FIELD_CENTER:
        film_x = film_x - film_center_before_alignment
        dicom_x = dicom_x - dicom_center_before_alignment

    transformed_film = ProfileCurve(
        label=film.label,
        path=film.path,
        x_mm=film_x,
        dose_gy=film_dose_gy,
        relative_percent=film_percent,
    )
    transformed_dicom = ProfileCurve(
        label=dicom.label,
        path=dicom.path,
        x_mm=dicom_x,
        dose_gy=dicom_dose_gy,
        relative_percent=dicom_percent,
    )

    alignment_info = {
        "film_center_before_alignment_mm": film_center_before_alignment,
        "dicom_center_before_alignment_mm": dicom_center_before_alignment,
        "center_difference_before_alignment_mm": (
            film_center_before_alignment - dicom_center_before_alignment
        ),
    }

    return transformed_film, transformed_dicom, alignment_info


def automatic_comparison_range(film, dicom):
    """Choose the x range where both profiles have data."""
    minimum_x = max(float(np.min(film.x_mm)), float(np.min(dicom.x_mm)))
    maximum_x = min(float(np.max(film.x_mm)), float(np.max(dicom.x_mm)))

    if COMPARISON_MIN_X_MM is not None:
        minimum_x = max(minimum_x, float(COMPARISON_MIN_X_MM))
    if COMPARISON_MAX_X_MM is not None:
        maximum_x = min(maximum_x, float(COMPARISON_MAX_X_MM))

    if minimum_x >= maximum_x:
        raise ValueError("The two profiles do not have an overlapping x range.")

    return minimum_x, maximum_x


def interpolate_to_film_positions(film, dicom, minimum_x, maximum_x):
    """Interpolate the DICOM profile onto film x positions."""
    comparison_mask = (
        (film.x_mm >= minimum_x)
        & (film.x_mm <= maximum_x)
    )
    comparison_x = film.x_mm[comparison_mask]
    film_percent = film.relative_percent[comparison_mask]
    dicom_percent = np.interp(comparison_x, dicom.x_mm, dicom.relative_percent)

    return comparison_x, film_percent, dicom_percent


def profile_metrics(curve):
    """Calculate basic profile metrics for one curve."""
    max_index = int(np.nanargmax(curve.relative_percent))
    metrics = {
        "max_percent": float(curve.relative_percent[max_index]),
        "max_x_mm": float(curve.x_mm[max_index]),
    }

    if np.any(np.isfinite(curve.dose_gy)):
        dose_index = int(np.nanargmax(curve.dose_gy))
        metrics["max_dose_gy"] = float(curve.dose_gy[dose_index])
        metrics["max_dose_x_mm"] = float(curve.x_mm[dose_index])
    else:
        metrics["max_dose_gy"] = None
        metrics["max_dose_x_mm"] = None

    for level in [80.0, 50.0, 20.0]:
        left, right = crossing_pair(curve.x_mm, curve.relative_percent, level)
        metrics[f"left_{int(level)}_mm"] = left
        metrics[f"right_{int(level)}_mm"] = right
        if left is None or right is None:
            metrics[f"width_{int(level)}_mm"] = None
        else:
            metrics[f"width_{int(level)}_mm"] = right - left

    left_80 = metrics["left_80_mm"]
    left_20 = metrics["left_20_mm"]
    right_80 = metrics["right_80_mm"]
    right_20 = metrics["right_20_mm"]

    metrics["left_penumbra_80_20_mm"] = (
        None if left_80 is None or left_20 is None else left_80 - left_20
    )
    metrics["right_penumbra_80_20_mm"] = (
        None if right_80 is None or right_20 is None else right_20 - right_80
    )

    return metrics


def calculate_metrics(film, dicom, alignment_info):
    """Calculate simple agreement metrics between film and DICOM profiles."""
    minimum_x, maximum_x = automatic_comparison_range(film, dicom)
    comparison_x, film_percent, dicom_percent = interpolate_to_film_positions(
        film,
        dicom,
        minimum_x,
        maximum_x,
    )

    percent_difference = film_percent - dicom_percent
    absolute_difference = np.abs(percent_difference)

    film_metrics = profile_metrics(film)
    dicom_metrics = profile_metrics(dicom)

    metrics = {
        "film": film_metrics,
        "dicom": dicom_metrics,
        "comparison_min_x_mm": minimum_x,
        "comparison_max_x_mm": maximum_x,
        "number_of_comparison_points": len(comparison_x),
        "mean_difference_percent_points": float(np.mean(percent_difference)),
        "mean_absolute_difference_percent_points": float(np.mean(absolute_difference)),
        "max_absolute_difference_percent_points": float(np.max(absolute_difference)),
        "rms_difference_percent_points": float(np.sqrt(np.mean(percent_difference ** 2))),
        **alignment_info,
    }

    for level in [80, 50, 20]:
        film_width = film_metrics[f"width_{level}_mm"]
        dicom_width = dicom_metrics[f"width_{level}_mm"]
        if film_width is None or dicom_width is None:
            metrics[f"width_{level}_difference_mm"] = None
        else:
            metrics[f"width_{level}_difference_mm"] = film_width - dicom_width

    for side in ["left", "right"]:
        film_penumbra = film_metrics[f"{side}_penumbra_80_20_mm"]
        dicom_penumbra = dicom_metrics[f"{side}_penumbra_80_20_mm"]
        if film_penumbra is None or dicom_penumbra is None:
            metrics[f"{side}_penumbra_difference_mm"] = None
        else:
            metrics[f"{side}_penumbra_difference_mm"] = film_penumbra - dicom_penumbra

    return metrics


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

def format_optional(value, digits=3):
    """Format numbers while keeping missing values readable."""
    if value is None:
        return "not found"
    if not math.isfinite(value):
        return "not finite"
    return f"{value:.{digits}f}"


def build_metrics_text(film, dicom, metrics):
    """Build the same agreement text that appears in the terminal."""
    lines = []

    lines.append(f"Film CSV:  {film.path}")
    lines.append(f"DICOM CSV: {dicom.path}")
    lines.append("")
    lines.append("Profile agreement metrics")
    lines.append("-------------------------")
    lines.append(f"Film x flipped:          {FLIP_FILM_X}")
    lines.append(f"DICOM x flipped:         {FLIP_DICOM_X}")
    lines.append(f"Aligned by field center: {ALIGN_PROFILES_BY_FIELD_CENTER}")
    lines.append(
        "DICOM normalization:     "
        f"dose at x = {DICOM_NORMALIZATION_X_MM:g} mm is 100%"
    )
    lines.append(
        "Field center before alignment: "
        f"film {metrics['film_center_before_alignment_mm']:.3f} mm, "
        f"DICOM {metrics['dicom_center_before_alignment_mm']:.3f} mm, "
        f"difference {metrics['center_difference_before_alignment_mm']:.3f} mm"
    )
    lines.append("")

    lines.append("Maximums")
    lines.append(f"  Film max relative:  {metrics['film']['max_percent']:.3f}% at {metrics['film']['max_x_mm']:.3f} mm")
    lines.append(f"  DICOM max relative: {metrics['dicom']['max_percent']:.3f}% at {metrics['dicom']['max_x_mm']:.3f} mm")
    lines.append(f"  Film max dose:      {format_optional(metrics['film']['max_dose_gy'], 5)} Gy")
    lines.append(f"  DICOM max dose:     {format_optional(metrics['dicom']['max_dose_gy'], 5)} Gy")
    lines.append("")

    lines.append("Point-by-point relative dose comparison")
    lines.append(
        "  Compared x range: "
        f"{metrics['comparison_min_x_mm']:.3f} to "
        f"{metrics['comparison_max_x_mm']:.3f} mm"
    )
    lines.append(f"  Comparison points: {metrics['number_of_comparison_points']}")
    lines.append(
        "  Mean difference:   "
        f"{metrics['mean_difference_percent_points']:.3f} percentage points"
    )
    lines.append(
        "  Mean abs diff:     "
        f"{metrics['mean_absolute_difference_percent_points']:.3f} percentage points"
    )
    lines.append(
        "  Max abs diff:      "
        f"{metrics['max_absolute_difference_percent_points']:.3f} percentage points"
    )
    lines.append(
        "  RMS difference:    "
        f"{metrics['rms_difference_percent_points']:.3f} percentage points"
    )
    lines.append("")

    lines.append("Field widths")
    for level in [80, 50, 20]:
        lines.append(
            f"  {level}% width: film {format_optional(metrics['film'][f'width_{level}_mm'])} mm, "
            f"DICOM {format_optional(metrics['dicom'][f'width_{level}_mm'])} mm, "
            f"difference {format_optional(metrics[f'width_{level}_difference_mm'])} mm"
        )
    lines.append("")

    lines.append("Penumbra 80%-20%")
    for side in ["left", "right"]:
        lines.append(
            f"  {side.capitalize()}: film {format_optional(metrics['film'][f'{side}_penumbra_80_20_mm'])} mm, "
            f"DICOM {format_optional(metrics['dicom'][f'{side}_penumbra_80_20_mm'])} mm, "
            f"difference {format_optional(metrics[f'{side}_penumbra_difference_mm'])} mm"
        )

    return "\n".join(lines)


def save_metrics_text_file(film, metrics_text):
    """Save the terminal metrics text with the same stem as the film CSV."""
    if SAVE_RESULTS:
        RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
        text_path = RESULTS_FOLDER / f"{film.path.stem}.txt"
        text_path.write_text(metrics_text + "\n")
        print()
        print(f"Saved metrics text: {text_path}")


def plot_profiles(film, dicom, metrics):
    """Plot the film and DICOM profiles together."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    plot_min_x = metrics["comparison_min_x_mm"]
    plot_max_x = metrics["comparison_max_x_mm"]
    film_plot_mask = (
        (film.x_mm >= plot_min_x)
        & (film.x_mm <= plot_max_x)
    )
    dicom_plot_mask = (
        (dicom.x_mm >= plot_min_x)
        & (dicom.x_mm <= plot_max_x)
    )

    axes[0].plot(
        film.x_mm[film_plot_mask],
        film.relative_percent[film_plot_mask],
        label="Film profile",
        linewidth=2,
    )
    axes[0].plot(
        dicom.x_mm[dicom_plot_mask],
        dicom.relative_percent[dicom_plot_mask],
        label="DICOM profile",
        linewidth=2,
    )
    axes[0].set_ylabel("Relative dose [%]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    comparison_x, film_percent, dicom_percent = interpolate_to_film_positions(
        film,
        dicom,
        plot_min_x,
        plot_max_x,
    )
    axes[1].plot(
        comparison_x,
        film_percent - dicom_percent,
        color="tab:red",
        linewidth=1.6,
    )
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[1].set_xlabel("Profile position [mm]")
    axes[1].set_ylabel("Film - DICOM [% pts]")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"6 MeV, 10x10, 105 SSD, Lead Collimator, 3mm depth")
    fig.tight_layout()

    if SAVE_RESULTS:
        RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
        plot_path = RESULTS_FOLDER / f"{film.path.stem}.png"
        fig.savefig(plot_path, dpi=200)
        print()
        print(f"Saved plot: {plot_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def parse_args():
    """Read optional command-line paths."""
    parser = argparse.ArgumentParser(
        description="Compare one film profile CSV against one DICOM profile CSV."
    )
    parser.add_argument(
        "film_csv",
        nargs="?",
        type=Path,
        help="Optional film profile CSV path. If omitted, FILM_PROFILE_CSV_PATH is used.",
    )
    parser.add_argument(
        "dicom_csv",
        nargs="?",
        type=Path,
        help="Optional DICOM profile CSV path. If omitted, DICOM_PROFILE_CSV_PATH is used.",
    )
    return parser.parse_args()


def main():
    """Run the comparison."""
    args = parse_args()

    film_csv_path = args.film_csv if args.film_csv is not None else FILM_PROFILE_CSV_PATH
    dicom_csv_path = args.dicom_csv if args.dicom_csv is not None else DICOM_PROFILE_CSV_PATH

    film = read_film_profile_csv(film_csv_path)
    dicom = read_dicom_profile_csv(dicom_csv_path)

    film, dicom, alignment_info = apply_orientation_and_alignment(film, dicom)
    metrics = calculate_metrics(film, dicom, alignment_info)
    metrics_text = build_metrics_text(film, dicom, metrics)

    print(metrics_text)
    save_metrics_text_file(film, metrics_text)
    plot_profiles(film, dicom, metrics)


if __name__ == "__main__":
    main()
