#!/usr/bin/env python3
"""
Compare one film PDD CSV against one DICOM PDD CSV.

This script reads two CSV files, prints agreement metrics, and plots both PDD
curves together. By default it does not save any outputs; it only prints to the
terminal and opens a matplotlib plot window.

The easiest way to use it is to edit the user settings below, then run:

    python compare_two_pdd_csvs.py

You can also pass paths from the command line:

    python compare_two_pdd_csvs.py path/to/film.csv path/to/dicom.csv
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

# Enter the film PDD CSV path here.
FILM_PDD_CSV_PATH = Path(
    "/Users/parkernew/Code/work/RS Project/Final/PDD/12MeV/Scans-CSV/PDD_12MeV_6_105_2_Lead.csv"
)

# Enter the DICOM/RayStation PDD CSV path here.
DICOM_PDD_CSV_PATH = Path(
    "/Users/parkernew/Code/work/RS Project/Final/PDD/12MeV/dcm-CSV/PDD_12MeV_6_105_2_Lead.csv"
)

# Enter a folder here if you later decide to save plots or metric files.
# SAVE_RESULTS is False by default, so this script will not save anything unless
# you intentionally change SAVE_RESULTS to True.
RESULTS_FOLDER = Path(
    "/Users/parkernew/Code/work/RS Project/Final/PDD/12MeV/Results"
)
SAVE_RESULTS = False

# Ignore all points shallower than this depth.
# This affects the plotted curves, dmax search, distal falloff metrics, and
# point-by-point PDD agreement metrics.
# Example: 2.0 means "start comparing at 2 mm depth."
IGNORE_DEPTH_BEFORE_MM = 2.0

# Stop comparing at this depth.
# Leave this as None to compare until the deepest overlapping depth.
COMPARISON_MAX_DEPTH_MM = None


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

class PddCurve:
    """One PDD curve loaded from a CSV file."""

    def __init__(self, label, path, depth_mm, dose_gy, pdd_percent):
        self.label = label
        self.path = path
        self.depth_mm = depth_mm
        self.dose_gy = dose_gy
        self.pdd_percent = pdd_percent


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
        .replace(" ", "")
    )


def find_column(fieldnames, possible_names):
    """Find one CSV column from a list of acceptable names."""
    normalized_to_original = {
        normalized_column_name(name): name
        for name in fieldnames
    }

    for possible_name in possible_names:
        normalized = normalized_column_name(possible_name)
        if normalized in normalized_to_original:
            return normalized_to_original[normalized]

    raise ValueError(
        "Could not find one of these columns in the CSV: "
        + ", ".join(possible_names)
    )


def read_pdd_csv(path, label):
    """Read depth, dose, and PDD columns from one CSV file."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV does not exist: {path}")

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not have a CSV header row.")

        depth_column = find_column(reader.fieldnames, ["Depth [mm]", "Depth mm", "Depth"])
        dose_column = find_column(reader.fieldnames, ["Dose [Gy]", "Dose Gy", "Dose"])
        pdd_column = find_column(reader.fieldnames, ["PDD [%]", "PDD percent", "PDD"])

        depth_values = []
        dose_values = []
        pdd_values = []

        for row_number, row in enumerate(reader, start=2):
            try:
                depth_values.append(float(row[depth_column]))
                dose_values.append(float(row[dose_column]))
                pdd_values.append(float(row[pdd_column]))
            except ValueError as error:
                raise ValueError(
                    f"Could not read numeric values from {path} on row {row_number}."
                ) from error

    depth_mm = np.asarray(depth_values, dtype=float)
    dose_gy = np.asarray(dose_values, dtype=float)
    pdd_percent = np.asarray(pdd_values, dtype=float)

    valid_points = (
        np.isfinite(depth_mm)
        & np.isfinite(dose_gy)
        & np.isfinite(pdd_percent)
    )
    depth_mm = depth_mm[valid_points]
    dose_gy = dose_gy[valid_points]
    pdd_percent = pdd_percent[valid_points]

    if len(depth_mm) == 0:
        raise ValueError(f"{path} does not contain any valid PDD points.")

    sorted_indices = np.argsort(depth_mm)
    return PddCurve(
        label=label,
        path=path,
        depth_mm=depth_mm[sorted_indices],
        dose_gy=dose_gy[sorted_indices],
        pdd_percent=pdd_percent[sorted_indices],
    )


# ---------------------------------------------------------------------------
# Agreement calculations
# ---------------------------------------------------------------------------

def find_dmax(curve):
    """Find the maximum dose point after the ignored shallow region."""
    search_mask = curve.depth_mm >= IGNORE_DEPTH_BEFORE_MM
    if not np.any(search_mask):
        raise ValueError(
            f"{curve.label} has no points at or deeper than "
            f"{IGNORE_DEPTH_BEFORE_MM:g} mm."
        )

    search_dose = curve.dose_gy[search_mask]
    search_depth = curve.depth_mm[search_mask]
    search_pdd = curve.pdd_percent[search_mask]

    max_index = int(np.nanargmax(search_dose))
    return {
        "depth_mm": float(search_depth[max_index]),
        "dose_gy": float(search_dose[max_index]),
        "pdd_percent": float(search_pdd[max_index]),
    }


def automatic_comparison_range(film, dicom):
    """Choose the depth range where both curves have data."""
    minimum_depth = max(float(np.min(film.depth_mm)), float(np.min(dicom.depth_mm)))
    maximum_depth = min(float(np.max(film.depth_mm)), float(np.max(dicom.depth_mm)))

    minimum_depth = max(minimum_depth, float(IGNORE_DEPTH_BEFORE_MM))
    if COMPARISON_MAX_DEPTH_MM is not None:
        maximum_depth = min(maximum_depth, float(COMPARISON_MAX_DEPTH_MM))

    if minimum_depth >= maximum_depth:
        raise ValueError("The two PDD curves do not have an overlapping depth range.")

    return minimum_depth, maximum_depth


def interpolate_to_film_depths(film, dicom, minimum_depth, maximum_depth):
    """Interpolate the DICOM PDD onto film depths for point-by-point comparison."""
    comparison_mask = (
        (film.depth_mm >= minimum_depth)
        & (film.depth_mm <= maximum_depth)
    )
    comparison_depth = film.depth_mm[comparison_mask]
    film_pdd = film.pdd_percent[comparison_mask]
    dicom_pdd = np.interp(comparison_depth, dicom.depth_mm, dicom.pdd_percent)

    return comparison_depth, film_pdd, dicom_pdd


def depth_at_pdd_percent(curve, percent_level):
    """Find the distal depth where the PDD curve falls through a percent level."""
    dmax = find_dmax(curve)
    after_dmax = (
        (curve.depth_mm >= dmax["depth_mm"])
        & (curve.depth_mm >= IGNORE_DEPTH_BEFORE_MM)
    )

    depth = curve.depth_mm[after_dmax]
    pdd = curve.pdd_percent[after_dmax]

    for index in range(len(depth) - 1):
        pdd_a = pdd[index]
        pdd_b = pdd[index + 1]

        if pdd_a >= percent_level >= pdd_b:
            if pdd_a == pdd_b:
                return float(depth[index])

            fraction = (percent_level - pdd_a) / (pdd_b - pdd_a)
            return float(depth[index] + fraction * (depth[index + 1] - depth[index]))

    return None


def calculate_metrics(film, dicom):
    """Calculate simple agreement metrics between film and DICOM PDDs."""
    film_dmax = find_dmax(film)
    dicom_dmax = find_dmax(dicom)
    minimum_depth, maximum_depth = automatic_comparison_range(film, dicom)
    comparison_depth, film_pdd, dicom_pdd = interpolate_to_film_depths(
        film,
        dicom,
        minimum_depth,
        maximum_depth,
    )

    pdd_difference = film_pdd - dicom_pdd
    absolute_difference = np.abs(pdd_difference)

    metrics = {
        "film_dmax": film_dmax,
        "dicom_dmax": dicom_dmax,
        "dmax_depth_difference_mm": film_dmax["depth_mm"] - dicom_dmax["depth_mm"],
        "dmax_dose_difference_gy": film_dmax["dose_gy"] - dicom_dmax["dose_gy"],
        "comparison_min_depth_mm": minimum_depth,
        "comparison_max_depth_mm": maximum_depth,
        "number_of_comparison_points": len(comparison_depth),
        "mean_pdd_difference_percent_points": float(np.mean(pdd_difference)),
        "mean_absolute_pdd_difference_percent_points": float(np.mean(absolute_difference)),
        "max_absolute_pdd_difference_percent_points": float(np.max(absolute_difference)),
        "rms_pdd_difference_percent_points": float(np.sqrt(np.mean(pdd_difference ** 2))),
    }

    for percent_level in [90, 80, 50]:
        film_depth = depth_at_pdd_percent(film, percent_level)
        dicom_depth = depth_at_pdd_percent(dicom, percent_level)
        metrics[f"film_r{percent_level}_depth_mm"] = film_depth
        metrics[f"dicom_r{percent_level}_depth_mm"] = dicom_depth
        if film_depth is None or dicom_depth is None:
            metrics[f"r{percent_level}_depth_difference_mm"] = None
        else:
            metrics[f"r{percent_level}_depth_difference_mm"] = film_depth - dicom_depth

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
    film_dmax = metrics["film_dmax"]
    dicom_dmax = metrics["dicom_dmax"]
    lines = []

    lines.append(f"Film CSV:  {film.path}")
    lines.append(f"DICOM CSV: {dicom.path}")
    lines.append("")
    lines.append("PDD agreement metrics")
    lines.append("---------------------")
    lines.append(f"Ignored shallow region: depths below {IGNORE_DEPTH_BEFORE_MM:g} mm")
    lines.append("")
    lines.append("Dmax")
    lines.append(f"  Film depth:        {film_dmax['depth_mm']:.3f} mm")
    lines.append(f"  DICOM depth:       {dicom_dmax['depth_mm']:.3f} mm")
    lines.append(f"  Depth difference:  {metrics['dmax_depth_difference_mm']:.3f} mm")
    lines.append(f"  Film dose:         {film_dmax['dose_gy']:.5g} Gy")
    lines.append(f"  DICOM dose:        {dicom_dmax['dose_gy']:.5g} Gy")
    lines.append(f"  Dose difference:   {metrics['dmax_dose_difference_gy']:.5g} Gy")
    lines.append("")
    lines.append("Point-by-point PDD comparison")
    lines.append(
        "  Compared depth range: "
        f"{metrics['comparison_min_depth_mm']:.3f} to "
        f"{metrics['comparison_max_depth_mm']:.3f} mm"
    )
    lines.append(f"  Comparison points: {metrics['number_of_comparison_points']}")
    lines.append(
        "  Mean difference:   "
        f"{metrics['mean_pdd_difference_percent_points']:.3f} percentage points"
    )
    lines.append(
        "  Mean abs diff:     "
        f"{metrics['mean_absolute_pdd_difference_percent_points']:.3f} percentage points"
    )
    lines.append(
        "  Max abs diff:      "
        f"{metrics['max_absolute_pdd_difference_percent_points']:.3f} percentage points"
    )
    lines.append(
        "  RMS difference:    "
        f"{metrics['rms_pdd_difference_percent_points']:.3f} percentage points"
    )
    lines.append("")
    lines.append("Distal falloff depths")
    for percent_level in [90, 80, 50]:
        film_depth = metrics[f"film_r{percent_level}_depth_mm"]
        dicom_depth = metrics[f"dicom_r{percent_level}_depth_mm"]
        difference = metrics[f"r{percent_level}_depth_difference_mm"]
        lines.append(
            f"  R{percent_level}: film {format_optional(film_depth)} mm, "
            f"DICOM {format_optional(dicom_depth)} mm, "
            f"difference {format_optional(difference)} mm"
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


def plot_pdds(film, dicom, metrics):
    """Plot the film and DICOM PDD curves together."""
    fig, ax = plt.subplots(figsize=(9, 6))

    plot_min_depth = metrics["comparison_min_depth_mm"]
    plot_max_depth = metrics["comparison_max_depth_mm"]
    film_plot_mask = (
        (film.depth_mm >= plot_min_depth)
        & (film.depth_mm <= plot_max_depth)
    )
    dicom_plot_mask = (
        (dicom.depth_mm >= plot_min_depth)
        & (dicom.depth_mm <= plot_max_depth)
    )

    ax.plot(
        film.depth_mm[film_plot_mask],
        film.pdd_percent[film_plot_mask],
        label="Film PDD",
        linewidth=2,
    )
    ax.plot(
        dicom.depth_mm[dicom_plot_mask],
        dicom.pdd_percent[dicom_plot_mask],
        label="DICOM PDD",
        linewidth=2,
    )

    ax.axvline(
        metrics["film_dmax"]["depth_mm"],
        color="tab:blue",
        linestyle="--",
        alpha=0.5,
        label="Film dmax",
    )
    ax.axvline(
        metrics["dicom_dmax"]["depth_mm"],
        color="tab:orange",
        linestyle="--",
        alpha=0.5,
        label="DICOM dmax",
    )

    ax.set_title("12MeV, 6x6, 105 SSD, Lead collimator")
    ax.set_xlabel("Depth [mm]")
    ax.set_ylabel("PDD [%]")
    ax.grid(True, alpha=0.3)
    ax.legend()
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
        description="Compare one film PDD CSV against one DICOM PDD CSV."
    )
    parser.add_argument(
        "film_csv",
        nargs="?",
        type=Path,
        help="Optional film PDD CSV path. If omitted, FILM_PDD_CSV_PATH is used.",
    )
    parser.add_argument(
        "dicom_csv",
        nargs="?",
        type=Path,
        help="Optional DICOM PDD CSV path. If omitted, DICOM_PDD_CSV_PATH is used.",
    )
    return parser.parse_args()


def main():
    """Run the comparison."""
    args = parse_args()

    film_csv_path = args.film_csv if args.film_csv is not None else FILM_PDD_CSV_PATH
    dicom_csv_path = args.dicom_csv if args.dicom_csv is not None else DICOM_PDD_CSV_PATH

    film = read_pdd_csv(film_csv_path, label="Film")
    dicom = read_pdd_csv(dicom_csv_path, label="DICOM")

    metrics = calculate_metrics(film, dicom)
    metrics_text = build_metrics_text(film, dicom, metrics)
    print(metrics_text)
    save_metrics_text_file(film, metrics_text)
    plot_pdds(film, dicom, metrics)


if __name__ == "__main__":
    main()
