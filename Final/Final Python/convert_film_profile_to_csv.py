#!/usr/bin/env python3
"""
Convert scanned film profile TIFF files into calibrated profile CSV files.

This script is intentionally written as a readable, standalone workflow:

1. Read calibration film TIFFs with known doses in their filenames.
2. Build a simple calibration curve from mean signal in one calibration ROI.
3. Read each profile film TIFF.
4. Average signal across one profile ROI.
5. Convert that signal to dose, normalize to percent, and write a CSV.

The user-editable settings are grouped at the top of the file.
"""

from pathlib import Path
import csv
import math
import re

import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

# Folder containing the scanned profile film TIFF files.
PROFILE_TIFF_FOLDER = Path('/Users/parkernew/Code/work/RS Project/Final/Profile/12MeV/Scans')

# Folder containing the scanned calibration film TIFF files.
CALIBRATION_TIFF_FOLDER = Path(
    '/Users/parkernew/Code/work/RS Project/Final/Calibration Films/6MeV_calibration_films'
)

# Folder where the profile CSV files should be written.
CSV_OUTPUT_FOLDER = Path('/Users/parkernew/Code/work/RS Project/Final/Profile/12MeV/Scans-CSV')

# Film channel to use. EBT film is commonly most sensitive in the red channel.
# Choices are "red", "green", or "blue".
COLOR_CHANNEL = "red"

# Calibration-film ROI in pixel coordinates.
# The script uses this same rectangular patch on every calibration film.
# Format: x_min, x_max, y_min, y_max. The max values are not included.
CALIBRATION_ROI = (70, 195, 85, 185)

# Profile-film ROI in pixel coordinates.
# Format: x_min, x_max, y_min, y_max. The max values are not included.
# To use the bottom edge of each TIFF image as y_max, write "none".
PROFILE_ROI = (100, 180, 60, 1420)

# Direction of the profile inside PROFILE_ROI.
# "y" means average across x and keep the top-to-bottom direction as the profile.
# "x" means average across y and keep the left-to-right direction as the profile.
PROFILE_AXIS = "y"

# Scanner resolution used for these films: 0.127 mm per pixel is 200 dpi.
MILLIMETERS_PER_PIXEL = 0.127

# Distance assigned to the first output point in the profile.
FIRST_PROFILE_POINT_MM = 0.0

# Optional smoothing for the normalized percent column.
# 1 means no smoothing. If you use smoothing, choose an odd number like 3 or 5.
SMOOTHING_WINDOW_PIXELS = 1

# How to choose 100% for the profile.
# "midpoint_20_20" finds the approximate 20% field edges, uses the midpoint
# between them, and normalizes to the dose at that midpoint.
# "max" normalizes to the highest dose in the profile.
NORMALIZATION_METHOD = "midpoint_20_20"


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

def find_tiff_files(folder):
    """Return all TIFF files in a folder."""
    tiff_files = []
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        tiff_files.extend(folder.glob(pattern))
    return sorted(tiff_files)


def channel_number(channel_name):
    """Convert a channel name to the matching image-array channel number."""
    channel_name = channel_name.lower()
    if channel_name == "red":
        return 0
    if channel_name == "green":
        return 1
    if channel_name == "blue":
        return 2
    raise ValueError('COLOR_CHANNEL must be "red", "green", or "blue".')


def read_tiff_channel(tiff_path, channel_name):
    """Read one TIFF file and return one color channel as floating-point values."""
    image = tifffile.imread(tiff_path)
    image = np.asarray(image)

    if image.ndim == 2:
        channel = image
    elif image.ndim == 3:
        channel = image[:, :, channel_number(channel_name)]
    else:
        raise ValueError(f"{tiff_path.name} has an unsupported image shape: {image.shape}")

    # Scaling 8-bit images to a 16-bit range keeps scanner signal values
    # comparable if one scan is saved differently.
    channel = channel.astype(float)
    if np.issubdtype(image.dtype, np.integer) and image.dtype.itemsize == 1:
        channel = channel * (65535.0 / 255.0)

    return channel


def check_roi_fits_image(image, roi, roi_name):
    """Stop with a clear message if an ROI is outside the image."""
    x_min, x_max, y_min, y_max = roi
    image_height, image_width = image.shape

    roi_is_valid = (
        0 <= x_min < x_max <= image_width
        and 0 <= y_min < y_max <= image_height
    )

    if not roi_is_valid:
        raise ValueError(
            f"{roi_name} {roi} does not fit inside image "
            f"with width={image_width} and height={image_height}."
        )


def resolve_roi_for_image(image, roi, roi_name):
    """Replace a y_max value of "none" with this image's height."""
    x_min, x_max, y_min, y_max = roi
    image_height = image.shape[0]

    if isinstance(y_max, str):
        if y_max.lower() == "none":
            y_max = image_height
        else:
            raise ValueError(
                f'{roi_name} y_max must be a number or "none", not {y_max!r}.'
            )

    return x_min, x_max, y_min, y_max


def crop_image(image, roi, roi_name):
    """Return the part of an image inside an ROI."""
    resolved_roi = resolve_roi_for_image(image, roi, roi_name)
    check_roi_fits_image(image, resolved_roi, roi_name)
    x_min, x_max, y_min, y_max = resolved_roi
    return image[y_min:y_max, x_min:x_max]


def profile_from_roi(roi_image):
    """Average a 2D ROI into one 1D profile."""
    axis = PROFILE_AXIS.lower()
    if axis == "x":
        return np.nanmean(roi_image, axis=0)
    if axis == "y":
        return np.nanmean(roi_image, axis=1)
    raise ValueError('PROFILE_AXIS must be "x" or "y".')


def net_optical_density(unexposed_signal, exposed_signal):
    """Calculate net optical density from unexposed and exposed film signals."""
    with np.errstate(divide="ignore", invalid="ignore"):
        net_od = np.log10(unexposed_signal / exposed_signal)
    return np.where(np.isfinite(net_od), net_od, np.nan)


def calibration_dose_from_filename(tiff_path):
    """Read calibration dose from names like 6MeV_100cGy.tif."""
    match = re.search(
        r"(?<!\d)(0|50|100|150|200|250|300)\s*c\s*gy(?!\d)",
        tiff_path.stem,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1))

    raise ValueError(
        f"Could not read a calibration dose from {tiff_path.name}. "
        "Use names like 6MeV_100cGy.tif."
    )


def moving_average(values, window):
    """Smooth a profile without changing its length."""
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("SMOOTHING_WINDOW_PIXELS must be odd, like 3 or 5.")
    if window > len(values):
        raise ValueError("SMOOTHING_WINDOW_PIXELS cannot exceed the profile length.")

    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_calibration_curve():
    """Build a dose-vs-netOD curve from the calibration film scans."""
    calibration_files = find_tiff_files(CALIBRATION_TIFF_FOLDER)
    if not calibration_files:
        raise FileNotFoundError(f"No calibration TIFF files found in {CALIBRATION_TIFF_FOLDER}")

    files_by_dose = {}
    for tiff_path in calibration_files:
        dose_cgy = calibration_dose_from_filename(tiff_path)
        files_by_dose.setdefault(dose_cgy, []).append(tiff_path)

    if 0.0 not in files_by_dose:
        raise FileNotFoundError("A 0 cGy calibration film is required.")

    zero_dose_file = sorted(files_by_dose[0.0])[0]
    zero_dose_image = read_tiff_channel(zero_dose_file, COLOR_CHANNEL)
    zero_dose_roi = crop_image(zero_dose_image, CALIBRATION_ROI, "CALIBRATION_ROI")
    unexposed_mean_signal = float(np.nanmean(zero_dose_roi))

    if not math.isfinite(unexposed_mean_signal) or unexposed_mean_signal <= 0:
        raise ValueError("The 0 cGy calibration ROI has an invalid mean signal.")

    calibration_rows = []
    for dose_cgy in sorted(files_by_dose):
        net_od_values_for_this_dose = []
        mean_signals_for_this_dose = []

        for tiff_path in sorted(files_by_dose[dose_cgy]):
            image = read_tiff_channel(tiff_path, COLOR_CHANNEL)
            roi = crop_image(image, CALIBRATION_ROI, "CALIBRATION_ROI")
            mean_signal = float(np.nanmean(roi))
            net_od = float(net_optical_density(unexposed_mean_signal, mean_signal))

            net_od_values_for_this_dose.append(net_od)
            mean_signals_for_this_dose.append(mean_signal)

        calibration_rows.append(
            {
                "dose_cgy": dose_cgy,
                "net_od": float(np.nanmean(net_od_values_for_this_dose)),
                "mean_signal": float(np.nanmean(mean_signals_for_this_dose)),
                "source_files": "; ".join(path.name for path in sorted(files_by_dose[dose_cgy])),
            }
        )

    doses_cgy = np.array([row["dose_cgy"] for row in calibration_rows], dtype=float)
    net_od = np.array([row["net_od"] for row in calibration_rows], dtype=float)

    if np.any(np.diff(net_od) <= 0):
        raise ValueError(
            "Calibration netOD does not increase with dose. "
            "Check the calibration ROI and calibration file dose labels."
        )

    return calibration_rows, doses_cgy, net_od, unexposed_mean_signal


def dose_from_net_od(net_od_values, calibration_net_od, calibration_doses_cgy):
    """Convert netOD values to dose by straight-line interpolation."""
    clipped_net_od = np.clip(net_od_values, calibration_net_od[0], calibration_net_od[-1])
    return np.interp(clipped_net_od, calibration_net_od, calibration_doses_cgy)


def write_calibration_csv(calibration_rows):
    """Write the calibration curve so the dose conversion is easy to inspect."""
    csv_path = CALIBRATION_TIFF_FOLDER / "film_calibration_curve.csv"

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["dose_cgy", "net_od", "mean_signal", "source_files"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(calibration_rows)

    return csv_path


# ---------------------------------------------------------------------------
# Profile conversion
# ---------------------------------------------------------------------------

def find_crossing_one_side(percent_profile, level, side, center_index):
    """Find a percent-level crossing on the left or right side of the profile."""
    if side == "left":
        search_indices = range(center_index, 0, -1)
        for index in search_indices:
            y1 = percent_profile[index]
            y0 = percent_profile[index - 1]
            if (y1 - level) * (y0 - level) <= 0 and y1 != y0:
                fraction = (level - y0) / (y1 - y0)
                return (index - 1) + fraction

    elif side == "right":
        search_indices = range(center_index, len(percent_profile) - 1)
        for index in search_indices:
            y0 = percent_profile[index]
            y1 = percent_profile[index + 1]
            if (y0 - level) * (y1 - level) <= 0 and y1 != y0:
                fraction = (level - y0) / (y1 - y0)
                return index + fraction

    else:
        raise ValueError('side must be "left" or "right".')

    return None


def choose_normalization_dose(dose_cgy):
    """Choose the dose value that should become 100%."""
    method = NORMALIZATION_METHOD.lower()

    if method == "max":
        return float(np.nanmax(dose_cgy))

    if method == "midpoint_20_20":
        temporary_percent = dose_cgy / np.nanmax(dose_cgy) * 100.0
        center_index = len(dose_cgy) // 2

        left_20 = find_crossing_one_side(
            temporary_percent,
            level=20.0,
            side="left",
            center_index=center_index,
        )
        right_20 = find_crossing_one_side(
            temporary_percent,
            level=20.0,
            side="right",
            center_index=center_index,
        )

        if left_20 is None or right_20 is None:
            return float(dose_cgy[center_index])

        midpoint_index = int(round((left_20 + right_20) / 2.0))
        midpoint_index = int(np.clip(midpoint_index, 0, len(dose_cgy) - 1))
        return float(dose_cgy[midpoint_index])

    raise ValueError('NORMALIZATION_METHOD must be "midpoint_20_20" or "max".')


def convert_one_profile_film(
    tiff_path,
    calibration_net_od,
    calibration_doses_cgy,
    unexposed_mean_signal,
):
    """Convert one profile TIFF image into distance, dose, and percent arrays."""
    image = read_tiff_channel(tiff_path, COLOR_CHANNEL)
    profile_roi = crop_image(image, PROFILE_ROI, "PROFILE_ROI")
    signal_profile = profile_from_roi(profile_roi)

    net_od_profile = net_optical_density(unexposed_mean_signal, signal_profile)
    dose_cgy = dose_from_net_od(net_od_profile, calibration_net_od, calibration_doses_cgy)
    dose_gy = dose_cgy / 100.0

    normalization_dose_cgy = choose_normalization_dose(dose_cgy)
    if not math.isfinite(normalization_dose_cgy) or normalization_dose_cgy <= 0:
        raise ValueError(f"{tiff_path.name} produced an invalid profile dose.")

    normalized_percent = dose_cgy / normalization_dose_cgy * 100.0
    smoothed_percent = moving_average(normalized_percent, SMOOTHING_WINDOW_PIXELS)

    x_mm = (
        np.arange(len(signal_profile), dtype=float) * MILLIMETERS_PER_PIXEL
        + FIRST_PROFILE_POINT_MM
    )

    return x_mm, signal_profile, net_od_profile, dose_cgy, dose_gy, normalized_percent, smoothed_percent


def write_profile_csv(
    tiff_path,
    x_mm,
    signal_profile,
    net_od_profile,
    dose_cgy,
    dose_gy,
    normalized_percent,
    smoothed_percent,
):
    """Write one profile curve to a CSV file."""
    csv_path = CSV_OUTPUT_FOLDER / f"profile_{tiff_path.stem}.csv"

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow([
            "x_mm",
            "intensity",
            "net_od",
            "dose_cgy",
            "dose_gy",
            "normalized_percent",
            "smoothed_percent",
        ])

        rows = zip(
            x_mm,
            signal_profile,
            net_od_profile,
            dose_cgy,
            dose_gy,
            normalized_percent,
            smoothed_percent,
        )
        for x, signal, net_od, dose_cgy_value, dose_gy_value, percent, smoothed in rows:
            writer.writerow([
                f"{x:.10g}",
                f"{signal:.10g}",
                f"{net_od:.10g}",
                f"{dose_cgy_value:.10g}",
                f"{dose_gy_value:.10g}",
                f"{percent:.10g}",
                f"{smoothed:.10g}",
            ])

    return csv_path


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main():
    """Convert every profile TIFF file in PROFILE_TIFF_FOLDER."""
    profile_files = find_tiff_files(PROFILE_TIFF_FOLDER)

    if not profile_files:
        print(f"No profile TIFF files were found in: {PROFILE_TIFF_FOLDER}")
        return

    CSV_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    calibration_rows, calibration_doses_cgy, calibration_net_od, unexposed_mean_signal = (
        build_calibration_curve()
    )
    calibration_csv = write_calibration_csv(calibration_rows)

    print("Film profile TIFF to CSV conversion")
    print(f"Profile TIFF folder:     {PROFILE_TIFF_FOLDER}")
    print(f"Calibration TIFF folder: {CALIBRATION_TIFF_FOLDER}")
    print(f"CSV output folder:       {CSV_OUTPUT_FOLDER}")
    print(f"Color channel:           {COLOR_CHANNEL}")
    print(f"Calibration ROI:         {CALIBRATION_ROI}")
    print(f"Profile ROI:             {PROFILE_ROI}")
    print(f"Profile axis:            {PROFILE_AXIS}")
    print(f"Pixel size:              {MILLIMETERS_PER_PIXEL:g} mm")
    print(f"Normalization method:    {NORMALIZATION_METHOD}")
    print(f"Calibration CSV:         {calibration_csv}")
    print()

    for tiff_path in profile_files:
        x_mm, signal, net_od, dose_cgy, dose_gy, normalized_percent, smoothed_percent = (
            convert_one_profile_film(
                tiff_path,
                calibration_net_od,
                calibration_doses_cgy,
                unexposed_mean_signal,
            )
        )

        csv_path = write_profile_csv(
            tiff_path,
            x_mm,
            signal,
            net_od,
            dose_cgy,
            dose_gy,
            normalized_percent,
            smoothed_percent,
        )

        maximum_dose_index = int(np.nanargmax(dose_cgy))
        print(
            f"{tiff_path.name} -> {csv_path.name} "
            f"(maximum dose {dose_cgy[maximum_dose_index]:.3f} cGy "
            f"at {x_mm[maximum_dose_index]:.2f} mm)"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
