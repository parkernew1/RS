#!/usr/bin/env python3
"""
Convert scanned film PDD TIFF files into calibrated PDD CSV files.

This script is intentionally written as a readable, standalone workflow:

1. Read calibration film TIFFs with known doses in their filenames.
2. Build a simple calibration curve from mean signal in one calibration ROI.
3. Read each PDD film TIFF.
4. Average signal across one PDD strip ROI.
5. Convert that signal to dose, normalize to PDD percent, and write a CSV.

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

# Folder containing the scanned PDD film TIFF files.
PDD_TIFF_FOLDER = Path('/Users/parkernew/Code/work/RS Project/Final/PDD/12MeV/Scans')

# Folder containing the scanned calibration film TIFF files.
CALIBRATION_TIFF_FOLDER = Path(
    '/Users/parkernew/Code/work/RS Project/Final/Calibration Films/12MeV_calibration_films'
)

# Folder where the CSV files should be written.
CSV_OUTPUT_FOLDER = Path('/Users/parkernew/Code/work/RS Project/Final/PDD/12MeV/Scans-CSV')

# Film channel to use. EBT film is commonly most sensitive in the red channel.
# Choices are "red", "green", or "blue".
COLOR_CHANNEL = "red"

# Calibration-film ROI in pixel coordinates.
# The script uses this same rectangular patch on every calibration film.
# Format: x_min, x_max, y_min, y_max. The max values are not included.
CALIBRATION_ROI = (75, 200, 150, 250)

# PDD-film ROI in pixel coordinates.
# The script averages across x and keeps the y direction as the depth curve.
# Format: x_min, x_max, y_min, y_max. The max values are not included.
# To use the bottom edge of each TIFF image as y_max, write "none".
PDD_ROI = (100, 200, 40, "none")

# The current PDD scans have the entrance/surface end near the bottom of the ROI.
# "reverse" makes the output start at that bottom end and move upward through
# the selected strip. Use "forward" if a future scan has surface at the top.
DEPTH_DIRECTION = "reverse"

# Scanner resolution used for these films: 0.127 mm per pixel is 200 dpi.
MILLIMETERS_PER_PIXEL = 0.127

# Depth assigned to the first output point.
# Choice made here: keep the previous project value of 3.66 mm because the
# current PDD ROI avoids the lower film edge/artifact rather than starting
# exactly at the physical surface. If you crop the ROI exactly to the surface,
# set this to 0.0.
DEPTH_OF_FIRST_OUTPUT_POINT_MM = 0.0
PDD_NORMALIZATION_MINIMUM_DEPTH_MM = 5.0


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

    # One current PDD scan is 8-bit while the others are 16-bit. Scaling 8-bit
    # images to a 16-bit range keeps scanner signal values comparable.
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


def net_optical_density(unexposed_signal, exposed_signal):
    """Calculate net optical density from unexposed and exposed film signals."""
    with np.errstate(divide="ignore", invalid="ignore"):
        net_od = np.log10(unexposed_signal / exposed_signal)
    return np.where(np.isfinite(net_od), net_od, np.nan)


def calibration_dose_from_filename(tiff_path):
    """Read calibration dose from names like 6MeV_100cGy.tif."""
    match = re.search(r"(?<!\d)(0|50|100|150|200|250|300)\s*c\s*gy(?!\d)", tiff_path.stem, re.IGNORECASE)
    if match:
        return float(match.group(1))

    raise ValueError(
        f"Could not read a calibration dose from {tiff_path.name}. "
        "Use names like 6MeV_100cGy.tif."
    )


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
# PDD film conversion
# ---------------------------------------------------------------------------

def pdd_signal_profile_from_image(image):
    """Average the PDD ROI across its width to get one signal value per depth pixel."""
    pdd_roi_image = crop_image(image, PDD_ROI, "PDD_ROI")

    # PDD_ROI is y by x. Averaging over x leaves one value for each y pixel.
    signal_profile = np.nanmean(pdd_roi_image, axis=1)

    if DEPTH_DIRECTION.lower() == "reverse":
        return signal_profile[::-1]
    if DEPTH_DIRECTION.lower() == "forward":
        return signal_profile

    raise ValueError('DEPTH_DIRECTION must be "reverse" or "forward".')


def convert_one_pdd_film(tiff_path, calibration_net_od, calibration_doses_cgy, unexposed_mean_signal):
    """Convert one PDD TIFF image into depth, dose, and PDD arrays."""
    image = read_tiff_channel(tiff_path, COLOR_CHANNEL)
    signal_profile = pdd_signal_profile_from_image(image)

    net_od_profile = net_optical_density(unexposed_mean_signal, signal_profile)
    dose_cgy = dose_from_net_od(net_od_profile, calibration_net_od, calibration_doses_cgy)
    dose_gy = dose_cgy / 100.0

    depth_mm = (
        np.arange(len(dose_cgy), dtype=float) * MILLIMETERS_PER_PIXEL
        + DEPTH_OF_FIRST_OUTPUT_POINT_MM
    )

    normalization_mask = depth_mm >= PDD_NORMALIZATION_MINIMUM_DEPTH_MM
    dose_used_for_normalization = dose_cgy[normalization_mask]

    maximum_dose_cgy = np.nanmax(dose_used_for_normalization)
    if not math.isfinite(maximum_dose_cgy) or maximum_dose_cgy <= 0:
        raise ValueError(f"{tiff_path.name} produced an invalid dose profile.")

    pdd_percent = dose_cgy / maximum_dose_cgy * 100.0

    #maximum_dose_cgy = np.nanmax(dose_cgy)
    #if not math.isfinite(maximum_dose_cgy) or maximum_dose_cgy <= 0:
        #raise ValueError(f"{tiff_path.name} produced an invalid dose profile.")

    #pdd_percent = dose_cgy / maximum_dose_cgy * 100.0
    #depth_mm = (
        #np.arange(len(pdd_percent), dtype=float) * MILLIMETERS_PER_PIXEL
        #+ DEPTH_OF_FIRST_OUTPUT_POINT_MM
    #)

    return depth_mm, signal_profile, net_od_profile, dose_cgy, dose_gy, pdd_percent


def write_pdd_csv(tiff_path, depth_mm, signal_profile, net_od_profile, dose_cgy, dose_gy, pdd_percent):
    """Write one converted PDD film to CSV."""
    csv_path = CSV_OUTPUT_FOLDER / f"{tiff_path.stem}.csv"

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["Depth [mm]", "Signal", "Net OD", "Dose [cGy]", "Dose [Gy]", "PDD [%]"])

        for row in zip(depth_mm, signal_profile, net_od_profile, dose_cgy, dose_gy, pdd_percent):
            writer.writerow([f"{value:.10g}" for value in row])

    return csv_path


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def main():
    """Run the calibration and convert every PDD TIFF file."""
    CSV_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    pdd_files = find_tiff_files(PDD_TIFF_FOLDER)
    if not pdd_files:
        raise FileNotFoundError(f"No PDD TIFF files found in {PDD_TIFF_FOLDER}")

    calibration_rows, calibration_doses_cgy, calibration_net_od, unexposed_mean_signal = build_calibration_curve()
    calibration_csv = write_calibration_csv(calibration_rows)

    print("Film PDD TIFF to CSV conversion")
    print(f"PDD TIFF folder:         {PDD_TIFF_FOLDER}")
    print(f"Calibration TIFF folder: {CALIBRATION_TIFF_FOLDER}")
    print(f"CSV output folder:       {CSV_OUTPUT_FOLDER}")
    print(f"Color channel:           {COLOR_CHANNEL}")
    print(f"Calibration ROI:         {CALIBRATION_ROI}")
    print(f"PDD ROI:                 {PDD_ROI}")
    print(f"Depth direction:         {DEPTH_DIRECTION}")
    print(f"First output depth:      {DEPTH_OF_FIRST_OUTPUT_POINT_MM:g} mm")
    print(f"Calibration CSV:         {calibration_csv.name}")
    print()

    for tiff_path in pdd_files:
        depth_mm, signal, net_od, dose_cgy, dose_gy, pdd_percent = convert_one_pdd_film(
            tiff_path,
            calibration_net_od,
            calibration_doses_cgy,
            unexposed_mean_signal,
        )

        csv_path = write_pdd_csv(tiff_path, depth_mm, signal, net_od, dose_cgy, dose_gy, pdd_percent)

        dmax_index = int(np.nanargmax(dose_cgy))
        print(
            f"{tiff_path.name} -> {csv_path.name} "
            f"(maximum dose {dose_cgy[dmax_index]:.3f} cGy at {depth_mm[dmax_index]:.2f} mm)"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
