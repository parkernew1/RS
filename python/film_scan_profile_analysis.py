"""

    1. Load an unexposed/reference TIFF scan.
    2. Load each exposed film TIFF in the same folder.
    3. Extract a rectangular region of interest (ROI).
    4. Average the ROI into a 1D profile.
    5. Convert scanner intensity to optical density (OD).
    6. Normalize the profile.
    7. Find left/right profile crossing positions, usually 80% and 20%.
    8. Report penumbra and field width metrics.
    9. Save plots and CSV outputs for traceability.

"""

from __future__ import annotations

import csv
import glob
import math
import os
import re
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

from film_calibration import (
    DEFAULT_CALIBRATION_DIR,
    LoadedCalibration,
    load_calibration_curve,
    net_od_profile as calibrated_net_od_profile,
    write_calibration_outputs,
)



# USER CONFIG

REPO_ROOT = Path(__file__).resolve().parents[1]

# folder should contain one or more exposed profile film scans.
SCAN_FOLDER = REPO_ROOT / "Actual Runs" / "Profiles" / "Scans"

# where results should be saved.
OUTPUT_FOLDER: Optional[Path] = REPO_ROOT / "Actual Runs" / "Profiles" / "Results"

# Use the 6 MeV calibration film set to convert netOD profiles to dose.
USE_DOSE_CALIBRATION = True
CALIBRATION_FOLDER = DEFAULT_CALIBRATION_DIR
CALIBRATION_LABEL = "6MeV"

# searches for this text case-insensitively in TIFF filenames.
# Only used when USE_DOSE_CALIBRATION is False.
REFERENCE_NAME_CONTAINS = "unexposed"

# include these file extensions
TIFF_PATTERNS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")

# options: "red", "green", "blue"
CHANNEL = "red"

# ROI definition in pixel coordinates.
# x_min/x_max define the profile length. y_min/y_max define the strip being averaged.
# get this by xamining the the film beforehand. 
# line up every film the same way when scanning
X_MIN = 100
X_MAX = 200
Y_MIN = 50
Y_MAX = 1480

# ROI used only to sample the uniform calibration films.
CALIBRATION_X_MIN = 50
CALIBRATION_X_MAX = 200
CALIBRATION_Y_MIN = 100
CALIBRATION_Y_MAX = 250

# Direction of profile.
# "x" - average over y rows and profile left-to-right across columns.
# "y" - average over x columns and profile top-to-bottom across rows.
# The current 6 MeV profile films are scanned as long vertical strips.
PROFILE_AXIS = "y"

# Pixel size in mm/pixel
#   72 dpi  -> 25.4 / 72  = 0.3528 mm/pixel
#   150 dpi -> 25.4 / 150 = 0.1693 mm/pixel
#   200 dpi -> 25.4 / 200 = 0.1270 mm/pixel
#   300 dpi -> 25.4 / 300 = 0.0847 mm/pixel
PIXEL_SIZE_MM = 0.127

# OD mode.
# "od"    : OD = log10(I0 / I), using the unexposed scan as I0.
# "netod" : netOD = log10(I_unexp / I_exp) - log10(I_unexp / I_background)
#           Here background is approximated by low-signal tails of the same exposed profile.
#           For many simple profile/penumbra tasks, "od" is clearer and adequate.
# I have put in the option for netod for if calibration curves are used in the future
OD_MODE = "od"

# define how to use the reference scan (unexposed film)
# "scalar_roi_mean" : I0 is one number, the mean reference intensity in the same ROI.
# "profile"         : I0 is a 1D reference profile from the same ROI.
# "image"           : I0 is the same-size reference image cropped to the same ROI.
#
# scalar_roi_mean is robust and default
REFERENCE_MODE = "scalar_roi_mean"

# prevent division by zero or log of invalid values.
INTENSITY_FLOOR = 1.0

# Optional smoothing
# 1 means no smoothing [x] 
# A small odd integer is also acceptable (ex: 3) [x-1] [x] [x+1]
SMOOTHING_WINDOW_PIXELS = 1

# Normalization method.
# "midpoint_20_20" : first find approximate 20%-20% field edges, then normalize to
#                    the OD at the midpoint between those edges
# "max"            : normalize to max profile value after background subtraction.
# "percentile"     : normalize to NORMALIZATION_PERCENTILE of the profile.
NORMALIZATION_METHOD = "midpoint_20_20"
NORMALIZATION_PERCENTILE = 99.0

# for penumbra, [80, 20] is standard.
# first value is the high-dose level and the second the low-dose level.
MEASUREMENT_LEVELS = (80.0, 20.0)

# Extra field-width levels to report. 50 is often useful for FWHM-like comparisons.
FIELD_WIDTH_LEVELS = (20.0, 50.0, 80.0)

# Grouping labels extracted from filenames.
# Examples matched: "6MeV", "6 MeV", "E6", "energy_6", "12MeV".
ENERGY_REGEX = re.compile(r"(?<!\d)(6|9|12|15)(?:\s*mev|mev|\b)", re.IGNORECASE)

# If True, the script saves per-film profile CSV files.
SAVE_PROFILE_CSVS = True

# If True, save plot PNGs.
SAVE_PLOTS = True

# If True, show plots interactively at the end.
SHOW_PLOTS = False

# Plot grouping mode.
# "individual" : one plot per film.
# "energy"     : one stacked figure per energy group.
# "both"       : save both individual and grouped plots.
PLOT_MODE = "both"

# If True, save a quick QA image showing the ROI rectangle on each film.
SAVE_ROI_QA_IMAGES = True

# If True, continue processing other films when one film fails.
# If False, stop immediately on the first error.
CONTINUE_ON_ERROR = False


# DATA STRUCTURES

@dataclass
class FilmResult:
    """One row of output metrics for a single analyzed film."""

    filename: str
    group: str
    n_pixels_profile: int
    roi_x_min: int
    roi_x_max: int
    roi_y_min: int
    roi_y_max: int
    profile_axis: str
    pixel_size_mm: float
    reference_mode: str
    od_mode: str
    normalization_method: str
    smoothing_window_pixels: int
    calibration_used: bool
    normalization_value_od: float
    normalization_dose_cgy: Optional[float]
    min_od: float
    max_od: float
    max_dose_cgy: Optional[float]
    max_normalized_percent: float
    left_80_mm: Optional[float]
    left_20_mm: Optional[float]
    right_20_mm: Optional[float]
    right_80_mm: Optional[float]
    left_penumbra_80_20_mm: Optional[float]
    right_penumbra_80_20_mm: Optional[float]
    field_width_20_20_mm: Optional[float]
    field_width_50_50_mm: Optional[float]
    field_width_80_80_mm: Optional[float]
    status: str
    warning: str


@dataclass
class ProfileData:
    """Full profile data for plotting and optional per-film CSV output."""

    filename: str
    group: str
    x_mm: np.ndarray
    intensity_profile: np.ndarray
    od_profile: np.ndarray
    normalized_profile: np.ndarray
    smoothed_profile: np.ndarray
    crossings: Dict[float, Tuple[Optional[float], Optional[float]]]
    true_mid_index: Optional[int]
    dose_cgy_profile: Optional[np.ndarray] = None
    dose_gy_profile: Optional[np.ndarray] = None


# BASIC HELPERS

def find_tiff_files(folder: Path, patterns: Sequence[str]) -> List[Path]:
    """Return all TIFF files in a folder for all configured filename patterns."""
    files: List[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob.glob(str(folder / pattern)))
    return sorted(set(files))


def select_reference_file(files: Sequence[Path], name_contains: str) -> Path:
    """Find exactly one reference/unexposed file based on a case-insensitive substring."""
    matches = [f for f in files if name_contains.lower() in f.name.lower()]
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No reference TIFF found containing {name_contains!r}. "
            f"Put an unexposed film in the folder or edit REFERENCE_NAME_CONTAINS."
        )
    if len(matches) > 1:
        names = "\n  ".join(str(m.name) for m in matches)
        raise RuntimeError(
            f"More than one possible reference file found. Make this unambiguous.\n  {names}"
        )
    return matches[0]


def read_tiff_channel(path: Path, channel: str) -> np.ndarray:
    """
    Read a TIFF and return a 2D float64 image for the requested channel.

    Handles common cases:
        - 2D grayscale TIFF: returned directly.
        - 3D RGB TIFF with shape (rows, cols, channels): selected channel returned.
        - 3D stacks are rejected because film scans should be 2D images, not volumes.
    """
    image = tiff.imread(path)

    if image.ndim == 2:
        return image.astype(np.float64)

    if image.ndim == 3 and image.shape[-1] >= 3:
        channel_lc = channel.lower()
        if channel_lc == "red":
            return image[..., 0].astype(np.float64)
        if channel_lc == "green":
            return image[..., 1].astype(np.float64)
        if channel_lc == "blue":
            return image[..., 2].astype(np.float64)
        if channel_lc == "gray":
            return np.mean(image[..., :3].astype(np.float64), axis=2)
        if channel_lc == "mean_rgb":
            return np.mean(image[..., :3].astype(np.float64), axis=2)

        raise ValueError(
            f"Unsupported CHANNEL={channel!r}. Use red, green, blue, gray, or mean_rgb."
        )

    raise ValueError(
        f"Unsupported TIFF shape {image.shape} for {path.name}. "
        "Expected 2D grayscale or 3D RGB/RGBA image."
    )


def channel_to_index(channel: str) -> int:
    """Map the configured film channel to an RGB channel index for calibration."""
    channel_lc = channel.lower()
    if channel_lc == "red":
        return 0
    if channel_lc == "green":
        return 1
    if channel_lc == "blue":
        return 2
    if channel_lc in {"gray", "mean_rgb"}:
        raise ValueError("Dose calibration currently requires CHANNEL red, green, or blue.")
    raise ValueError(f"Unsupported CHANNEL={channel!r}. Use red, green, or blue for calibration.")


def validate_roi(image_shape: Tuple[int, int], x_min: int, x_max: int, y_min: int, y_max: int) -> None:
    """Make sure the requested ROI is inside the image and has positive area."""
    height, width = image_shape
    if not (0 <= x_min < x_max <= width):
        raise ValueError(
            f"Invalid x ROI [{x_min}, {x_max}) for image width {width}."
        )
    if not (0 <= y_min < y_max <= height):
        raise ValueError(
            f"Invalid y ROI [{y_min}, {y_max}) for image height {height}."
        )


def crop_roi(image: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
    """Return the rectangular ROI using Python slice semantics."""
    return image[y_min:y_max, x_min:x_max]


def profile_from_roi(roi: np.ndarray, axis: str) -> np.ndarray:
    """
    Collapse a 2D ROI into a 1D profile.

    axis="x": average over rows, producing a left-right profile.
    axis="y": average over columns, producing a top-bottom profile.
    """
    axis_lc = axis.lower()
    if axis_lc == "x":
        return np.mean(roi, axis=0)
    if axis_lc == "y":
        return np.mean(roi, axis=1)
    raise ValueError("PROFILE_AXIS must be 'x' or 'y'.")


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding so output length equals input length."""
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        raise ValueError("SMOOTHING_WINDOW_PIXELS must be odd, or 0/1 for no smoothing.")
    if window > len(y):
        raise ValueError("SMOOTHING_WINDOW_PIXELS cannot exceed the profile length.")

    pad = window // 2
    padded = np.pad(y, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def safe_log10_ratio(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    """Compute log10(numerator / denominator) after applying an intensity floor."""
    num = np.maximum(numerator, INTENSITY_FLOOR)
    den = np.maximum(denominator, INTENSITY_FLOOR)
    return np.log10(num / den)


def estimate_group_from_filename(path: Path) -> str:
    """Extract a conservative energy group label from the filename."""
    match = ENERGY_REGEX.search(path.stem)
    if match:
        return f"{match.group(1)} MeV"
    return "Unknown"


# PROFILE ANALYSIS HELPERS

def compute_od_profile(
    exposed_roi: np.ndarray,
    reference_roi: np.ndarray,
    axis: str,
    reference_mode: str,
    od_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute intensity and OD profiles from exposed/reference ROIs.

    Returns:
        intensity_profile: averaged exposed intensity profile.
        od_profile: OD or netOD profile.
    """
    intensity_profile = profile_from_roi(exposed_roi, axis)

    ref_mode = reference_mode.lower()
    od_mode_lc = od_mode.lower()

    if ref_mode == "scalar_roi_mean":
        i0 = float(np.mean(reference_roi))
        od_profile = safe_log10_ratio(i0, intensity_profile)
    elif ref_mode == "profile":
        i0_profile = profile_from_roi(reference_roi, axis)
        od_profile = safe_log10_ratio(i0_profile, intensity_profile)
    elif ref_mode == "image":
        # Pixel-wise OD image, then average to profile. This can correct spatial scanner response
        # if reference and exposed scans are aligned in the scanner bed.
        od_image = safe_log10_ratio(reference_roi, exposed_roi)
        od_profile = profile_from_roi(od_image, axis)
    else:
        raise ValueError(
            "REFERENCE_MODE must be 'scalar_roi_mean', 'profile', or 'image'."
        )

    if od_mode_lc == "od":
        return intensity_profile, od_profile

    if od_mode_lc == "netod":
        # Background/tail estimate from the lower 10% of OD values.
        # This is a generic fallback, not a substitute for a proper calibration protocol.
        tail_level = np.percentile(od_profile, 10)
        od_profile = od_profile - tail_level
        return intensity_profile, od_profile

    raise ValueError("OD_MODE must be 'od' or 'netod'.")


def normalize_signal_profile(
    signal_profile: np.ndarray,
    method: str,
    levels: Sequence[float],
    subtract_min: bool,
) -> Tuple[np.ndarray, float, Optional[int], str]:
    """
    Normalize a profile to percent.

    Returns:
        normalized_profile
        normalization_value
        true_mid_index, when applicable
        warning string
    """
    warning = ""
    signal = signal_profile.astype(np.float64)
    baseline = float(np.min(signal)) if subtract_min else 0.0
    shifted = signal - baseline

    if np.allclose(np.max(shifted), 0):
        raise ValueError("Profile is flat after baseline handling; cannot normalize.")

    method_lc = method.lower()

    if method_lc == "max":
        norm_value_shifted = float(np.max(shifted))
        true_mid_index = None

    elif method_lc == "percentile":
        norm_value_shifted = float(np.percentile(shifted, NORMALIZATION_PERCENTILE))
        true_mid_index = None

    elif method_lc == "midpoint_20_20":
        # First temporary normalization to estimate low-level field edges.
        temp_norm = shifted / float(np.max(shifted)) * 100.0
        low_level = float(min(levels))
        mid_idx = len(temp_norm) // 2

        left = find_crossing_one_side(temp_norm, low_level, side="left", center_index=mid_idx)
        right = find_crossing_one_side(temp_norm, low_level, side="right", center_index=mid_idx)

        if left is None or right is None:
            # Fallback to the geometric center if low-level crossings fail.
            true_mid_index = mid_idx
            warning = (
                f"Could not find both temporary {low_level:g}% crossings for midpoint normalization; "
                "used geometric center instead."
            )
        else:
            true_mid_index = int(round((left + right) / 2.0))
            true_mid_index = int(np.clip(true_mid_index, 0, len(signal) - 1))

        norm_value_shifted = float(shifted[true_mid_index])

    else:
        raise ValueError(
            "NORMALIZATION_METHOD must be 'midpoint_20_20', 'max', or 'percentile'."
        )

    if norm_value_shifted <= 0 or not np.isfinite(norm_value_shifted):
        raise ValueError("Invalid normalization value; check ROI, film orientation, and reference scan.")

    normalized = shifted / norm_value_shifted * 100.0
    normalization_value = norm_value_shifted + baseline
    return normalized, normalization_value, true_mid_index, warning


def normalize_profile(
    od_profile: np.ndarray,
    method: str,
    levels: Sequence[float],
) -> Tuple[np.ndarray, float, Optional[int], str]:
    """Normalize OD profile to percent with the historical background subtraction."""
    return normalize_signal_profile(
        signal_profile=od_profile,
        method=method,
        levels=levels,
        subtract_min=True,
    )


def find_crossing_one_side(
    y_percent: np.ndarray,
    level: float,
    side: str,
    center_index: int,
) -> Optional[float]:
    """
    Find one interpolated crossing index for a percent level on one side of the profile.

    The profile is expected to be high near the center and low outside the field.
    For the left side, we search from the center outward to the left.
    For the right side, we search from the center outward to the right.

    Returns an index in pixel units, not mm. The result may be fractional due to interpolation.
    """
    y = np.asarray(y_percent, dtype=np.float64)
    n = len(y)
    center_index = int(np.clip(center_index, 0, n - 1))

    if side == "left":
        search_indices = range(center_index, 0, -1)
        for i in search_indices:
            y1 = y[i]
            y0 = y[i - 1]
            if (y1 - level) * (y0 - level) <= 0 and y1 != y0:
                # Linear interpolation between i-1 and i.
                frac = (level - y0) / (y1 - y0)
                return (i - 1) + frac

    elif side == "right":
        search_indices = range(center_index, n - 1)
        for i in search_indices:
            y0 = y[i]
            y1 = y[i + 1]
            if (y0 - level) * (y1 - level) <= 0 and y1 != y0:
                frac = (level - y0) / (y1 - y0)
                return i + frac

    else:
        raise ValueError("side must be 'left' or 'right'.")

    return None


def find_crossings(
    y_percent: np.ndarray,
    levels: Iterable[float],
    center_index: Optional[int] = None,
) -> Dict[float, Tuple[Optional[float], Optional[float]]]:
    """Find left/right interpolated pixel-index crossings for multiple percent levels."""
    if center_index is None:
        center_index = len(y_percent) // 2
    crossings: Dict[float, Tuple[Optional[float], Optional[float]]] = {}
    for level in levels:
        left = find_crossing_one_side(y_percent, float(level), "left", center_index)
        right = find_crossing_one_side(y_percent, float(level), "right", center_index)
        crossings[float(level)] = (left, right)
    return crossings


def index_to_mm(index: Optional[float], pixel_size_mm: float) -> Optional[float]:
    """Convert a possibly missing fractional pixel index to mm."""
    if index is None:
        return None
    return float(index) * pixel_size_mm


def diff_mm(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Return a - b, preserving None if either value is missing."""
    if a is None or b is None:
        return None
    return float(a - b)


def fmt(value: Optional[float], digits: int = 3) -> str:
    """Format optional float values for terminal output."""
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


# PLOTTING AND OUTPUT

def save_profile_csv(profile: ProfileData, output_dir: Path) -> None:
    """Save one CSV file containing the full profile for one film."""
    safe_name = sanitize_filename(Path(profile.filename).stem)
    out_path = output_dir / f"profile_{safe_name}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "x_mm",
            "intensity",
            "net_od",
            "dose_cgy",
            "dose_gy",
            "normalized_percent",
            "smoothed_percent",
        ])
        dose_cgy = profile.dose_cgy_profile
        dose_gy = profile.dose_gy_profile
        for i, row in enumerate(
            zip(
                profile.x_mm,
                profile.intensity_profile,
                profile.od_profile,
                profile.normalized_profile,
                profile.smoothed_profile,
            )
        ):
            dose_cgy_value = "" if dose_cgy is None else f"{dose_cgy[i]:.10g}"
            dose_gy_value = "" if dose_gy is None else f"{dose_gy[i]:.10g}"
            writer.writerow([
                f"{row[0]:.10g}",
                f"{row[1]:.10g}",
                f"{row[2]:.10g}",
                dose_cgy_value,
                dose_gy_value,
                f"{row[3]:.10g}",
                f"{row[4]:.10g}",
            ])


def save_summary_csv(results: Sequence[FilmResult], output_dir: Path) -> Path:
    """Save all scalar metrics into one summary CSV file."""
    out_path = output_dir / "film_scan_summary_metrics.csv"
    if not results:
        return out_path
    fieldnames = list(asdict(results[0]).keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return out_path


def plot_profile(
    profile: ProfileData,
    result: FilmResult,
    output_dir: Path,
    show: bool,
) -> None:
    """Create and optionally save one plot for one film profile."""
    profile_label = "Relative dose" if result.calibration_used else "Normalized OD"
    y_label = "Dose (% normalized)" if result.calibration_used else "OD (% normalized)"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(profile.x_mm, profile.normalized_profile, linewidth=1.5, alpha=0.55, label=profile_label)
    ax.plot(profile.x_mm, profile.smoothed_profile, linewidth=2.0, label="Smoothed for metrics")

    draw_measurement_lines(ax, profile)

    ax.set_title(f"{profile.group} - {Path(profile.filename).stem}")
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    subtitle = (
        f"Left 80-20: {fmt(result.left_penumbra_80_20_mm)} mm | "
        f"Right 80-20: {fmt(result.right_penumbra_80_20_mm)} mm | "
        f"20-20 width: {fmt(result.field_width_20_20_mm)} mm"
    )
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if SAVE_PLOTS:
        safe_name = sanitize_filename(Path(profile.filename).stem)
        fig.savefig(output_dir / f"plot_{safe_name}.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_grouped_profiles(
    profiles: Sequence[ProfileData],
    results_by_filename: Dict[str, FilmResult],
    output_dir: Path,
    show: bool,
) -> None:
    """Save stacked profile plots grouped by energy/label."""
    groups: Dict[str, List[ProfileData]] = {}
    for profile in profiles:
        groups.setdefault(profile.group, []).append(profile)

    for group, group_profiles in groups.items():
        if not group_profiles:
            continue
        n = len(group_profiles)
        fig, axes = plt.subplots(n, 1, figsize=(9, max(3, 2.8 * n)), sharex=True)
        if n == 1:
            axes = [axes]
        fig.suptitle(f"Film Profiles - {group}", fontsize=14)

        for ax, profile in zip(axes, group_profiles):
            result = results_by_filename[profile.filename]
            ax.plot(profile.x_mm, profile.smoothed_profile, linewidth=2.0, label=Path(profile.filename).stem)
            draw_measurement_lines(ax, profile)
            ax.set_ylabel("%")
            ax.grid(True, alpha=0.35)
            ax.legend(fontsize=8)
            ax.text(
                0.01,
                0.93,
                f"L 80-20={fmt(result.left_penumbra_80_20_mm)} mm, "
                f"R 80-20={fmt(result.right_penumbra_80_20_mm)} mm, "
                f"20-20={fmt(result.field_width_20_20_mm)} mm",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )

        axes[-1].set_xlabel("Distance (mm)")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if SAVE_PLOTS:
            fig.savefig(output_dir / f"group_{sanitize_filename(group)}.png", dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)


def draw_measurement_lines(ax: plt.Axes, profile: ProfileData) -> None:
    """Draw horizontal percent levels and vertical crossing locations."""
    for level in MEASUREMENT_LEVELS:
        ax.axhline(level, linestyle="--", linewidth=1.0, alpha=0.8)

    ax.axhline(100, linestyle="--", linewidth=1.0, alpha=0.8, label="100% normalization")

    for level, (left_idx, right_idx) in profile.crossings.items():
        if level not in set(float(v) for v in MEASUREMENT_LEVELS):
            continue
        left_mm = index_to_mm(left_idx, PIXEL_SIZE_MM)
        right_mm = index_to_mm(right_idx, PIXEL_SIZE_MM)
        if left_mm is not None:
            ax.axvline(left_mm, linestyle=":", linewidth=1.0, alpha=0.9)
        if right_mm is not None:
            ax.axvline(right_mm, linestyle=":", linewidth=1.0, alpha=0.9)


def save_roi_qa_image(image: np.ndarray, source_path: Path, output_dir: Path) -> None:
    """Save a simple QA image showing where the ROI is located on the film scan."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap="gray")

    width = X_MAX - X_MIN
    height = Y_MAX - Y_MIN
    rect = plt.Rectangle((X_MIN, Y_MIN), width, height, fill=False, linewidth=2)
    ax.add_patch(rect)
    ax.set_title(f"ROI QA - {source_path.name}")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    fig.tight_layout()

    safe_name = sanitize_filename(source_path.stem)
    fig.savefig(output_dir / f"roi_{safe_name}.png", dpi=150)
    plt.close(fig)


def sanitize_filename(text: str) -> str:
    """Return a filesystem-safe filename fragment."""
    text = text.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


# SINGLE-FILM ANALYSIS

def analyze_one_film(
    film_path: Path,
    reference_image: np.ndarray,
    output_dir: Path,
    calibration: Optional[LoadedCalibration] = None,
) -> Tuple[FilmResult, Optional[ProfileData]]:
    """Analyze one exposed film and return scalar metrics plus full profile data."""
    group = estimate_group_from_filename(film_path)

    try:
        exposed_image = read_tiff_channel(film_path, CHANNEL)

        validate_roi(exposed_image.shape, X_MIN, X_MAX, Y_MIN, Y_MAX)

        if calibration is not None:
            intensity_profile, od_profile = calibrated_net_od_profile(
                exposed_image=exposed_image,
                reference_image=calibration.reference_image,
                roi=(X_MIN, X_MAX, Y_MIN, Y_MAX),
                axis=PROFILE_AXIS,
                reference_mode=REFERENCE_MODE,
                reference_roi=calibration.curve.roi,
            )
            dose_cgy_profile = calibration.curve.dose_cgy_from_net_od(od_profile)
            range_warnings: List[str] = []
            high_clip_count = int(np.count_nonzero(od_profile > calibration.curve.net_od[-1]))
            low_clip_count = int(np.count_nonzero(od_profile < calibration.curve.net_od[0]))
            if high_clip_count:
                range_warnings.append(
                    f"{high_clip_count}/{len(od_profile)} profile points exceeded "
                    f"{calibration.curve.max_dose_cgy:g} cGy calibration and were clipped"
                )
            if low_clip_count:
                range_warnings.append(
                    f"{low_clip_count}/{len(od_profile)} profile points were below 0 cGy calibration and were clipped"
                )
            dose_gy_profile = dose_cgy_profile / 100.0
            normalized_profile, norm_dose_cgy, true_mid_idx, warning = normalize_signal_profile(
                signal_profile=dose_cgy_profile,
                method=NORMALIZATION_METHOD,
                levels=MEASUREMENT_LEVELS,
                subtract_min=False,
            )
            if range_warnings:
                warning = "; ".join([part for part in [warning, *range_warnings] if part])
            if true_mid_idx is None:
                norm_idx = int(np.nanargmin(np.abs(dose_cgy_profile - norm_dose_cgy)))
            else:
                norm_idx = true_mid_idx
            norm_value_od = float(od_profile[norm_idx])
            calibration_used = True

        else:
            dose_cgy_profile = None
            dose_gy_profile = None
            norm_dose_cgy = None
            calibration_used = False

            if exposed_image.shape != reference_image.shape:
                raise ValueError(
                    f"Image shape mismatch: exposed {exposed_image.shape}, reference {reference_image.shape}. "
                    "Scans must have the same pixel dimensions for this script."
                )

            exposed_roi = crop_roi(exposed_image, X_MIN, X_MAX, Y_MIN, Y_MAX)
            reference_roi = crop_roi(reference_image, X_MIN, X_MAX, Y_MIN, Y_MAX)

            intensity_profile, od_profile = compute_od_profile(
                exposed_roi=exposed_roi,
                reference_roi=reference_roi,
                axis=PROFILE_AXIS,
                reference_mode=REFERENCE_MODE,
                od_mode=OD_MODE,
            )

            normalized_profile, norm_value_od, true_mid_idx, warning = normalize_profile(
                od_profile=od_profile,
                method=NORMALIZATION_METHOD,
                levels=MEASUREMENT_LEVELS,
            )

        smoothed_profile = moving_average(normalized_profile, SMOOTHING_WINDOW_PIXELS)

        all_levels = sorted(set(float(v) for v in list(MEASUREMENT_LEVELS) + list(FIELD_WIDTH_LEVELS)))
        center_idx = true_mid_idx if true_mid_idx is not None else len(smoothed_profile) // 2
        crossings = find_crossings(smoothed_profile, all_levels, center_index=center_idx)

        x_mm = np.arange(len(smoothed_profile), dtype=np.float64) * PIXEL_SIZE_MM

        # Convert key crossings from pixel-index units to mm.
        left_80_idx, right_80_idx = crossings.get(80.0, (None, None))
        left_20_idx, right_20_idx = crossings.get(20.0, (None, None))

        left_80_mm = index_to_mm(left_80_idx, PIXEL_SIZE_MM)
        left_20_mm = index_to_mm(left_20_idx, PIXEL_SIZE_MM)
        right_20_mm = index_to_mm(right_20_idx, PIXEL_SIZE_MM)
        right_80_mm = index_to_mm(right_80_idx, PIXEL_SIZE_MM)

        left_penumbra = diff_mm(left_20_mm, left_80_mm)
        right_penumbra = diff_mm(right_80_mm, right_20_mm)

        def width_at(level: float) -> Optional[float]:
            left_idx, right_idx = crossings.get(float(level), (None, None))
            left_mm = index_to_mm(left_idx, PIXEL_SIZE_MM)
            right_mm = index_to_mm(right_idx, PIXEL_SIZE_MM)
            return diff_mm(right_mm, left_mm)

        result = FilmResult(
            filename=film_path.name,
            group=group,
            n_pixels_profile=len(smoothed_profile),
            roi_x_min=X_MIN,
            roi_x_max=X_MAX,
            roi_y_min=Y_MIN,
            roi_y_max=Y_MAX,
            profile_axis=PROFILE_AXIS,
            pixel_size_mm=PIXEL_SIZE_MM,
            reference_mode=REFERENCE_MODE,
            od_mode=OD_MODE,
            normalization_method=NORMALIZATION_METHOD,
            smoothing_window_pixels=SMOOTHING_WINDOW_PIXELS,
            calibration_used=calibration_used,
            normalization_value_od=float(norm_value_od),
            normalization_dose_cgy=None if norm_dose_cgy is None else float(norm_dose_cgy),
            min_od=float(np.min(od_profile)),
            max_od=float(np.max(od_profile)),
            max_dose_cgy=None if dose_cgy_profile is None else float(np.nanmax(dose_cgy_profile)),
            max_normalized_percent=float(np.max(normalized_profile)),
            left_80_mm=left_80_mm,
            left_20_mm=left_20_mm,
            right_20_mm=right_20_mm,
            right_80_mm=right_80_mm,
            left_penumbra_80_20_mm=left_penumbra,
            right_penumbra_80_20_mm=right_penumbra,
            field_width_20_20_mm=width_at(20.0),
            field_width_50_50_mm=width_at(50.0),
            field_width_80_80_mm=width_at(80.0),
            status="ok",
            warning=warning,
        )

        profile = ProfileData(
            filename=film_path.name,
            group=group,
            x_mm=x_mm,
            intensity_profile=intensity_profile,
            od_profile=od_profile,
            normalized_profile=normalized_profile,
            smoothed_profile=smoothed_profile,
            crossings=crossings,
            true_mid_index=true_mid_idx,
            dose_cgy_profile=dose_cgy_profile,
            dose_gy_profile=dose_gy_profile,
        )

        if SAVE_ROI_QA_IMAGES:
            save_roi_qa_image(exposed_image, film_path, output_dir)

        return result, profile

    except Exception as exc:
        if not CONTINUE_ON_ERROR:
            raise

        result = FilmResult(
            filename=film_path.name,
            group=group,
            n_pixels_profile=0,
            roi_x_min=X_MIN,
            roi_x_max=X_MAX,
            roi_y_min=Y_MIN,
            roi_y_max=Y_MAX,
            profile_axis=PROFILE_AXIS,
            pixel_size_mm=PIXEL_SIZE_MM,
            reference_mode=REFERENCE_MODE,
            od_mode=OD_MODE,
            normalization_method=NORMALIZATION_METHOD,
            smoothing_window_pixels=SMOOTHING_WINDOW_PIXELS,
            calibration_used=calibration is not None,
            normalization_value_od=math.nan,
            normalization_dose_cgy=None,
            min_od=math.nan,
            max_od=math.nan,
            max_dose_cgy=None,
            max_normalized_percent=math.nan,
            left_80_mm=None,
            left_20_mm=None,
            right_20_mm=None,
            right_80_mm=None,
            left_penumbra_80_20_mm=None,
            right_penumbra_80_20_mm=None,
            field_width_20_20_mm=None,
            field_width_50_50_mm=None,
            field_width_80_80_mm=None,
            status="error",
            warning=str(exc),
        )
        return result, None


# MAIN SCRIPT

def main() -> None:
    """Run the full analysis pipeline."""
    plt.close("all")

    scan_folder = SCAN_FOLDER.expanduser().resolve()
    if not scan_folder.exists():
        raise FileNotFoundError(f"SCAN_FOLDER does not exist: {scan_folder}")

    output_dir = OUTPUT_FOLDER if OUTPUT_FOLDER is not None else scan_folder / "film_scan_analysis_outputs"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tiffs = find_tiff_files(scan_folder, TIFF_PATTERNS)
    if not all_tiffs:
        raise FileNotFoundError(f"No TIFF files found in {scan_folder}")

    calibration: Optional[LoadedCalibration] = None
    if USE_DOSE_CALIBRATION:
        calibration = load_calibration_curve(
            calibration_dir=CALIBRATION_FOLDER,
            channel=channel_to_index(CHANNEL),
            roi=(CALIBRATION_X_MIN, CALIBRATION_X_MAX, CALIBRATION_Y_MIN, CALIBRATION_Y_MAX),
            patterns=TIFF_PATTERNS,
        )
        write_calibration_outputs(calibration.curve, output_dir, label=CALIBRATION_LABEL)
        reference_path = calibration.curve.reference_file
        reference_image = calibration.reference_image
        exposed_files = [
            f for f in all_tiffs
            if REFERENCE_NAME_CONTAINS.lower() not in f.name.lower()
        ]
    else:
        reference_path = select_reference_file(all_tiffs, REFERENCE_NAME_CONTAINS)
        exposed_files = [f for f in all_tiffs if f != reference_path]
        reference_image = read_tiff_channel(reference_path, CHANNEL)
        validate_roi(reference_image.shape, X_MIN, X_MAX, Y_MIN, Y_MAX)

    if not exposed_files:
        raise FileNotFoundError("No exposed profile film TIFFs found in the scan folder.")

    print("=" * 78)
    print("Robust film scan profile analysis")
    print("=" * 78)
    print(f"Scan folder      : {scan_folder}")
    print(f"Output folder    : {output_dir}")
    print(f"Reference file   : {reference_path.name}")
    print(f"Dose calibration : {'enabled' if calibration is not None else 'disabled'}")
    if calibration is not None:
        print(f"Calibration dir  : {calibration.curve.calibration_dir}")
        print(f"Calibration max  : {calibration.curve.max_dose_cgy:g} cGy")
    print(f"Exposed films    : {len(exposed_files)}")
    print(f"Channel          : {CHANNEL}")
    print(f"ROI              : x=[{X_MIN}, {X_MAX}), y=[{Y_MIN}, {Y_MAX})")
    print(f"Profile axis     : {PROFILE_AXIS}")
    print(f"Pixel size       : {PIXEL_SIZE_MM} mm/pixel")
    print(f"Normalization    : {NORMALIZATION_METHOD}")
    print("=" * 78)

    if SAVE_ROI_QA_IMAGES:
        save_roi_qa_image(reference_image, reference_path, output_dir)

    results: List[FilmResult] = []
    profiles: List[ProfileData] = []

    for film_path in exposed_files:
        result, profile = analyze_one_film(film_path, reference_image, output_dir, calibration)
        results.append(result)
        if profile is not None:
            profiles.append(profile)
            if SAVE_PROFILE_CSVS:
                save_profile_csv(profile, output_dir)

        print(f"\n{result.group} | {result.filename}")
        if result.status == "ok":
            print(f"  Left 80-20 penumbra : {fmt(result.left_penumbra_80_20_mm)} mm")
            print(f"  Right 80-20 penumbra: {fmt(result.right_penumbra_80_20_mm)} mm")
            print(f"  20-20 field width   : {fmt(result.field_width_20_20_mm)} mm")
            print(f"  50-50 field width   : {fmt(result.field_width_50_50_mm)} mm")
            if result.warning:
                print(f"  Warning             : {result.warning}")
        else:
            print(f"  ERROR: {result.warning}")

    # Save scalar results.
    summary_path = save_summary_csv(results, output_dir)

    # Save plots.
    results_by_filename = {r.filename: r for r in results}
    if PLOT_MODE.lower() in {"individual", "both"}:
        for profile in profiles:
            plot_profile(profile, results_by_filename[profile.filename], output_dir, SHOW_PLOTS)

    if PLOT_MODE.lower() in {"energy", "both"}:
        plot_grouped_profiles(profiles, results_by_filename, output_dir, SHOW_PLOTS)

    print("\n" + "=" * 78)
    print("Done.")
    print(f"Summary CSV: {summary_path}")
    print(f"Outputs saved in: {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
