#!/usr/bin/env python3
"""Create calibrated film percent-depth-dose curves from scanned TIFF films.

Default inputs:
    Actual Runs/6MeV_calibration_films_07062026/
        0, 50, 100, 150, 200, 250, and 300 cGy calibration TIFFs
    Actual Runs/PDDs/Scans/
        one or more PDD film TIFFs

Default outputs:
    Actual Runs/PDDs/Results/
        calibration curve CSV/PNG, per-film PDD CSV/PNG, and summary metrics
"""

from __future__ import annotations

import csv
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np

from film_calibration import (
    DEFAULT_CALIBRATION_DIR,
    TIFF_PATTERNS,
    find_tiff_files,
    load_calibration_curve,
    net_od_profile,
    read_tiff_channel,
    write_calibration_outputs,
)


# USER CONFIG

REPO_ROOT = Path(__file__).resolve().parents[1]

SCAN_FOLDER = REPO_ROOT / "Actual Runs" / "PDDs" / "Scans"
OUTPUT_FOLDER = REPO_ROOT / "Actual Runs" / "PDDs" / "Results"

CALIBRATION_FOLDER = DEFAULT_CALIBRATION_DIR
CALIBRATION_LABEL = "6MeV"

# options: "red", "green", "blue"
CHANNEL = "red"

# PDD measurement ROI in pixel coordinates.
# For PDDs, x_min/x_max define the strip width being averaged and y_min/y_max
# define the depth direction when PDD_AXIS is "y".
X_MIN = 50
X_MAX = 220
Y_MIN = 40
Y_MAX = 770

# Uniform patch sampled on the calibration films to build the dose curve.
CALIBRATION_X_MIN = 60
CALIBRATION_X_MAX = 210
CALIBRATION_Y_MIN = 80
CALIBRATION_Y_MAX = 260

# "y" means depth runs top-to-bottom through the ROI. "x" means left-to-right.
PDD_AXIS = "y"

# Use "forward" if depth increases with increasing pixel index after cropping.
# Use "reverse" if the film was scanned with the deepest end first.
DEPTH_DIRECTION = "reverse"

# Pixel size in mm/pixel.
PIXEL_SIZE_MM = 0.127
SURFACE_OFFSET_MM = 0.0

# Reference handling for netOD from the 0 cGy calibration film.
# "scalar_roi_mean" is the most robust default.
REFERENCE_MODE = "scalar_roi_mean"

SMOOTHING_WINDOW_PIXELS = 1

SAVE_PDD_CSVS = True
SAVE_PLOTS = True
SAVE_ROI_QA_IMAGES = True
SHOW_PLOTS = False


@dataclass
class PddResult:
    filename: str
    n_depth_pixels: int
    roi_x_min: int
    roi_x_max: int
    roi_y_min: int
    roi_y_max: int
    pdd_axis: str
    depth_direction: str
    pixel_size_mm: float
    surface_offset_mm: float
    calibration_folder: str
    calibration_max_cgy: float
    dmax_depth_mm: float
    dmax_dose_cgy: float
    surface_percent: float
    r90_depth_mm: Optional[float]
    r80_depth_mm: Optional[float]
    r50_depth_mm: Optional[float]
    distal_20_depth_mm: Optional[float]
    status: str
    warning: str


@dataclass
class PddProfile:
    filename: str
    depth_mm: np.ndarray
    intensity_profile: np.ndarray
    net_od: np.ndarray
    dose_cgy: np.ndarray
    dose_gy: np.ndarray
    smoothed_dose_cgy: np.ndarray
    pdd_percent: np.ndarray


def channel_to_index(channel: str) -> int:
    channel_lc = channel.lower()
    if channel_lc == "red":
        return 0
    if channel_lc == "green":
        return 1
    if channel_lc == "blue":
        return 2
    raise ValueError("CHANNEL must be red, green, or blue for calibrated PDD analysis.")


def sanitize_filename(text: str) -> str:
    text = text.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        raise ValueError("SMOOTHING_WINDOW_PIXELS must be odd, or 0/1 for no smoothing.")
    if window > len(y):
        raise ValueError("SMOOTHING_WINDOW_PIXELS cannot exceed the PDD length.")
    pad = window // 2
    padded = np.pad(y, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def apply_depth_direction(*profiles: np.ndarray) -> tuple[np.ndarray, ...]:
    direction = DEPTH_DIRECTION.lower()
    if direction == "forward":
        return tuple(profile.copy() for profile in profiles)
    if direction == "reverse":
        return tuple(profile[::-1].copy() for profile in profiles)
    raise ValueError("DEPTH_DIRECTION must be 'forward' or 'reverse'.")


def distal_depth_at_percent(depth_mm: np.ndarray, pdd_percent: np.ndarray, level: float) -> Optional[float]:
    dmax_idx = int(np.nanargmax(pdd_percent))
    for i in range(dmax_idx, len(pdd_percent) - 1):
        y0 = pdd_percent[i]
        y1 = pdd_percent[i + 1]
        if not np.isfinite(y0) or not np.isfinite(y1) or y0 == y1:
            continue
        if (y0 - level) * (y1 - level) <= 0:
            frac = (level - y0) / (y1 - y0)
            return float(depth_mm[i] + frac * (depth_mm[i + 1] - depth_mm[i]))
    return None


def save_roi_qa_image(image: np.ndarray, source_path: Path, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap="gray")
    rect = plt.Rectangle((X_MIN, Y_MIN), X_MAX - X_MIN, Y_MAX - Y_MIN, fill=False, linewidth=2)
    ax.add_patch(rect)
    ax.set_title(f"PDD ROI QA - {source_path.name}")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    fig.tight_layout()
    fig.savefig(output_dir / f"pdd_roi_{sanitize_filename(source_path.stem)}.png", dpi=150)
    plt.close(fig)


def save_pdd_csv(profile: PddProfile, output_dir: Path) -> Path:
    out_path = output_dir / f"pdd_{sanitize_filename(Path(profile.filename).stem)}.csv"
    with out_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow([
            "depth_mm",
            "intensity",
            "net_od",
            "dose_cgy",
            "dose_gy",
            "smoothed_dose_cgy",
            "pdd_percent",
        ])
        for row in zip(
            profile.depth_mm,
            profile.intensity_profile,
            profile.net_od,
            profile.dose_cgy,
            profile.dose_gy,
            profile.smoothed_dose_cgy,
            profile.pdd_percent,
        ):
            writer.writerow([f"{value:.10g}" for value in row])
    return out_path


def plot_pdd(profile: PddProfile, result: PddResult, output_dir: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(profile.depth_mm, profile.pdd_percent, linewidth=2.0)
    for level in (90.0, 80.0, 50.0, 20.0):
        ax.axhline(level, linestyle="--", linewidth=0.9, alpha=0.6)
    ax.axvline(result.dmax_depth_mm, linestyle=":", linewidth=1.2, alpha=0.9, label="dmax")
    ax.set_title(Path(profile.filename).stem)
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("PDD (%)")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(output_dir / f"pdd_{sanitize_filename(Path(profile.filename).stem)}.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def analyze_one_pdd(
    film_path: Path,
    calibration,
    output_dir: Path,
) -> tuple[PddResult, PddProfile]:
    channel_index = channel_to_index(CHANNEL)
    image = read_tiff_channel(film_path, channel_index)

    intensity, net_od = net_od_profile(
        exposed_image=image,
        reference_image=calibration.reference_image,
        roi=(X_MIN, X_MAX, Y_MIN, Y_MAX),
        axis=PDD_AXIS,
        reference_mode=REFERENCE_MODE,
        reference_roi=calibration.curve.roi,
    )
    intensity, net_od = apply_depth_direction(intensity, net_od)

    dose_cgy = calibration.curve.dose_cgy_from_net_od(net_od)
    range_warnings: list[str] = []
    high_clip_count = int(np.count_nonzero(net_od > calibration.curve.net_od[-1]))
    low_clip_count = int(np.count_nonzero(net_od < calibration.curve.net_od[0]))
    if high_clip_count:
        range_warnings.append(
            f"{high_clip_count}/{len(net_od)} PDD points exceeded "
            f"{calibration.curve.max_dose_cgy:g} cGy calibration and were clipped"
        )
    if low_clip_count:
        range_warnings.append(
            f"{low_clip_count}/{len(net_od)} PDD points were below 0 cGy calibration and were clipped"
        )
    dose_gy = dose_cgy / 100.0
    smoothed_dose_cgy = moving_average(dose_cgy, SMOOTHING_WINDOW_PIXELS)

    max_dose = float(np.nanmax(smoothed_dose_cgy))
    if not math.isfinite(max_dose) or max_dose <= 0:
        raise ValueError("PDD dose profile is flat or invalid; check ROI and calibration scans.")

    pdd_percent = smoothed_dose_cgy / max_dose * 100.0
    depth_mm = np.arange(len(pdd_percent), dtype=float) * PIXEL_SIZE_MM + SURFACE_OFFSET_MM
    dmax_idx = int(np.nanargmax(smoothed_dose_cgy))

    profile = PddProfile(
        filename=film_path.name,
        depth_mm=depth_mm,
        intensity_profile=intensity,
        net_od=net_od,
        dose_cgy=dose_cgy,
        dose_gy=dose_gy,
        smoothed_dose_cgy=smoothed_dose_cgy,
        pdd_percent=pdd_percent,
    )

    result = PddResult(
        filename=film_path.name,
        n_depth_pixels=len(depth_mm),
        roi_x_min=X_MIN,
        roi_x_max=X_MAX,
        roi_y_min=Y_MIN,
        roi_y_max=Y_MAX,
        pdd_axis=PDD_AXIS,
        depth_direction=DEPTH_DIRECTION,
        pixel_size_mm=PIXEL_SIZE_MM,
        surface_offset_mm=SURFACE_OFFSET_MM,
        calibration_folder=str(calibration.curve.calibration_dir),
        calibration_max_cgy=calibration.curve.max_dose_cgy,
        dmax_depth_mm=float(depth_mm[dmax_idx]),
        dmax_dose_cgy=max_dose,
        surface_percent=float(pdd_percent[0]),
        r90_depth_mm=distal_depth_at_percent(depth_mm, pdd_percent, 90.0),
        r80_depth_mm=distal_depth_at_percent(depth_mm, pdd_percent, 80.0),
        r50_depth_mm=distal_depth_at_percent(depth_mm, pdd_percent, 50.0),
        distal_20_depth_mm=distal_depth_at_percent(depth_mm, pdd_percent, 20.0),
        status="ok",
        warning="; ".join(range_warnings),
    )

    if SAVE_ROI_QA_IMAGES:
        save_roi_qa_image(image, film_path, output_dir)
    if SAVE_PDD_CSVS:
        save_pdd_csv(profile, output_dir)
    plot_pdd(profile, result, output_dir, SHOW_PLOTS)

    return result, profile


def save_summary_csv(results: Sequence[PddResult], output_dir: Path) -> Path:
    out_path = output_dir / "film_pdd_summary_metrics.csv"
    if not results:
        return out_path
    with out_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(results[0]).keys()), lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return out_path


def main() -> None:
    plt.close("all")
    scan_folder = SCAN_FOLDER.expanduser().resolve()
    output_dir = OUTPUT_FOLDER.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scan_folder.exists():
        raise FileNotFoundError(f"SCAN_FOLDER does not exist: {scan_folder}")

    pdd_files = find_tiff_files(scan_folder, TIFF_PATTERNS)
    if not pdd_files:
        raise FileNotFoundError(f"No PDD TIFF files found in {scan_folder}")

    calibration = load_calibration_curve(
        calibration_dir=CALIBRATION_FOLDER,
        channel=channel_to_index(CHANNEL),
        roi=(CALIBRATION_X_MIN, CALIBRATION_X_MAX, CALIBRATION_Y_MIN, CALIBRATION_Y_MAX),
        patterns=TIFF_PATTERNS,
    )
    write_calibration_outputs(calibration.curve, output_dir, label=CALIBRATION_LABEL)

    print("=" * 78)
    print("Calibrated film PDD analysis")
    print("=" * 78)
    print(f"Scan folder      : {scan_folder}")
    print(f"Output folder    : {output_dir}")
    print(f"Calibration dir  : {calibration.curve.calibration_dir}")
    print(f"Calibration max  : {calibration.curve.max_dose_cgy:g} cGy")
    print(f"PDD films        : {len(pdd_files)}")
    print(f"Channel          : {CHANNEL}")
    print(f"PDD ROI          : x=[{X_MIN}, {X_MAX}), y=[{Y_MIN}, {Y_MAX})")
    print(f"Calibration ROI  : x=[{CALIBRATION_X_MIN}, {CALIBRATION_X_MAX}), "
          f"y=[{CALIBRATION_Y_MIN}, {CALIBRATION_Y_MAX})")
    print(f"Depth axis       : {PDD_AXIS}, {DEPTH_DIRECTION}")
    print("=" * 78)

    results: list[PddResult] = []
    for film_path in pdd_files:
        result, _profile = analyze_one_pdd(film_path, calibration, output_dir)
        results.append(result)
        print(f"\n{result.filename}")
        print(f"  dmax depth : {result.dmax_depth_mm:.3f} mm")
        print(f"  dmax dose  : {result.dmax_dose_cgy:.3f} cGy")
        print(f"  R90/R80/R50: {fmt_optional(result.r90_depth_mm)} / "
              f"{fmt_optional(result.r80_depth_mm)} / {fmt_optional(result.r50_depth_mm)} mm")
        if result.warning:
            print(f"  Warning    : {result.warning}")

    summary_path = save_summary_csv(results, output_dir)
    print("\n" + "=" * 78)
    print("Done.")
    print(f"Summary CSV: {summary_path}")
    print(f"Outputs saved in: {output_dir}")
    print("=" * 78)


def fmt_optional(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
