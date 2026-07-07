#!/usr/bin/env python3
"""Shared film calibration helpers for scanned EBT film workflows.

The calibration expects TIFF files named with their delivered dose, for example
``0cGy.tif``, ``50_cGy.tiff``, or ``6MeV_100cGy_scan.tif``.
"""

from __future__ import annotations

import csv
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_DIR = REPO_ROOT / "Actual Runs" / "6MeV_calibration_films_07062026"
TIFF_PATTERNS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
EXPECTED_CALIBRATION_DOSES_CGY = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0)
DOSE_RE = re.compile(r"(?<!\d)(0|50|100|150|200|250|300)\s*c\s*gy(?!\d)", re.IGNORECASE)


@dataclass(frozen=True)
class CalibrationPoint:
    dose_cgy: float
    net_od: float
    mean_signal: float
    source_file: Path


@dataclass(frozen=True)
class CalibrationCurve:
    calibration_dir: Path
    reference_file: Path
    channel: int
    roi: tuple[int, int, int, int]
    doses_cgy: np.ndarray
    net_od: np.ndarray
    points: tuple[CalibrationPoint, ...]

    def dose_cgy_from_net_od(self, values: np.ndarray | Sequence[float] | float) -> np.ndarray:
        """Convert net optical density values to dose in cGy by interpolation."""
        od_values = np.asarray(values, dtype=float)
        finite = np.isfinite(od_values)
        dose = np.full(od_values.shape, np.nan, dtype=float)
        clipped = np.clip(od_values[finite], self.net_od[0], self.net_od[-1])
        dose[finite] = np.interp(clipped, self.net_od, self.doses_cgy)
        return dose

    @property
    def max_dose_cgy(self) -> float:
        return float(np.nanmax(self.doses_cgy))


@dataclass(frozen=True)
class LoadedCalibration:
    curve: CalibrationCurve
    reference_image: np.ndarray


def find_tiff_files(folder: Path, patterns: Iterable[str] = TIFF_PATTERNS) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(folder.glob(pattern))
    return sorted(set(files))


def parse_calibration_dose_cgy(path: Path) -> float | None:
    """Parse the calibration dose from a filename.

    Preferred names include a cGy suffix. As a convenience, bare numeric tokens
    matching the expected calibration doses are accepted while ignoring dates
    and beam labels such as 6MeV.
    """
    stem = path.stem
    match = DOSE_RE.search(stem)
    if match:
        return float(match.group(1))

    expected = {str(int(dose)): dose for dose in EXPECTED_CALIBRATION_DOSES_CGY}
    for token in re.split(r"[^0-9]+", stem):
        if token in expected:
            return expected[token]
    return None


def read_tiff_channel(path: Path, channel: int) -> np.ndarray:
    image = tifffile.imread(path)
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(float)
    if array.ndim == 3:
        if channel < 0 or channel >= array.shape[-1]:
            raise ValueError(f"{path.name} has {array.shape[-1]} channels; channel {channel} is invalid")
        return array[..., channel].astype(float)
    raise ValueError(f"{path.name} has unsupported TIFF shape {array.shape}")


def validate_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> None:
    x_min, x_max, y_min, y_max = roi
    if not (0 <= x_min < x_max <= image.shape[1] and 0 <= y_min < y_max <= image.shape[0]):
        raise ValueError(
            f"ROI {(x_min, x_max, y_min, y_max)} is outside image bounds "
            f"(width={image.shape[1]}, height={image.shape[0]})"
        )


def crop_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    validate_roi(image, roi)
    x_min, x_max, y_min, y_max = roi
    return image[y_min:y_max, x_min:x_max]


def profile_from_roi(roi_image: np.ndarray, axis: str) -> np.ndarray:
    if axis.lower() == "x":
        return np.nanmean(roi_image, axis=0)
    if axis.lower() == "y":
        return np.nanmean(roi_image, axis=1)
    raise ValueError("axis must be 'x' or 'y'")


def net_od_from_signals(reference_signal: np.ndarray | float, exposed_signal: np.ndarray | float) -> np.ndarray:
    reference = np.asarray(reference_signal, dtype=float)
    exposed = np.asarray(exposed_signal, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        od = np.log10(reference / exposed)
    return np.where(np.isfinite(od), od, np.nan)


def net_od_profile(
    exposed_image: np.ndarray,
    reference_image: np.ndarray,
    roi: tuple[int, int, int, int],
    axis: str,
    reference_mode: str = "scalar_roi_mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Return exposed intensity and netOD profiles for a scanned film.

    ``scalar_roi_mean`` uses the mean 0 cGy calibration signal as I0. ``profile``
    uses the matching reference-film profile across the same ROI. ``image``
    computes pixel-wise netOD and then averages it into a profile.
    """
    exposed_roi = crop_roi(exposed_image, roi)
    reference_roi = crop_roi(reference_image, roi)
    exposed_profile = profile_from_roi(exposed_roi, axis)

    mode = reference_mode.lower()
    if mode == "scalar_roi_mean":
        reference_signal: np.ndarray | float = float(np.nanmean(reference_roi))
        od_profile = net_od_from_signals(reference_signal, exposed_profile)
    elif mode == "profile":
        reference_signal = profile_from_roi(reference_roi, axis)
        od_profile = net_od_from_signals(reference_signal, exposed_profile)
    elif mode == "image":
        od_image = net_od_from_signals(reference_roi, exposed_roi)
        od_profile = profile_from_roi(od_image, axis)
    else:
        raise ValueError("reference_mode must be 'scalar_roi_mean', 'profile', or 'image'")

    return exposed_profile, od_profile


def load_calibration_curve(
    calibration_dir: Path = DEFAULT_CALIBRATION_DIR,
    channel: int = 0,
    roi: tuple[int, int, int, int] = (100, 900, 100, 900),
    patterns: Iterable[str] = TIFF_PATTERNS,
) -> LoadedCalibration:
    calibration_dir = Path(calibration_dir)
    files = find_tiff_files(calibration_dir, patterns)
    if not files:
        raise FileNotFoundError(
            f"No calibration TIFFs found in {calibration_dir}. "
            "Add scans named with 0, 50, 100, 150, 200, 250, and 300 cGy."
        )

    dose_files: dict[float, list[Path]] = {}
    skipped: list[Path] = []
    for path in files:
        dose = parse_calibration_dose_cgy(path)
        if dose is None:
            skipped.append(path)
            continue
        dose_files.setdefault(dose, []).append(path)

    if 0.0 not in dose_files:
        names = ", ".join(path.name for path in files)
        raise FileNotFoundError(f"No 0 cGy calibration film was found in {calibration_dir}. Files: {names}")

    nonzero_doses = sorted(dose for dose in dose_files if dose > 0)
    if len(nonzero_doses) < 2:
        raise ValueError("At least two nonzero calibration films are required to build a dose curve")

    reference_file = sorted(dose_files[0.0])[0]
    reference_image = read_tiff_channel(reference_file, channel)
    reference_roi = crop_roi(reference_image, roi)
    reference_mean = float(np.nanmean(reference_roi))
    if not math.isfinite(reference_mean) or reference_mean <= 0:
        raise ValueError(f"Invalid 0 cGy reference signal in {reference_file.name}: {reference_mean}")

    raw_points = [CalibrationPoint(0.0, 0.0, reference_mean, reference_file)]
    for dose in nonzero_doses:
        for path in sorted(dose_files[dose]):
            image = read_tiff_channel(path, channel)
            roi_image = crop_roi(image, roi)
            mean_signal = float(np.nanmean(roi_image))
            net_od = float(net_od_from_signals(reference_mean, mean_signal))
            raw_points.append(CalibrationPoint(dose, net_od, mean_signal, path))

    averaged_points: list[CalibrationPoint] = []
    for dose in sorted({point.dose_cgy for point in raw_points}):
        matching = [point for point in raw_points if point.dose_cgy == dose]
        if len(matching) == 1:
            averaged_points.append(matching[0])
            continue
        averaged_points.append(
            CalibrationPoint(
                dose_cgy=dose,
                net_od=float(np.nanmean([point.net_od for point in matching])),
                mean_signal=float(np.nanmean([point.mean_signal for point in matching])),
                source_file=matching[0].source_file,
            )
        )

    doses = np.array([point.dose_cgy for point in averaged_points], dtype=float)
    net_od = np.array([point.net_od for point in averaged_points], dtype=float)
    if not np.all(np.isfinite(net_od)):
        raise ValueError("Calibration produced non-finite netOD values")
    if np.any(np.diff(net_od) <= 0):
        raise ValueError(
            "Calibration netOD values are not strictly increasing with dose. "
            "Check dose labels, scan orientation, and ROI placement."
        )

    curve = CalibrationCurve(
        calibration_dir=calibration_dir,
        reference_file=reference_file,
        channel=channel,
        roi=roi,
        doses_cgy=doses,
        net_od=net_od,
        points=tuple(averaged_points),
    )
    return LoadedCalibration(curve=curve, reference_image=reference_image)


def write_calibration_outputs(curve: CalibrationCurve, output_dir: Path, label: str = "6MeV") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{label}_film_calibration_curve.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["dose_cgy", "net_od", "mean_signal", "source_file"])
        for point in curve.points:
            writer.writerow([point.dose_cgy, point.net_od, point.mean_signal, point.source_file.name])

    png_path = output_dir / f"{label}_film_calibration_curve.png"
    plt.figure(figsize=(6, 4))
    plt.plot(curve.net_od, curve.doses_cgy, marker="o", linewidth=1.5)
    plt.xlabel("Net optical density")
    plt.ylabel("Dose (cGy)")
    plt.title(f"{label} film calibration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
