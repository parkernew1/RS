#!/usr/bin/env python3
"""Convert RayStation RT Dose DICOM files to central-axis PDD CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pydicom
from scipy.interpolate import RegularGridInterpolator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "Actual Runs" / "PDDs" / "dcm"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Actual Runs" / "PDDs" / "dcm-CSV"

# Legacy central-axis geometry from the earlier RayStation export script.
DEFAULT_X_ISO_MM = 0.6
DEFAULT_SURFACE_Y_MM = -377.1
DEFAULT_Z_ISO_MM = 0.0
DEFAULT_MAX_DEPTH_MM = 105.0
DEFAULT_POINTS = 1051


def load_dose_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dose_file = pydicom.dcmread(path)
    dose = dose_file.pixel_array.astype(float) * float(dose_file.DoseGridScaling)

    image_position = [float(value) for value in dose_file.ImagePositionPatient]
    row_spacing_mm = float(dose_file.PixelSpacing[0])
    col_spacing_mm = float(dose_file.PixelSpacing[1])

    z_mm = image_position[2] + np.asarray(dose_file.GridFrameOffsetVector, dtype=float)
    y_mm = image_position[1] + row_spacing_mm * np.arange(dose.shape[1], dtype=float)
    x_mm = image_position[0] + col_spacing_mm * np.arange(dose.shape[2], dtype=float)
    return z_mm, y_mm, x_mm, dose


def extract_pdd(
    dcm_path: Path,
    x_iso_mm: float,
    surface_y_mm: float,
    z_iso_mm: float,
    max_depth_mm: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_mm, y_mm, x_mm, dose = load_dose_grid(dcm_path)
    interpolator = RegularGridInterpolator((z_mm, y_mm, x_mm), dose, bounds_error=False, fill_value=0.0)

    depth_mm = np.linspace(0.0, max_depth_mm, n_points)
    x_positions = np.full_like(depth_mm, x_iso_mm)
    y_positions = surface_y_mm + depth_mm
    z_positions = np.full_like(depth_mm, z_iso_mm)
    dose_gy = interpolator((z_positions, y_positions, x_positions))
    max_dose = float(np.nanmax(dose_gy))
    if max_dose <= 0 or not np.isfinite(max_dose):
        pdd_percent = np.full_like(dose_gy, np.nan)
    else:
        pdd_percent = dose_gy / max_dose * 100.0
    return depth_mm, dose_gy, pdd_percent


def write_pdd_csv(path: Path, depth_mm: np.ndarray, dose_gy: np.ndarray, pdd_percent: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["Depth [mm]", "Dose [Gy]", "PDD [%]"])
        for depth_value, dose_value, pdd_value in zip(depth_mm, dose_gy, pdd_percent):
            writer.writerow([f"{depth_value:.10g}", f"{dose_value:.10g}", f"{pdd_value:.10g}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-iso-mm", type=float, default=DEFAULT_X_ISO_MM)
    parser.add_argument("--surface-y-mm", type=float, default=DEFAULT_SURFACE_Y_MM)
    parser.add_argument("--z-iso-mm", type=float, default=DEFAULT_Z_ISO_MM)
    parser.add_argument("--max-depth-mm", type=float, default=DEFAULT_MAX_DEPTH_MM)
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    dcm_files = sorted(input_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {input_dir}")

    print("=" * 78)
    print("RayStation DICOM PDD export")
    print("=" * 78)
    print(f"Input folder     : {input_dir}")
    print(f"Output folder    : {output_dir}")
    print(f"Surface y        : {args.surface_y_mm:g} mm")
    print(f"Central axis     : x={args.x_iso_mm:g} mm, z={args.z_iso_mm:g} mm")
    print(f"Max depth        : {args.max_depth_mm:g} mm")
    print("=" * 78)

    for dcm_path in dcm_files:
        depth_mm, dose_gy, pdd_percent = extract_pdd(
            dcm_path=dcm_path,
            x_iso_mm=args.x_iso_mm,
            surface_y_mm=args.surface_y_mm,
            z_iso_mm=args.z_iso_mm,
            max_depth_mm=args.max_depth_mm,
            n_points=args.points,
        )
        out_path = output_dir / f"{dcm_path.stem}.csv"
        write_pdd_csv(out_path, depth_mm, dose_gy, pdd_percent)
        dmax_idx = int(np.nanargmax(dose_gy))
        print(
            f"{dcm_path.name} -> {out_path.name}  "
            f"max={dose_gy[dmax_idx]:.6g} Gy at {depth_mm[dmax_idx]:.3f} mm"
        )

    print("=" * 78)
    print("Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
