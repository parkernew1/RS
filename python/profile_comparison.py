from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "profile_comparison.py requires numpy. Use the same Python environment used for "
        "octavius_mcc_profile_to_csv.py, or install numpy."
    ) from exc

from octavius_mcc_profile_to_csv import (
    AxisMetrics,
    analyze_axis_profile,
    build_aligned_dose_matrix,
    read_mcc,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "Trial Runs" / "Octavius_Raystation_comparison_copper"
DEFAULT_CSV_DIR = DEFAULT_DATA_DIR / "All_CSV"
DEFAULT_MCC_DIR = DEFAULT_DATA_DIR / "All_MCC"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "comparison_results"


@dataclass
class ProfileData:
    source: str
    name: str
    x_mm: np.ndarray
    dose_gy: np.ndarray
    relative: np.ndarray
    metrics: AxisMetrics
    central_y_mm: Optional[float] = None


@dataclass
class ComparisonResult:
    name: str
    csv_file: Path
    mcc_file: Path
    output_dir: Path
    alignment_method: str
    csv_shift_mm: float
    field_center_shift_mm: float
    common_x_min_mm: float
    common_x_max_mm: float
    n_points: int
    mcc: ProfileData
    csv_profile: ProfileData
    point_rows: List[Dict[str, object]]
    stats: Dict[str, Optional[float]]


def format_optional(value: Optional[float], digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.{digits}f}"


def csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def read_raystation_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read the RayStation point-dose profile exported by analyze.py."""
    x_values: List[float] = []
    dose_values: List[float] = []

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name}: CSV is empty.")

        fieldnames = {name.strip(): name for name in reader.fieldnames}
        try:
            x_col = fieldnames["X [mm]"]
            dose_col = fieldnames["Dose [Gy]"]
        except KeyError as exc:
            raise ValueError(
                f"{path.name}: expected columns 'X [mm]' and 'Dose [Gy]'."
            ) from exc

        for row in reader:
            if not row:
                continue
            x_text = row.get(x_col, "").strip()
            dose_text = row.get(dose_col, "").strip()
            if not x_text or not dose_text:
                continue
            x_values.append(float(x_text))
            dose_values.append(float(dose_text))

    if len(x_values) < 2:
        raise ValueError(f"{path.name}: need at least two profile points.")

    return sorted_profile_arrays(np.asarray(x_values, dtype=float), np.asarray(dose_values, dtype=float))


def sorted_profile_arrays(x_mm: np.ndarray, dose_gy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x_mm) & np.isfinite(dose_gy)
    x_mm = x_mm[finite]
    dose_gy = dose_gy[finite]
    if x_mm.size < 2:
        raise ValueError("Profile has fewer than two finite points.")

    order = np.argsort(x_mm)
    x_mm = x_mm[order]
    dose_gy = dose_gy[order]

    unique_x, inverse = np.unique(x_mm, return_inverse=True)
    if unique_x.size != x_mm.size:
        summed = np.zeros(unique_x.shape, dtype=float)
        counts = np.zeros(unique_x.shape, dtype=float)
        np.add.at(summed, inverse, dose_gy)
        np.add.at(counts, inverse, 1.0)
        x_mm = unique_x
        dose_gy = summed / counts

    return x_mm, dose_gy


def make_profile(source: str, name: str, x_mm: np.ndarray, dose_gy: np.ndarray) -> ProfileData:
    x_mm, dose_gy = sorted_profile_arrays(x_mm, dose_gy)
    relative, metrics = analyze_axis_profile(f"{source}_crossplane", x_mm, dose_gy)
    return ProfileData(source=source, name=name, x_mm=x_mm, dose_gy=dose_gy, relative=relative, metrics=metrics)


def read_mcc_central_crossplane(path: Path) -> ProfileData:
    """Extract the central-axis cross-plane profile from an OCTAVIUS MCC file."""
    profiles = read_mcc(path)
    x_positions_mm, y_positions_mm, dose_matrix, _, _ = build_aligned_dose_matrix(profiles)
    central_y_idx = int(np.argmin(np.abs(y_positions_mm)))
    central_y_mm = float(y_positions_mm[central_y_idx])
    profile = make_profile("mcc", path.stem, x_positions_mm, dose_matrix[central_y_idx, :])
    profile.central_y_mm = central_y_mm
    return profile


def read_csv_profile(path: Path) -> ProfileData:
    x_mm, dose_gy = read_raystation_csv(path)
    return make_profile("csv", path.stem, x_mm, dose_gy)


def shifted_metric(value: Optional[float], shift_mm: float) -> Optional[float]:
    if value is None:
        return None
    return float(value) + shift_mm


def metric_difference(csv_value_mm: Optional[float], mcc_value_mm: Optional[float], shift_mm: float = 0.0) -> Optional[float]:
    if csv_value_mm is None or mcc_value_mm is None:
        return None
    return float(csv_value_mm) + shift_mm - float(mcc_value_mm)


def width_difference(csv_width_mm: Optional[float], mcc_width_mm: Optional[float]) -> Optional[float]:
    if csv_width_mm is None or mcc_width_mm is None:
        return None
    return float(csv_width_mm) - float(mcc_width_mm)


def make_common_grid(
    mcc_x_mm: np.ndarray,
    csv_x_mm: np.ndarray,
    csv_shift_mm: float,
    grid_step_mm: float,
) -> np.ndarray:
    start = max(float(np.min(mcc_x_mm)), float(np.min(csv_x_mm) + csv_shift_mm))
    stop = min(float(np.max(mcc_x_mm)), float(np.max(csv_x_mm) + csv_shift_mm))
    if stop <= start:
        raise ValueError("MCC and shifted CSV profiles do not overlap in x.")

    start = math.ceil(start / grid_step_mm) * grid_step_mm
    stop = math.floor(stop / grid_step_mm) * grid_step_mm
    if stop <= start:
        raise ValueError("Common x range is smaller than the requested grid spacing.")

    n_steps = int(round((stop - start) / grid_step_mm))
    return start + grid_step_mm * np.arange(n_steps + 1, dtype=float)


def interpolate_shifted(
    grid_x_mm: np.ndarray,
    source_x_mm: np.ndarray,
    source_values: np.ndarray,
    source_shift_mm: float = 0.0,
) -> np.ndarray:
    return np.interp(grid_x_mm - source_shift_mm, source_x_mm, source_values)


def profile_stats(diff_percent_points: np.ndarray, mask: np.ndarray) -> Dict[str, Optional[float]]:
    if not np.any(mask):
        return {
            "mean_diff_percent_points": None,
            "mean_abs_diff_percent_points": None,
            "rms_diff_percent_points": None,
            "max_abs_diff_percent_points": None,
        }

    values = diff_percent_points[mask]
    return {
        "mean_diff_percent_points": float(np.mean(values)),
        "mean_abs_diff_percent_points": float(np.mean(np.abs(values))),
        "rms_diff_percent_points": float(np.sqrt(np.mean(values * values))),
        "max_abs_diff_percent_points": float(np.max(np.abs(values))),
    }


def classify_region(mcc_relative: float, csv_relative: float) -> str:
    high = max(mcc_relative, csv_relative)
    low = min(mcc_relative, csv_relative)
    if high >= 0.8:
        return "high_dose"
    if high >= 0.2 and low <= 0.8:
        return "penumbra_or_shoulder"
    if high >= 0.1:
        return "low_dose"
    return "tail"


def field_center_shift(mcc: ProfileData, csv_profile: ProfileData) -> float:
    return float(mcc.metrics.normalization_position_mm - csv_profile.metrics.normalization_position_mm)


def best_fit_shift(
    mcc: ProfileData,
    csv_profile: ProfileData,
    initial_shift_mm: float,
    grid_step_mm: float,
    search_window_mm: float,
    search_step_mm: float,
) -> float:
    """Refine a lateral shift by minimizing normalized profile RMS above 20% dose."""
    best_shift = initial_shift_mm
    best_score = math.inf
    offsets = np.arange(-search_window_mm, search_window_mm + 0.5 * search_step_mm, search_step_mm)

    for offset in offsets:
        shift = initial_shift_mm + float(offset)
        try:
            grid = make_common_grid(mcc.x_mm, csv_profile.x_mm, shift, grid_step_mm)
        except ValueError:
            continue
        mcc_rel = interpolate_shifted(grid, mcc.x_mm, mcc.relative)
        csv_rel = interpolate_shifted(grid, csv_profile.x_mm, csv_profile.relative, shift)
        mask = (mcc_rel >= 0.2) | (csv_rel >= 0.2)
        if np.count_nonzero(mask) < 5:
            continue
        diff = csv_rel[mask] - mcc_rel[mask]
        score = float(np.sqrt(np.mean(diff * diff)))
        if score < best_score:
            best_score = score
            best_shift = shift

    return best_shift


def compare_pair(
    name: str,
    csv_file: Path,
    mcc_file: Path,
    output_root: Path,
    grid_step_mm: float,
    alignment_method: str,
    best_fit_window_mm: float,
    best_fit_step_mm: float,
) -> ComparisonResult:
    mcc = read_mcc_central_crossplane(mcc_file)
    csv_profile = read_csv_profile(csv_file)

    fc_shift = field_center_shift(mcc, csv_profile)
    if alignment_method == "none":
        csv_shift = 0.0
    elif alignment_method == "field-center":
        csv_shift = fc_shift
    elif alignment_method == "best-fit":
        csv_shift = best_fit_shift(
            mcc,
            csv_profile,
            fc_shift,
            grid_step_mm,
            best_fit_window_mm,
            best_fit_step_mm,
        )
    else:
        raise ValueError(f"Unsupported alignment method: {alignment_method}")

    grid = make_common_grid(mcc.x_mm, csv_profile.x_mm, csv_shift, grid_step_mm)
    mcc_dose = interpolate_shifted(grid, mcc.x_mm, mcc.dose_gy)
    mcc_rel = interpolate_shifted(grid, mcc.x_mm, mcc.relative)
    csv_dose = interpolate_shifted(grid, csv_profile.x_mm, csv_profile.dose_gy, csv_shift)
    csv_rel = interpolate_shifted(grid, csv_profile.x_mm, csv_profile.relative, csv_shift)
    diff_pp = 100.0 * (csv_rel - mcc_rel)

    profile_mask = (mcc_rel >= 0.1) | (csv_rel >= 0.1)
    high_dose_mask = (mcc_rel >= 0.8) | (csv_rel >= 0.8)
    penumbra_mask = ((mcc_rel >= 0.2) & (mcc_rel <= 0.8)) | ((csv_rel >= 0.2) & (csv_rel <= 0.8))

    stats: Dict[str, Optional[float]] = {}
    for prefix, mask in [
        ("profile_ge10", profile_mask),
        ("high_dose_ge80", high_dose_mask),
        ("penumbra_20_80", penumbra_mask),
    ]:
        for key, value in profile_stats(diff_pp, mask).items():
            stats[f"{prefix}_{key}"] = value

    stats.update(
        {
            "csv_to_mcc_norm_dose_ratio": float(
                csv_profile.metrics.normalization_dose_gy / mcc.metrics.normalization_dose_gy
            ),
            "fwhm_diff_mm": width_difference(csv_profile.metrics.fwhm_mm, mcc.metrics.fwhm_mm),
            "left_penumbra_diff_mm": width_difference(
                csv_profile.metrics.left_penumbra_mm,
                mcc.metrics.left_penumbra_mm,
            ),
            "right_penumbra_diff_mm": width_difference(
                csv_profile.metrics.right_penumbra_mm,
                mcc.metrics.right_penumbra_mm,
            ),
            "left_50_position_diff_mm": metric_difference(
                csv_profile.metrics.left_50_mm,
                mcc.metrics.left_50_mm,
                csv_shift,
            ),
            "right_50_position_diff_mm": metric_difference(
                csv_profile.metrics.right_50_mm,
                mcc.metrics.right_50_mm,
                csv_shift,
            ),
            "left_80_position_diff_mm": metric_difference(
                csv_profile.metrics.left_80_mm,
                mcc.metrics.left_80_mm,
                csv_shift,
            ),
            "right_80_position_diff_mm": metric_difference(
                csv_profile.metrics.right_80_mm,
                mcc.metrics.right_80_mm,
                csv_shift,
            ),
        }
    )

    point_rows: List[Dict[str, object]] = []
    for x, md, cd, mr, cr, diff in zip(grid, mcc_dose, csv_dose, mcc_rel, csv_rel, diff_pp):
        point_rows.append(
            {
                "x_aligned_mm": float(x),
                "mcc_dose_gy": float(md),
                "csv_dose_gy": float(cd),
                "mcc_relative": float(mr),
                "csv_relative": float(cr),
                "difference_percent_points": float(diff),
                "abs_difference_percent_points": abs(float(diff)),
                "region": classify_region(float(mr), float(cr)),
            }
        )

    output_dir = output_root / safe_filename(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = ComparisonResult(
        name=name,
        csv_file=csv_file,
        mcc_file=mcc_file,
        output_dir=output_dir,
        alignment_method=alignment_method,
        csv_shift_mm=csv_shift,
        field_center_shift_mm=fc_shift,
        common_x_min_mm=float(grid[0]),
        common_x_max_mm=float(grid[-1]),
        n_points=int(grid.size),
        mcc=mcc,
        csv_profile=csv_profile,
        point_rows=point_rows,
        stats=stats,
    )

    write_pair_outputs(result)
    return result


def safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def write_dict_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def metric_summary_rows(result: ComparisonResult) -> List[Tuple[str, object, object, object]]:
    mcc = result.mcc.metrics
    csv_metrics = result.csv_profile.metrics
    shift = result.csv_shift_mm
    return [
        ("normalization_position_mm", mcc.normalization_position_mm, shifted_metric(csv_metrics.normalization_position_mm, shift), metric_difference(csv_metrics.normalization_position_mm, mcc.normalization_position_mm, shift)),
        ("normalization_dose_gy", mcc.normalization_dose_gy, csv_metrics.normalization_dose_gy, csv_metrics.normalization_dose_gy - mcc.normalization_dose_gy),
        ("peak_position_mm", mcc.peak_position_mm, shifted_metric(csv_metrics.peak_position_mm, shift), metric_difference(csv_metrics.peak_position_mm, mcc.peak_position_mm, shift)),
        ("peak_relative_dose", mcc.peak_relative_dose, csv_metrics.peak_relative_dose, csv_metrics.peak_relative_dose - mcc.peak_relative_dose),
        ("fwhm_mm", mcc.fwhm_mm, csv_metrics.fwhm_mm, width_difference(csv_metrics.fwhm_mm, mcc.fwhm_mm)),
        ("left_50_mm", mcc.left_50_mm, shifted_metric(csv_metrics.left_50_mm, shift), metric_difference(csv_metrics.left_50_mm, mcc.left_50_mm, shift)),
        ("right_50_mm", mcc.right_50_mm, shifted_metric(csv_metrics.right_50_mm, shift), metric_difference(csv_metrics.right_50_mm, mcc.right_50_mm, shift)),
        ("left_penumbra_mm", mcc.left_penumbra_mm, csv_metrics.left_penumbra_mm, width_difference(csv_metrics.left_penumbra_mm, mcc.left_penumbra_mm)),
        ("right_penumbra_mm", mcc.right_penumbra_mm, csv_metrics.right_penumbra_mm, width_difference(csv_metrics.right_penumbra_mm, mcc.right_penumbra_mm)),
        ("flatness_percent", mcc.flatness_percent, csv_metrics.flatness_percent, width_difference(csv_metrics.flatness_percent, mcc.flatness_percent)),
        ("symmetry_percent", mcc.symmetry_percent, csv_metrics.symmetry_percent, width_difference(csv_metrics.symmetry_percent, mcc.symmetry_percent)),
    ]


def write_pair_outputs(result: ComparisonResult) -> None:
    point_fields = [
        "x_aligned_mm",
        "mcc_dose_gy",
        "csv_dose_gy",
        "mcc_relative",
        "csv_relative",
        "difference_percent_points",
        "abs_difference_percent_points",
        "region",
    ]
    write_dict_rows(result.output_dir / "point_by_point_comparison.csv", result.point_rows, point_fields)

    metric_rows = [
        {
            "metric": metric,
            "mcc": mcc_value,
            "csv_aligned_or_width": csv_value_aligned,
            "csv_minus_mcc": difference,
        }
        for metric, mcc_value, csv_value_aligned, difference in metric_summary_rows(result)
    ]
    write_dict_rows(
        result.output_dir / "metric_comparison.csv",
        metric_rows,
        ["metric", "mcc", "csv_aligned_or_width", "csv_minus_mcc"],
    )

    write_pair_report(result.output_dir / "comparison_report.txt", result)
    write_comparison_svg(result.output_dir / "comparison.svg", result)


def write_pair_report(path: Path, result: ComparisonResult) -> None:
    lines = [
        f"Profile pair: {result.name}\n",
        f"MCC file: {result.mcc_file.name}\n",
        f"CSV file: {result.csv_file.name}\n",
        f"MCC central in-plane y: {format_optional(result.mcc.central_y_mm, 3)} mm\n",
        "\nNormalization\n",
        "  Each profile is normalized to its own dose at the midpoint between the left and right 80% crossings.\n",
        f"  MCC normalization: x = {result.mcc.metrics.normalization_position_mm:.4f} mm, dose = {result.mcc.metrics.normalization_dose_gy:.8g} Gy\n",
        f"  CSV normalization: x = {result.csv_profile.metrics.normalization_position_mm:.4f} mm, dose = {result.csv_profile.metrics.normalization_dose_gy:.8g} Gy\n",
        f"  CSV/MCC normalization dose ratio: {result.stats['csv_to_mcc_norm_dose_ratio']:.6g}\n",
        "\nAlignment\n",
        f"  Method: {result.alignment_method}\n",
        f"  Field-center shift from 80% midpoint: {result.field_center_shift_mm:.4f} mm\n",
        f"  Applied CSV x shift: {result.csv_shift_mm:.4f} mm\n",
        f"  Aligned x convention: x_aligned = csv_x + {result.csv_shift_mm:.4f} mm\n",
        f"  Common comparison range: {result.common_x_min_mm:.3f} to {result.common_x_max_mm:.3f} mm ({result.n_points} points)\n",
        "\nPoint-by-point normalized-dose differences\n",
    ]

    for prefix, label in [
        ("profile_ge10", ">=10% profile"),
        ("high_dose_ge80", ">=80% high-dose region"),
        ("penumbra_20_80", "20%-80% penumbra/shoulder region"),
    ]:
        lines.append(f"  {label}:\n")
        lines.append(f"    Mean CSV-MCC: {format_optional(result.stats[prefix + '_mean_diff_percent_points'], 4)} percentage points\n")
        lines.append(f"    Mean abs: {format_optional(result.stats[prefix + '_mean_abs_diff_percent_points'], 4)} percentage points\n")
        lines.append(f"    RMS: {format_optional(result.stats[prefix + '_rms_diff_percent_points'], 4)} percentage points\n")
        lines.append(f"    Max abs: {format_optional(result.stats[prefix + '_max_abs_diff_percent_points'], 4)} percentage points\n")

    lines.extend(
        [
            "\nPenumbra and edge metrics, CSV minus MCC\n",
            f"  FWHM difference: {format_optional(result.stats['fwhm_diff_mm'], 4)} mm\n",
            f"  Left 80%-20% penumbra difference: {format_optional(result.stats['left_penumbra_diff_mm'], 4)} mm\n",
            f"  Right 80%-20% penumbra difference: {format_optional(result.stats['right_penumbra_diff_mm'], 4)} mm\n",
            f"  Left 50% crossing position difference after alignment: {format_optional(result.stats['left_50_position_diff_mm'], 4)} mm\n",
            f"  Right 50% crossing position difference after alignment: {format_optional(result.stats['right_50_position_diff_mm'], 4)} mm\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


def polyline(points: Iterable[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_comparison_svg(path: Path, result: ComparisonResult) -> None:
    rows = result.point_rows
    x = np.asarray([float(row["x_aligned_mm"]) for row in rows])
    mcc = np.asarray([float(row["mcc_relative"]) for row in rows])
    csv_rel = np.asarray([float(row["csv_relative"]) for row in rows])
    diff = np.asarray([float(row["difference_percent_points"]) for row in rows])

    width = 980
    height = 660
    left = 78
    right = 28
    top = 58
    dose_h = 385
    gap = 48
    diff_h = 115
    dose_bottom = top + dose_h
    diff_top = dose_bottom + gap
    plot_w = width - left - right
    xmin = float(x[0])
    xmax = float(x[-1])
    ymax = max(1.05, float(max(np.max(mcc), np.max(csv_rel))) * 1.08)
    diff_abs = max(1.0, float(np.max(np.abs(diff))) * 1.15)

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_w

    def sy_dose(value: float) -> float:
        return top + (ymax - value) / ymax * dose_h

    def sy_diff(value: float) -> float:
        return diff_top + (diff_abs - value) / (2.0 * diff_abs) * diff_h

    mcc_points = polyline((sx(float(xx)), sy_dose(float(yy))) for xx, yy in zip(x, mcc))
    csv_points = polyline((sx(float(xx)), sy_dose(float(yy))) for xx, yy in zip(x, csv_rel))
    diff_points = polyline((sx(float(xx)), sy_diff(float(yy))) for xx, yy in zip(x, diff))

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="Arial" font-size="18" font-weight="700">{escape(result.name)}</text>',
        f'<text x="{left}" y="48" font-family="Arial" font-size="12">CSV shift {result.csv_shift_mm:.3f} mm | RMS >=10% {format_optional(result.stats["profile_ge10_rms_diff_percent_points"], 3)} percentage points</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{dose_h}" fill="#fbfbfb" stroke="#222"/>',
        f'<rect x="{left}" y="{diff_top}" width="{plot_w}" height="{diff_h}" fill="#fbfbfb" stroke="#222"/>',
    ]

    for level in np.linspace(0.0, 1.0, 6):
        y_line = sy_dose(float(level))
        elements.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y_line:.2f}" y2="{y_line:.2f}" stroke="#e1e1e1"/>')
        elements.append(f'<text x="{left - 10}" y="{y_line + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{level:.1f}</text>')

    for tick in np.linspace(xmin, xmax, 9):
        xp = sx(float(tick))
        elements.append(f'<line x1="{xp:.2f}" x2="{xp:.2f}" y1="{top}" y2="{dose_bottom}" stroke="#eeeeee"/>')
        elements.append(f'<line x1="{xp:.2f}" x2="{xp:.2f}" y1="{diff_top}" y2="{diff_top + diff_h}" stroke="#eeeeee"/>')
        elements.append(f'<text x="{xp:.2f}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="11">{tick:.0f}</text>')

    for level in [-diff_abs, 0.0, diff_abs]:
        y_line = sy_diff(level)
        stroke = "#999999" if abs(level) < 1e-9 else "#e1e1e1"
        elements.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y_line:.2f}" y2="{y_line:.2f}" stroke="{stroke}"/>')
        elements.append(f'<text x="{left - 10}" y="{y_line + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{level:.1f}</text>')

    for metric_value, color, dash in [
        (result.mcc.metrics.left_50_mm, "#0b5cad", ""),
        (result.mcc.metrics.right_50_mm, "#0b5cad", ""),
        (shifted_metric(result.csv_profile.metrics.left_50_mm, result.csv_shift_mm), "#c65f00", "6 5"),
        (shifted_metric(result.csv_profile.metrics.right_50_mm, result.csv_shift_mm), "#c65f00", "6 5"),
    ]:
        if metric_value is None or metric_value < xmin or metric_value > xmax:
            continue
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        xp = sx(float(metric_value))
        elements.append(f'<line x1="{xp:.2f}" x2="{xp:.2f}" y1="{top}" y2="{dose_bottom}" stroke="{color}" opacity="0.45"{dash_attr}/>')

    elements.extend(
        [
            f'<polyline fill="none" stroke="#0b5cad" stroke-width="2.2" points="{mcc_points}"/>',
            f'<polyline fill="none" stroke="#c65f00" stroke-width="2.0" points="{csv_points}"/>',
            f'<polyline fill="none" stroke="#555555" stroke-width="1.7" points="{diff_points}"/>',
            f'<text x="{left + 12}" y="{top + 20}" font-family="Arial" font-size="12" fill="#0b5cad">MCC measured</text>',
            f'<text x="{left + 120}" y="{top + 20}" font-family="Arial" font-size="12" fill="#c65f00">RayStation CSV</text>',
            f'<text x="{left}" y="{top + dose_h + 28}" font-family="Arial" font-size="12">CSV-MCC difference in percentage points</text>',
            f'<text x="{left + plot_w / 2}" y="{height - 5}" text-anchor="middle" font-family="Arial" font-size="13">Aligned cross-plane x (mm)</text>',
            f'<text x="18" y="{top + dose_h / 2}" transform="rotate(-90 18 {top + dose_h / 2})" text-anchor="middle" font-family="Arial" font-size="13">Relative dose</text>',
            "</svg>",
        ]
    )

    path.write_text("\n".join(elements), encoding="utf-8")


def pair_files(csv_dir: Path, mcc_dir: Path) -> List[Tuple[str, Path, Path]]:
    csv_by_key = {path.stem.lower(): path for path in sorted(csv_dir.glob("*.csv"))}
    mcc_by_key = {path.stem.lower(): path for path in sorted(mcc_dir.glob("*.mcc"))}
    pairs: List[Tuple[str, Path, Path]] = []
    for key in sorted(set(csv_by_key) & set(mcc_by_key)):
        csv_path = csv_by_key[key]
        mcc_path = mcc_by_key[key]
        name = csv_path.stem if csv_path.stem == mcc_path.stem else mcc_path.stem
        pairs.append((name, csv_path, mcc_path))
    return pairs


def summary_row(result: ComparisonResult) -> Dict[str, object]:
    row: Dict[str, object] = {
        "profile": result.name,
        "csv_file": result.csv_file.name,
        "mcc_file": result.mcc_file.name,
        "alignment_method": result.alignment_method,
        "csv_shift_mm": result.csv_shift_mm,
        "field_center_shift_mm": result.field_center_shift_mm,
        "common_x_min_mm": result.common_x_min_mm,
        "common_x_max_mm": result.common_x_max_mm,
        "n_points": result.n_points,
        "mcc_norm_dose_gy": result.mcc.metrics.normalization_dose_gy,
        "csv_norm_dose_gy": result.csv_profile.metrics.normalization_dose_gy,
        "mcc_fwhm_mm": result.mcc.metrics.fwhm_mm,
        "csv_fwhm_mm": result.csv_profile.metrics.fwhm_mm,
        "mcc_left_penumbra_mm": result.mcc.metrics.left_penumbra_mm,
        "csv_left_penumbra_mm": result.csv_profile.metrics.left_penumbra_mm,
        "mcc_right_penumbra_mm": result.mcc.metrics.right_penumbra_mm,
        "csv_right_penumbra_mm": result.csv_profile.metrics.right_penumbra_mm,
        "comparison_svg": str(result.output_dir / "comparison.svg"),
        "point_csv": str(result.output_dir / "point_by_point_comparison.csv"),
    }
    row.update(result.stats)
    return row


def write_summary(path: Path, results: Sequence[ComparisonResult]) -> None:
    rows = [summary_row(result) for result in results]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    write_dict_rows(path, rows, fieldnames)


def write_index_html(path: Path, results: Sequence[ComparisonResult]) -> None:
    cards = []
    for result in results:
        rel_svg = result.output_dir.relative_to(path.parent) / "comparison.svg"
        rel_report = result.output_dir.relative_to(path.parent) / "comparison_report.txt"
        cards.append(
            "\n".join(
                [
                    '<section class="pair">',
                    f'<h2>{escape(result.name)}</h2>',
                    f'<p>Shift: {result.csv_shift_mm:.3f} mm. RMS >=10%: {format_optional(result.stats["profile_ge10_rms_diff_percent_points"], 3)} percentage points. '
                    f'<a href="{escape(str(rel_report))}">Report</a></p>',
                    f'<img src="{escape(str(rel_svg))}" alt="{escape(result.name)} comparison plot">',
                    "</section>",
                ]
            )
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OCTAVIUS vs RayStation Profile Comparisons</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
    h1 {{ font-size: 22px; }}
    .pair {{ margin: 26px 0 38px; }}
    .pair h2 {{ font-size: 17px; margin-bottom: 4px; }}
    .pair p {{ margin: 0 0 8px; font-size: 13px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>OCTAVIUS MCC vs RayStation CSV Central Crossplane Comparisons</h1>
  <p>Profiles are independently normalized at the midpoint between their 80% crossings. The CSV profile is shifted in x according to the selected alignment method before point-by-point comparison.</p>
  {''.join(cards)}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RayStation CSV central-axis crossplane profiles to paired OCTAVIUS MCC profiles."
    )
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR, help="Directory containing RayStation CSV profiles.")
    parser.add_argument("--mcc-dir", type=Path, default=DEFAULT_MCC_DIR, help="Directory containing paired MCC files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for comparison outputs.")
    parser.add_argument("--grid-step-mm", type=float, default=1.0, help="Spacing for the common comparison grid.")
    parser.add_argument(
        "--alignment",
        choices=["field-center", "best-fit", "none"],
        default="field-center",
        help="How to laterally align CSV to MCC before point-by-point comparison.",
    )
    parser.add_argument("--best-fit-window-mm", type=float, default=5.0, help="Search half-window for --alignment best-fit.")
    parser.add_argument("--best-fit-step-mm", type=float, default=0.1, help="Search step for --alignment best-fit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid_step_mm <= 0:
        raise ValueError("--grid-step-mm must be positive.")
    if args.best_fit_window_mm < 0:
        raise ValueError("--best-fit-window-mm must be non-negative.")
    if args.best_fit_step_mm <= 0:
        raise ValueError("--best-fit-step-mm must be positive.")
    if not args.csv_dir.exists():
        raise FileNotFoundError(f"CSV directory does not exist: {args.csv_dir}")
    if not args.mcc_dir.exists():
        raise FileNotFoundError(f"MCC directory does not exist: {args.mcc_dir}")

    pairs = pair_files(args.csv_dir, args.mcc_dir)
    if not pairs:
        raise FileNotFoundError(f"No paired .csv/.mcc basenames found in {args.csv_dir} and {args.mcc_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: List[ComparisonResult] = []
    print(f"Found {len(pairs)} paired profiles.")
    print(f"Writing comparison outputs to {args.output_dir}")

    for name, csv_file, mcc_file in pairs:
        print(f"Comparing {name}")
        results.append(
            compare_pair(
                name=name,
                csv_file=csv_file,
                mcc_file=mcc_file,
                output_root=args.output_dir,
                grid_step_mm=args.grid_step_mm,
                alignment_method=args.alignment,
                best_fit_window_mm=args.best_fit_window_mm,
                best_fit_step_mm=args.best_fit_step_mm,
            )
        )

    write_summary(args.output_dir / "summary.csv", results)
    write_index_html(args.output_dir / "index.html", results)
    print(f"Done. Summary: {args.output_dir / 'summary.csv'}")
    print(f"Done. HTML index: {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
