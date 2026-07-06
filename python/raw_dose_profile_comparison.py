from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import numpy as np

from profile_comparison import (
    DEFAULT_CSV_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MCC_DIR,
    best_fit_shift,
    field_center_shift,
    format_optional,
    interpolate_shifted,
    make_common_grid,
    pair_files,
    read_csv_profile,
    read_mcc_central_crossplane,
    safe_filename,
    shifted_metric,
)


DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "raw_gy_comparison_results"


@dataclass
class RawComparisonResult:
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
    point_rows: List[Dict[str, object]]
    stats: Dict[str, Optional[float]]
    mcc_norm_dose_gy: float
    csv_norm_dose_gy: float
    mcc_fwhm_mm: Optional[float]
    csv_fwhm_mm: Optional[float]
    mcc_left_penumbra_mm: Optional[float]
    csv_left_penumbra_mm: Optional[float]
    mcc_right_penumbra_mm: Optional[float]
    csv_right_penumbra_mm: Optional[float]
    mcc_norm_x_mm: float
    csv_norm_x_aligned_mm: float


def csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_dict_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def profile_stats(diff_gy: np.ndarray, percent_diff: np.ndarray, ratio: np.ndarray, mask: np.ndarray) -> Dict[str, Optional[float]]:
    if not np.any(mask):
        return {
            "mean_diff_gy": None,
            "mean_abs_diff_gy": None,
            "rms_diff_gy": None,
            "max_abs_diff_gy": None,
            "mean_percent_diff_of_mcc": None,
            "mean_abs_percent_diff_of_mcc": None,
            "mean_csv_to_mcc_ratio": None,
        }

    diff_values = diff_gy[mask]
    percent_values = percent_diff[mask]
    ratio_values = ratio[mask]
    finite_percent = percent_values[np.isfinite(percent_values)]
    finite_ratio = ratio_values[np.isfinite(ratio_values)]

    return {
        "mean_diff_gy": float(np.mean(diff_values)),
        "mean_abs_diff_gy": float(np.mean(np.abs(diff_values))),
        "rms_diff_gy": float(np.sqrt(np.mean(diff_values * diff_values))),
        "max_abs_diff_gy": float(np.max(np.abs(diff_values))),
        "mean_percent_diff_of_mcc": float(np.mean(finite_percent)) if finite_percent.size else None,
        "mean_abs_percent_diff_of_mcc": float(np.mean(np.abs(finite_percent))) if finite_percent.size else None,
        "mean_csv_to_mcc_ratio": float(np.mean(finite_ratio)) if finite_ratio.size else None,
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


def compare_raw_pair(
    name: str,
    csv_file: Path,
    mcc_file: Path,
    output_root: Path,
    grid_step_mm: float,
    alignment_method: str,
    best_fit_window_mm: float,
    best_fit_step_mm: float,
) -> RawComparisonResult:
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
    csv_dose = interpolate_shifted(grid, csv_profile.x_mm, csv_profile.dose_gy, csv_shift)
    mcc_rel = interpolate_shifted(grid, mcc.x_mm, mcc.relative)
    csv_rel = interpolate_shifted(grid, csv_profile.x_mm, csv_profile.relative, csv_shift)

    diff_gy = csv_dose - mcc_dose
    with np.errstate(divide="ignore", invalid="ignore"):
        percent_diff = 100.0 * diff_gy / mcc_dose
        ratio = csv_dose / mcc_dose

    profile_mask = (mcc_rel >= 0.1) | (csv_rel >= 0.1)
    high_dose_mask = (mcc_rel >= 0.8) | (csv_rel >= 0.8)
    penumbra_mask = ((mcc_rel >= 0.2) & (mcc_rel <= 0.8)) | ((csv_rel >= 0.2) & (csv_rel <= 0.8))

    stats: Dict[str, Optional[float]] = {}
    for prefix, mask in [
        ("profile_ge10", profile_mask),
        ("high_dose_ge80", high_dose_mask),
        ("penumbra_20_80", penumbra_mask),
    ]:
        for key, value in profile_stats(diff_gy, percent_diff, ratio, mask).items():
            stats[f"{prefix}_{key}"] = value

    stats.update(
        {
            "csv_to_mcc_norm_dose_ratio": float(
                csv_profile.metrics.normalization_dose_gy / mcc.metrics.normalization_dose_gy
            ),
            "norm_dose_diff_gy": float(
                csv_profile.metrics.normalization_dose_gy - mcc.metrics.normalization_dose_gy
            ),
            "norm_dose_percent_diff_of_mcc": float(
                100.0
                * (csv_profile.metrics.normalization_dose_gy - mcc.metrics.normalization_dose_gy)
                / mcc.metrics.normalization_dose_gy
            ),
            "fwhm_diff_mm": optional_difference(csv_profile.metrics.fwhm_mm, mcc.metrics.fwhm_mm),
            "left_penumbra_diff_mm": optional_difference(
                csv_profile.metrics.left_penumbra_mm,
                mcc.metrics.left_penumbra_mm,
            ),
            "right_penumbra_diff_mm": optional_difference(
                csv_profile.metrics.right_penumbra_mm,
                mcc.metrics.right_penumbra_mm,
            ),
        }
    )

    point_rows: List[Dict[str, object]] = []
    for x, md, cd, mr, cr, diff, pct, local_ratio in zip(
        grid,
        mcc_dose,
        csv_dose,
        mcc_rel,
        csv_rel,
        diff_gy,
        percent_diff,
        ratio,
    ):
        point_rows.append(
            {
                "x_aligned_mm": float(x),
                "mcc_dose_gy": float(md),
                "csv_dose_gy": float(cd),
                "dose_difference_gy": float(diff),
                "dose_difference_percent_of_mcc": float(pct) if np.isfinite(pct) else None,
                "dose_ratio_csv_to_mcc": float(local_ratio) if np.isfinite(local_ratio) else None,
                "mcc_relative": float(mr),
                "csv_relative": float(cr),
                "region": classify_region(float(mr), float(cr)),
            }
        )

    output_dir = output_root / safe_filename(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = RawComparisonResult(
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
        point_rows=point_rows,
        stats=stats,
        mcc_norm_dose_gy=float(mcc.metrics.normalization_dose_gy),
        csv_norm_dose_gy=float(csv_profile.metrics.normalization_dose_gy),
        mcc_fwhm_mm=mcc.metrics.fwhm_mm,
        csv_fwhm_mm=csv_profile.metrics.fwhm_mm,
        mcc_left_penumbra_mm=mcc.metrics.left_penumbra_mm,
        csv_left_penumbra_mm=csv_profile.metrics.left_penumbra_mm,
        mcc_right_penumbra_mm=mcc.metrics.right_penumbra_mm,
        csv_right_penumbra_mm=csv_profile.metrics.right_penumbra_mm,
        mcc_norm_x_mm=float(mcc.metrics.normalization_position_mm),
        csv_norm_x_aligned_mm=float(shifted_metric(csv_profile.metrics.normalization_position_mm, csv_shift)),
    )

    write_raw_pair_outputs(result)
    return result


def optional_difference(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def write_raw_pair_outputs(result: RawComparisonResult) -> None:
    point_fields = [
        "x_aligned_mm",
        "mcc_dose_gy",
        "csv_dose_gy",
        "dose_difference_gy",
        "dose_difference_percent_of_mcc",
        "dose_ratio_csv_to_mcc",
        "mcc_relative",
        "csv_relative",
        "region",
    ]
    write_dict_rows(result.output_dir / "point_by_point_raw_gy_comparison.csv", result.point_rows, point_fields)

    metric_rows = [
        {
            "metric": "normalization_dose_gy",
            "mcc": result.mcc_norm_dose_gy,
            "csv": result.csv_norm_dose_gy,
            "csv_minus_mcc": result.stats["norm_dose_diff_gy"],
        },
        {
            "metric": "normalization_position_mm_after_alignment",
            "mcc": result.mcc_norm_x_mm,
            "csv": result.csv_norm_x_aligned_mm,
            "csv_minus_mcc": result.csv_norm_x_aligned_mm - result.mcc_norm_x_mm,
        },
        {
            "metric": "fwhm_mm",
            "mcc": result.mcc_fwhm_mm,
            "csv": result.csv_fwhm_mm,
            "csv_minus_mcc": result.stats["fwhm_diff_mm"],
        },
        {
            "metric": "left_80_20_penumbra_mm",
            "mcc": result.mcc_left_penumbra_mm,
            "csv": result.csv_left_penumbra_mm,
            "csv_minus_mcc": result.stats["left_penumbra_diff_mm"],
        },
        {
            "metric": "right_80_20_penumbra_mm",
            "mcc": result.mcc_right_penumbra_mm,
            "csv": result.csv_right_penumbra_mm,
            "csv_minus_mcc": result.stats["right_penumbra_diff_mm"],
        },
    ]
    write_dict_rows(
        result.output_dir / "raw_metric_comparison.csv",
        metric_rows,
        ["metric", "mcc", "csv", "csv_minus_mcc"],
    )

    write_raw_pair_report(result.output_dir / "raw_gy_comparison_report.txt", result)
    write_raw_svg(result.output_dir / "raw_gy_comparison.svg", result)


def write_raw_pair_report(path: Path, result: RawComparisonResult) -> None:
    lines = [
        f"Raw Gy profile pair: {result.name}\n",
        f"MCC file: {result.mcc_file.name}\n",
        f"CSV file: {result.csv_file.name}\n",
        "\nAlignment\n",
        f"  Method: {result.alignment_method}\n",
        f"  Field-center shift from normalized 80% midpoint: {result.field_center_shift_mm:.4f} mm\n",
        f"  Applied CSV x shift: {result.csv_shift_mm:.4f} mm\n",
        f"  Aligned x convention: x_aligned = csv_x + {result.csv_shift_mm:.4f} mm\n",
        f"  Common comparison range: {result.common_x_min_mm:.3f} to {result.common_x_max_mm:.3f} mm ({result.n_points} points)\n",
        "\nAbsolute dose at 80%-midpoint normalization point\n",
        f"  MCC: {result.mcc_norm_dose_gy:.8g} Gy at x = {result.mcc_norm_x_mm:.4f} mm\n",
        f"  CSV: {result.csv_norm_dose_gy:.8g} Gy at aligned x = {result.csv_norm_x_aligned_mm:.4f} mm\n",
        f"  CSV-MCC: {result.stats['norm_dose_diff_gy']:.8g} Gy ({result.stats['norm_dose_percent_diff_of_mcc']:.4f}% of MCC)\n",
        f"  CSV/MCC ratio: {result.stats['csv_to_mcc_norm_dose_ratio']:.6g}\n",
        "\nPoint-by-point raw-dose differences\n",
    ]

    for prefix, label in [
        ("profile_ge10", ">=10% profile"),
        ("high_dose_ge80", ">=80% high-dose region"),
        ("penumbra_20_80", "20%-80% penumbra/shoulder region"),
    ]:
        lines.append(f"  {label}:\n")
        lines.append(f"    Mean CSV-MCC: {format_optional(result.stats[prefix + '_mean_diff_gy'], 6)} Gy\n")
        lines.append(f"    Mean abs: {format_optional(result.stats[prefix + '_mean_abs_diff_gy'], 6)} Gy\n")
        lines.append(f"    RMS: {format_optional(result.stats[prefix + '_rms_diff_gy'], 6)} Gy\n")
        lines.append(f"    Max abs: {format_optional(result.stats[prefix + '_max_abs_diff_gy'], 6)} Gy\n")
        lines.append(f"    Mean percent diff of MCC: {format_optional(result.stats[prefix + '_mean_percent_diff_of_mcc'], 4)}%\n")
        lines.append(f"    Mean CSV/MCC ratio: {format_optional(result.stats[prefix + '_mean_csv_to_mcc_ratio'], 6)}\n")

    lines.extend(
        [
            "\nShape metrics retained for context, CSV minus MCC\n",
            f"  FWHM difference: {format_optional(result.stats['fwhm_diff_mm'], 4)} mm\n",
            f"  Left 80%-20% penumbra difference: {format_optional(result.stats['left_penumbra_diff_mm'], 4)} mm\n",
            f"  Right 80%-20% penumbra difference: {format_optional(result.stats['right_penumbra_diff_mm'], 4)} mm\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


def polyline(points: Iterable[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_raw_svg(path: Path, result: RawComparisonResult) -> None:
    rows = result.point_rows
    x = np.asarray([float(row["x_aligned_mm"]) for row in rows])
    mcc = np.asarray([float(row["mcc_dose_gy"]) for row in rows])
    csv_dose = np.asarray([float(row["csv_dose_gy"]) for row in rows])
    diff = np.asarray([float(row["dose_difference_gy"]) for row in rows])

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
    ymax = max(2.05, float(max(np.max(mcc), np.max(csv_dose))) * 1.08)
    diff_abs = max(0.02, float(np.max(np.abs(diff))) * 1.15)

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_w

    def sy_dose(value: float) -> float:
        return top + (ymax - value) / ymax * dose_h

    def sy_diff(value: float) -> float:
        return diff_top + (diff_abs - value) / (2.0 * diff_abs) * diff_h

    mcc_points = polyline((sx(float(xx)), sy_dose(float(yy))) for xx, yy in zip(x, mcc))
    csv_points = polyline((sx(float(xx)), sy_dose(float(yy))) for xx, yy in zip(x, csv_dose))
    diff_points = polyline((sx(float(xx)), sy_diff(float(yy))) for xx, yy in zip(x, diff))

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="Arial" font-size="18" font-weight="700">{escape(result.name)} raw Gy</text>',
        f'<text x="{left}" y="48" font-family="Arial" font-size="12">CSV shift {result.csv_shift_mm:.3f} mm | high-dose mean ratio {format_optional(result.stats["high_dose_ge80_mean_csv_to_mcc_ratio"], 4)} | RMS >=10% {format_optional(result.stats["profile_ge10_rms_diff_gy"], 5)} Gy</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{dose_h}" fill="#fbfbfb" stroke="#222"/>',
        f'<rect x="{left}" y="{diff_top}" width="{plot_w}" height="{diff_h}" fill="#fbfbfb" stroke="#222"/>',
    ]

    for level in np.linspace(0.0, ymax, 6):
        y_line = sy_dose(float(level))
        elements.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y_line:.2f}" y2="{y_line:.2f}" stroke="#e1e1e1"/>')
        elements.append(f'<text x="{left - 10}" y="{y_line + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{level:.2f}</text>')

    for tick in np.linspace(xmin, xmax, 9):
        xp = sx(float(tick))
        elements.append(f'<line x1="{xp:.2f}" x2="{xp:.2f}" y1="{top}" y2="{dose_bottom}" stroke="#eeeeee"/>')
        elements.append(f'<line x1="{xp:.2f}" x2="{xp:.2f}" y1="{diff_top}" y2="{diff_top + diff_h}" stroke="#eeeeee"/>')
        elements.append(f'<text x="{xp:.2f}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="11">{tick:.0f}</text>')

    for level in [-diff_abs, 0.0, diff_abs]:
        y_line = sy_diff(level)
        stroke = "#999999" if abs(level) < 1e-12 else "#e1e1e1"
        elements.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y_line:.2f}" y2="{y_line:.2f}" stroke="{stroke}"/>')
        elements.append(f'<text x="{left - 10}" y="{y_line + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{level:.3f}</text>')

    elements.extend(
        [
            f'<polyline fill="none" stroke="#0b5cad" stroke-width="2.2" points="{mcc_points}"/>',
            f'<polyline fill="none" stroke="#c65f00" stroke-width="2.0" points="{csv_points}"/>',
            f'<polyline fill="none" stroke="#555555" stroke-width="1.7" points="{diff_points}"/>',
            f'<text x="{left + 12}" y="{top + 20}" font-family="Arial" font-size="12" fill="#0b5cad">MCC measured Gy</text>',
            f'<text x="{left + 140}" y="{top + 20}" font-family="Arial" font-size="12" fill="#c65f00">RayStation CSV Gy</text>',
            f'<text x="{left}" y="{top + dose_h + 28}" font-family="Arial" font-size="12">CSV-MCC raw-dose difference (Gy)</text>',
            f'<text x="{left + plot_w / 2}" y="{height - 5}" text-anchor="middle" font-family="Arial" font-size="13">Aligned cross-plane x (mm)</text>',
            f'<text x="18" y="{top + dose_h / 2}" transform="rotate(-90 18 {top + dose_h / 2})" text-anchor="middle" font-family="Arial" font-size="13">Dose (Gy)</text>',
            "</svg>",
        ]
    )

    path.write_text("\n".join(elements), encoding="utf-8")


def summary_row(result: RawComparisonResult) -> Dict[str, object]:
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
        "mcc_norm_dose_gy": result.mcc_norm_dose_gy,
        "csv_norm_dose_gy": result.csv_norm_dose_gy,
        "norm_dose_diff_gy": result.stats["norm_dose_diff_gy"],
        "norm_dose_percent_diff_of_mcc": result.stats["norm_dose_percent_diff_of_mcc"],
        "mcc_fwhm_mm": result.mcc_fwhm_mm,
        "csv_fwhm_mm": result.csv_fwhm_mm,
        "mcc_left_penumbra_mm": result.mcc_left_penumbra_mm,
        "csv_left_penumbra_mm": result.csv_left_penumbra_mm,
        "mcc_right_penumbra_mm": result.mcc_right_penumbra_mm,
        "csv_right_penumbra_mm": result.csv_right_penumbra_mm,
        "raw_svg": str(result.output_dir / "raw_gy_comparison.svg"),
        "point_csv": str(result.output_dir / "point_by_point_raw_gy_comparison.csv"),
    }
    row.update(result.stats)
    return row


def write_summary(path: Path, results: Sequence[RawComparisonResult]) -> None:
    rows = [summary_row(result) for result in results]
    if not rows:
        return
    write_dict_rows(path, rows, list(rows[0].keys()))


def write_index_html(path: Path, results: Sequence[RawComparisonResult]) -> None:
    cards = []
    for result in results:
        rel_svg = result.output_dir.relative_to(path.parent) / "raw_gy_comparison.svg"
        rel_report = result.output_dir.relative_to(path.parent) / "raw_gy_comparison_report.txt"
        cards.append(
            "\n".join(
                [
                    '<section class="pair">',
                    f'<h2>{escape(result.name)}</h2>',
                    f'<p>High-dose mean CSV/MCC ratio: {format_optional(result.stats["high_dose_ge80_mean_csv_to_mcc_ratio"], 4)}. '
                    f'Raw RMS >=10%: {format_optional(result.stats["profile_ge10_rms_diff_gy"], 5)} Gy. '
                    f'<a href="{escape(str(rel_report))}">Report</a></p>',
                    f'<img src="{escape(str(rel_svg))}" alt="{escape(result.name)} raw Gy comparison plot">',
                    "</section>",
                ]
            )
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Raw Gy OCTAVIUS vs RayStation Profile Comparisons</title>
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
  <h1>Raw Gy OCTAVIUS MCC vs RayStation CSV Central Crossplane Comparisons</h1>
  <p>Profiles are compared in absolute Gy with no dose normalization or rescaling. The CSV profile is shifted in x using the selected alignment method, then raw dose differences are reported directly.</p>
  {''.join(cards)}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RayStation CSV and OCTAVIUS MCC central-axis crossplane profiles in raw Gy."
    )
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR, help="Directory containing RayStation CSV profiles.")
    parser.add_argument("--mcc-dir", type=Path, default=DEFAULT_MCC_DIR, help="Directory containing paired MCC files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for raw Gy comparison outputs.")
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
    results: List[RawComparisonResult] = []
    print(f"Found {len(pairs)} paired profiles.")
    print(f"Writing raw Gy comparison outputs to {args.output_dir}")

    for name, csv_file, mcc_file in pairs:
        print(f"Comparing {name}")
        results.append(
            compare_raw_pair(
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
    print(f"Done. Raw Gy summary: {args.output_dir / 'summary.csv'}")
    print(f"Done. Raw Gy HTML index: {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
