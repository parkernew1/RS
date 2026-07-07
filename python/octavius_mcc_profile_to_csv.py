from __future__ import annotations

import csv
import os
import re
import tempfile
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ModuleNotFoundError:
    plt = None
    PdfPages = None

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    Image = None
    ImageDraw = None


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "Trial Runs" / "Octavius_Raystation_comparison_copper" / "All_MCC"
OUTPUT_DIR = REPO_ROOT / "Trial Runs" / "octavius_1500_copper_crossplane_profiles"

NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass
class Profile:
    """One MCC BEGIN_SCAN block, representing one detector row."""

    scan_number: int
    inplane_position_mm: float
    x_mm: np.ndarray
    dose_gy: np.ndarray


@dataclass
class AxisMetrics:
    """Summary metrics for one 1D central-axis dose profile."""

    axis_name: str
    normalization_position_mm: float
    normalization_dose_gy: float
    peak_relative_dose: float
    peak_position_mm: float
    half_dose_level: float
    fwhm_mm: Optional[float]
    left_50_mm: Optional[float]
    right_50_mm: Optional[float]
    left_20_mm: Optional[float]
    left_80_mm: Optional[float]
    right_80_mm: Optional[float]
    right_20_mm: Optional[float]
    left_penumbra_mm: Optional[float]
    right_penumbra_mm: Optional[float]
    flatness_percent: Optional[float]
    symmetry_percent: Optional[float]
    field_center_mm: Optional[float]
    flat_region_left_mm: Optional[float]
    flat_region_right_mm: Optional[float]


def read_mcc(filepath: Path) -> List[Profile]:
    """Read an MCC file into row profiles using physical coordinates."""
    profiles: List[Profile] = []

    inside_scan = False
    inside_data = False
    current_scan_number: Optional[int] = None
    current_inplane_position: Optional[float] = None
    current_x: List[float] = []
    current_dose: List[float] = []

    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue

            match_scan = re.fullmatch(r"BEGIN_SCAN\s+(\d+)", stripped)
            if match_scan:
                inside_scan = True
                inside_data = False
                current_scan_number = int(match_scan.group(1))
                current_inplane_position = None
                current_x = []
                current_dose = []
                continue

            if not inside_scan:
                continue

            if stripped.startswith("SCAN_OFFAXIS_INPLANE="):
                current_inplane_position = float(stripped.split("=", 1)[1])
                continue

            if stripped == "BEGIN_DATA":
                inside_data = True
                continue

            if stripped == "END_DATA":
                inside_data = False
                continue

            if re.fullmatch(r"END_SCAN\s+\d+", stripped):
                if current_scan_number is None:
                    raise ValueError(f"{filepath.name}: END_SCAN without BEGIN_SCAN.")
                if current_inplane_position is None:
                    raise ValueError(
                        f"{filepath.name}: scan {current_scan_number} is missing "
                        "SCAN_OFFAXIS_INPLANE."
                    )
                if not current_x:
                    raise ValueError(
                        f"{filepath.name}: scan {current_scan_number} has no data."
                    )

                x_array = np.asarray(current_x, dtype=float)
                dose_array = np.asarray(current_dose, dtype=float)
                order = np.argsort(x_array)
                x_array = x_array[order]
                dose_array = dose_array[order]

                if np.unique(x_array).size != x_array.size:
                    raise ValueError(
                        f"{filepath.name}: scan {current_scan_number} has duplicate x values."
                    )

                profiles.append(
                    Profile(
                        scan_number=current_scan_number,
                        inplane_position_mm=float(current_inplane_position),
                        x_mm=x_array,
                        dose_gy=dose_array,
                    )
                )

                inside_scan = False
                inside_data = False
                current_scan_number = None
                current_inplane_position = None
                current_x = []
                current_dose = []
                continue

            if inside_data:
                data_text = stripped.split("#", 1)[0].strip()
                numbers = NUMBER_PATTERN.findall(data_text)
                if len(numbers) >= 2:
                    current_x.append(float(numbers[0]))
                    current_dose.append(float(numbers[1]))

    if not profiles:
        raise ValueError(f"No valid MCC profiles were found in {filepath}.")

    profiles.sort(key=lambda p: p.inplane_position_mm)
    return profiles


def estimate_common_grid_spacing(profiles: List[Profile], decimals: int = 8) -> float:
    """Return the smallest real x spacing present in the detector coordinates."""
    all_positions = np.concatenate([p.x_mm for p in profiles])
    unique_positions = np.unique(np.round(all_positions, decimals=decimals))
    unique_positions.sort()

    differences = np.diff(unique_positions)
    differences = differences[differences > 10 ** (-decimals)]
    if differences.size == 0:
        raise ValueError("Could not infer a common cross-plane grid spacing.")

    return float(np.min(differences))


def build_aligned_dose_matrix(
    profiles: List[Profile],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a rectangular, coordinate-correct dose matrix.

    The OCTAVIUS 1500 rows are staggered: neighboring rows are shifted by half a
    detector pitch in x. Direct detector measurements are preserved. For grid
    points missing from a row because of the stagger, the preferred estimate is
    vertical interpolation between the same x coordinate measured on the nearest
    rows above and below. Same-row horizontal interpolation is used only as an
    edge fallback where a vertical bracket does not exist.
    """
    y_positions = np.asarray([p.inplane_position_mm for p in profiles], dtype=float)

    common_min = max(float(np.min(p.x_mm)) for p in profiles)
    common_max = min(float(np.max(p.x_mm)) for p in profiles)
    if common_max <= common_min:
        raise ValueError("Profiles do not share a common cross-plane range.")

    spacing = estimate_common_grid_spacing(profiles)
    n_steps = int(np.floor((common_max - common_min) / spacing + 1e-9))
    x_positions = common_min + spacing * np.arange(n_steps + 1, dtype=float)
    if common_max - x_positions[-1] > spacing * 1e-6:
        x_positions = np.append(x_positions, common_max)

    dose_matrix = np.full((len(profiles), len(x_positions)), np.nan, dtype=float)
    measured_mask = np.zeros(dose_matrix.shape, dtype=bool)
    interpolation_method = np.full(dose_matrix.shape, "unfilled", dtype="U32")

    measured_by_row: List[Dict[float, float]] = [
        {round(float(x), 8): float(dose) for x, dose in zip(profile.x_mm, profile.dose_gy)}
        for profile in profiles
    ]
    x_keys = [round(float(x), 8) for x in x_positions]

    for row_idx, row_measurements in enumerate(measured_by_row):
        for col_idx, x_key in enumerate(x_keys):
            if x_key in row_measurements:
                dose_matrix[row_idx, col_idx] = row_measurements[x_key]
                measured_mask[row_idx, col_idx] = True
                interpolation_method[row_idx, col_idx] = "direct"

    for row_idx, profile in enumerate(profiles):
        for col_idx, x_key in enumerate(x_keys):
            if np.isfinite(dose_matrix[row_idx, col_idx]):
                continue

            lower_idx = None
            upper_idx = None
            for candidate_idx in range(row_idx - 1, -1, -1):
                if x_key in measured_by_row[candidate_idx]:
                    lower_idx = candidate_idx
                    break
            for candidate_idx in range(row_idx + 1, len(profiles)):
                if x_key in measured_by_row[candidate_idx]:
                    upper_idx = candidate_idx
                    break

            if lower_idx is not None and upper_idx is not None:
                y_lower = y_positions[lower_idx]
                y_upper = y_positions[upper_idx]
                fraction = (y_positions[row_idx] - y_lower) / (y_upper - y_lower)
                lower_dose = measured_by_row[lower_idx][x_key]
                upper_dose = measured_by_row[upper_idx][x_key]
                dose_matrix[row_idx, col_idx] = lower_dose + fraction * (upper_dose - lower_dose)
                interpolation_method[row_idx, col_idx] = "vertical_offset_rows"
                continue

            dose_matrix[row_idx, col_idx] = np.interp(
                x_positions[col_idx],
                profile.x_mm,
                profile.dose_gy,
            )
            interpolation_method[row_idx, col_idx] = "horizontal_edge_fallback"

    return (
        x_positions,
        y_positions,
        dose_matrix,
        measured_mask,
        interpolation_method,
    )


def interpolate_crossing(
    positions: np.ndarray,
    relative_dose: np.ndarray,
    level: float,
    side: str,
) -> Optional[float]:
    """Linearly interpolate the crossing nearest the central axis on one side."""
    if side == "left":
        mask = positions <= 0
    elif side == "right":
        mask = positions >= 0
    else:
        raise ValueError("side must be 'left' or 'right'.")

    xs = positions[mask]
    ys = relative_dose[mask]
    if xs.size < 2:
        return None

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    crossings: List[float] = []

    for i in range(xs.size - 1):
        y1 = ys[i]
        y2 = ys[i + 1]
        if (y1 - level) * (y2 - level) > 0:
            continue
        if np.isclose(y1, y2):
            if np.isclose(y1, level):
                crossings.append(float(0.5 * (xs[i] + xs[i + 1])))
            continue
        fraction = (level - y1) / (y2 - y1)
        if 0 <= fraction <= 1:
            crossings.append(float(xs[i] + fraction * (xs[i + 1] - xs[i])))

    if not crossings:
        return None
    return min(crossings, key=abs)


def all_level_crossings(
    positions: np.ndarray,
    relative_dose: np.ndarray,
    level: float,
) -> List[float]:
    """Return all linearly interpolated crossings of a dose level."""
    order = np.argsort(positions)
    xs = positions[order]
    ys = relative_dose[order]
    crossings: List[float] = []

    for i in range(xs.size - 1):
        y1 = ys[i]
        y2 = ys[i + 1]
        if (y1 - level) * (y2 - level) > 0:
            continue
        if np.isclose(y1, y2):
            if np.isclose(y1, level):
                crossings.append(float(0.5 * (xs[i] + xs[i + 1])))
            continue
        fraction = (level - y1) / (y2 - y1)
        if 0 <= fraction <= 1:
            crossings.append(float(xs[i] + fraction * (xs[i + 1] - xs[i])))

    return crossings


def peak_side_crossing(
    positions: np.ndarray,
    relative_dose: np.ndarray,
    level: float,
    peak_position: float,
    side: str,
) -> Optional[float]:
    """
    Return the level crossing belonging to one edge of the profile peak.

    This is intentionally peak-relative rather than central-axis-relative.
    For shielded profiles, there may be extra crossings near x=0. The left
    penumbra should pair the 20% and 80% crossings on the low-x side of the
    peak, and the right penumbra should pair crossings on the high-x side.
    """
    crossings = all_level_crossings(positions, relative_dose, level)
    if side == "left":
        candidates = [x for x in crossings if x <= peak_position]
        if not candidates:
            return None
        return max(candidates)
    if side == "right":
        candidates = [x for x in crossings if x >= peak_position]
        if not candidates:
            return None
        return min(candidates)
    raise ValueError("side must be 'left' or 'right'.")


def analyze_axis_profile(
    axis_name: str,
    positions_mm: np.ndarray,
    dose_gy: np.ndarray,
) -> Tuple[np.ndarray, AxisMetrics]:
    """Normalize and calculate FWHM, flatness, symmetry, and penumbra."""
    peak_index = int(np.argmax(dose_gy))
    peak_position_mm = float(positions_mm[peak_index])
    normalization_position_mm = peak_position_mm
    normalization_dose_gy = float(dose_gy[peak_index])

    if not np.isfinite(normalization_dose_gy) or np.isclose(normalization_dose_gy, 0.0):
        raise ValueError("Profile normalization dose is zero or non-finite.")

    for _ in range(20):
        relative_trial = dose_gy / normalization_dose_gy
        left_80_trial = peak_side_crossing(
            positions_mm,
            relative_trial,
            0.8,
            peak_position_mm,
            "left",
        )
        right_80_trial = peak_side_crossing(
            positions_mm,
            relative_trial,
            0.8,
            peak_position_mm,
            "right",
        )
        if left_80_trial is None or right_80_trial is None:
            break

        next_position = 0.5 * (left_80_trial + right_80_trial)
        next_dose = float(np.interp(next_position, positions_mm, dose_gy))
        if not np.isfinite(next_dose) or np.isclose(next_dose, 0.0):
            break
        if (
            abs(next_position - normalization_position_mm) < 1e-6
            and abs(next_dose - normalization_dose_gy) < 1e-9
        ):
            normalization_position_mm = float(next_position)
            normalization_dose_gy = next_dose
            break

        normalization_position_mm = float(next_position)
        normalization_dose_gy = next_dose

    relative = dose_gy / normalization_dose_gy
    peak_relative_dose = float(np.max(relative))
    half_dose_level = 0.5

    left_50 = peak_side_crossing(
        positions_mm,
        relative,
        half_dose_level,
        peak_position_mm,
        "left",
    )
    right_50 = peak_side_crossing(
        positions_mm,
        relative,
        half_dose_level,
        peak_position_mm,
        "right",
    )
    left_20 = peak_side_crossing(
        positions_mm,
        relative,
        0.2,
        peak_position_mm,
        "left",
    )
    left_80 = peak_side_crossing(
        positions_mm,
        relative,
        0.8,
        peak_position_mm,
        "left",
    )
    right_80 = peak_side_crossing(
        positions_mm,
        relative,
        0.8,
        peak_position_mm,
        "right",
    )
    right_20 = peak_side_crossing(
        positions_mm,
        relative,
        0.2,
        peak_position_mm,
        "right",
    )

    fwhm: Optional[float] = None
    field_center: Optional[float] = None
    flat_left: Optional[float] = None
    flat_right: Optional[float] = None
    flatness: Optional[float] = None
    symmetry: Optional[float] = None

    if left_50 is not None and right_50 is not None:
        fwhm = float(right_50 - left_50)
        field_center = 0.5 * (left_50 + right_50)
        flat_left = field_center - 0.4 * fwhm
        flat_right = field_center + 0.4 * fwhm

        flat_mask = (positions_mm >= flat_left) & (positions_mm <= flat_right)
        flat_values = relative[flat_mask]
        if flat_values.size > 0:
            dmax = float(np.max(flat_values))
            dmin = float(np.min(flat_values))
            if dmax + dmin > 0:
                flatness = 100.0 * (dmax - dmin) / (dmax + dmin)

        pair_differences: List[float] = []
        for i in np.where(flat_mask)[0]:
            mirrored_position = 2.0 * field_center - positions_mm[i]
            mirrored_value = float(np.interp(mirrored_position, positions_mm, relative))
            pair_differences.append(abs(float(relative[i]) - mirrored_value))
        if pair_differences:
            symmetry = 100.0 * max(pair_differences)

    left_penumbra = None
    right_penumbra = None
    if left_20 is not None and left_80 is not None:
        left_penumbra = abs(float(left_80 - left_20))
    if right_20 is not None and right_80 is not None:
        right_penumbra = abs(float(right_20 - right_80))

    metrics = AxisMetrics(
        axis_name=axis_name,
        normalization_position_mm=float(normalization_position_mm),
        normalization_dose_gy=float(normalization_dose_gy),
        peak_relative_dose=peak_relative_dose,
        peak_position_mm=peak_position_mm,
        half_dose_level=float(half_dose_level),
        fwhm_mm=fwhm,
        left_50_mm=left_50,
        right_50_mm=right_50,
        left_20_mm=left_20,
        left_80_mm=left_80,
        right_80_mm=right_80,
        right_20_mm=right_20,
        left_penumbra_mm=left_penumbra,
        right_penumbra_mm=right_penumbra,
        flatness_percent=flatness,
        symmetry_percent=symmetry,
        field_center_mm=field_center,
        flat_region_left_mm=flat_left,
        flat_region_right_mm=flat_right,
    )
    return relative, metrics


def write_matrix_csv(
    path: Path,
    x_positions_mm: np.ndarray,
    y_positions_mm: np.ndarray,
    matrix: np.ndarray,
    value_name: str,
    fmt: str = "%.10g",
) -> None:
    """Write an x-by-y matrix with physical-coordinate headers."""
    header = ["crossplane_x_mm"] + [f"{value_name}_at_y_{y:g}_mm" for y in y_positions_mm]
    output = np.column_stack([x_positions_mm, matrix.T])
    np.savetxt(path, output, delimiter=",", header=",".join(header), comments="", fmt=fmt)


def write_method_matrix_csv(
    path: Path,
    x_positions_mm: np.ndarray,
    y_positions_mm: np.ndarray,
    interpolation_method: np.ndarray,
) -> None:
    """Write how each aligned matrix cell was produced."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["crossplane_x_mm"] + [f"method_at_y_{y:g}_mm" for y in y_positions_mm])
        for col_idx, x_mm in enumerate(x_positions_mm):
            writer.writerow([x_mm] + list(interpolation_method[:, col_idx]))


def write_raw_measurements_csv(path: Path, profiles: List[Profile]) -> None:
    """Write every original detector point before alignment/interpolation."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scan_number", "inplane_y_mm", "crossplane_x_mm", "dose_gy"])
        for profile in profiles:
            for x_mm, dose_gy in zip(profile.x_mm, profile.dose_gy):
                writer.writerow([profile.scan_number, profile.inplane_position_mm, x_mm, dose_gy])


def write_profile_csv(
    path: Path,
    position_header: str,
    positions_mm: np.ndarray,
    dose_gy: np.ndarray,
    relative_dose: np.ndarray,
    measured_mask: np.ndarray,
    interpolation_method: np.ndarray,
) -> None:
    """Write one central-axis profile with direct-measurement flags."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                position_header,
                "dose_gy",
                "relative_to_80_percent_midpoint",
                "percent_of_80_percent_midpoint",
                "directly_measured",
                "interpolation_method",
            ]
        )
        for position, dose, relative, measured, method in zip(
            positions_mm,
            dose_gy,
            relative_dose,
            measured_mask,
            interpolation_method,
        ):
            writer.writerow([position, dose, relative, 100.0 * relative, int(bool(measured)), method])


def write_metrics_report(
    path: Path,
    mcc_file: Path,
    profiles: List[Profile],
    x_positions_mm: np.ndarray,
    y_positions_mm: np.ndarray,
    dose_matrix: np.ndarray,
    measured_mask: np.ndarray,
    central_y_mm: float,
    crossplane: AxisMetrics,
) -> str:
    """Write a text report and return its contents."""
    measured_count = int(measured_mask.sum())
    total_count = int(measured_mask.size)
    global_row, global_col = np.unravel_index(np.argmax(dose_matrix), dose_matrix.shape)

    lines = [
        f"MCC file: {mcc_file.name}\n",
        f"Profiles/rows parsed: {len(profiles)}\n",
        f"Original detector spacing within a row: 10 mm\n",
        f"Aligned x grid: {x_positions_mm[0]:.1f} to {x_positions_mm[-1]:.1f} mm, "
        f"{np.median(np.diff(x_positions_mm)):.1f} mm spacing\n",
        f"Y row grid: {y_positions_mm[0]:.1f} to {y_positions_mm[-1]:.1f} mm, "
        f"{np.median(np.diff(y_positions_mm)):.1f} mm spacing\n",
        f"Directly measured aligned cells: {measured_count} / {total_count} "
        f"({100.0 * measured_count / total_count:.1f}%)\n",
        "\nProfile normalization\n",
        "  100% is set to the interpolated dose at the midpoint between the "
        "left and right 80% crossings.\n",
        f"  Normalization point: x = {crossplane.normalization_position_mm:.3f} mm, "
        f"y = {central_y_mm:.1f} mm\n",
        f"  Normalization dose: {crossplane.normalization_dose_gy:.8g} Gy\n",
        "\n2D dose maximum\n",
        f"  Max dose: {float(np.max(dose_matrix)):.8g} Gy at "
        f"x = {x_positions_mm[global_col]:.1f} mm, y = {y_positions_mm[global_row]:.1f} mm\n",
    ]

    lines.extend(
        [
            f"\n{crossplane.axis_name} profile\n",
            f"  Peak relative dose: {crossplane.peak_relative_dose:.6g} "
            f"at {crossplane.peak_position_mm:.3f} mm\n",
            f"  50% dose level: {crossplane.half_dose_level:.6g}\n",
            f"  50% crossings: {format_optional(crossplane.left_50_mm)} mm, "
            f"{format_optional(crossplane.right_50_mm)} mm\n",
            f"  FWHM: {format_optional(crossplane.fwhm_mm)} mm\n",
            f"  80%-20% penumbra left/right: "
            f"{format_optional(crossplane.left_penumbra_mm)} mm, "
            f"{format_optional(crossplane.right_penumbra_mm)} mm\n",
            f"  Flatness in central 80% of FWHM field: "
            f"{format_optional(crossplane.flatness_percent)} %\n",
            f"  Symmetry in central 80% of FWHM field: "
            f"{format_optional(crossplane.symmetry_percent)} %\n",
        ]
    )

    report_text = "".join(lines)
    path.write_text(report_text, encoding="utf-8")
    return report_text


def format_optional(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def plot_axis_profile(
    path: Path,
    title: str,
    xlabel: str,
    positions_mm: np.ndarray,
    relative_dose: np.ndarray,
    metrics: AxisMetrics,
) -> plt.Figure:
    """Plot one central-axis profile with analysis markers."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(positions_mm, relative_dose, marker="o", markersize=3, linewidth=1.4)
    ax.axhline(metrics.half_dose_level, color="0.35", linestyle="--", linewidth=1, label="50% dose")
    ax.axvline(
        metrics.normalization_position_mm,
        color="#1f7a1f",
        linestyle="-.",
        linewidth=1,
        label="100% normalization point",
    )

    if metrics.flat_region_left_mm is not None and metrics.flat_region_right_mm is not None:
        ax.axvspan(
            metrics.flat_region_left_mm,
            metrics.flat_region_right_mm,
            alpha=0.14,
            label="Central 80% of FWHM field",
        )

    for value, label, linestyle in [
        (metrics.left_50_mm, "50%", "--"),
        (metrics.right_50_mm, "50%", "--"),
        (metrics.left_20_mm, "20%", ":"),
        (metrics.left_80_mm, "80%", ":"),
        (metrics.right_80_mm, "80%", ":"),
        (metrics.right_20_mm, "20%", ":"),
    ]:
        if value is None:
            continue
        ax.axvline(value, color="0.25", linestyle=linestyle, linewidth=0.9)
        ax.text(value, 0.04, label, rotation=90, va="bottom", ha="right", fontsize=7)

    penumbra_specs = [
        (metrics.left_20_mm, metrics.left_80_mm, metrics.left_penumbra_mm, "Left penumbra"),
        (metrics.right_80_mm, metrics.right_20_mm, metrics.right_penumbra_mm, "Right penumbra"),
    ]
    for x_start, x_end, width_mm, label in penumbra_specs:
        if x_start is None or x_end is None or width_mm is None:
            continue
        y_level = 0.8
        ax.annotate(
            "",
            xy=(x_start, y_level),
            xytext=(x_end, y_level),
            arrowprops={"arrowstyle": "<->", "color": "#8a4f00", "linewidth": 1.4},
        )
        ax.text(
            0.5 * (x_start + x_end),
            y_level + 0.04 * max(1.0, metrics.peak_relative_dose),
            f"{label}: {width_mm:.1f} mm",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#8a4f00",
        )

    subtitle = []
    if metrics.fwhm_mm is not None:
        subtitle.append(f"FWHM {metrics.fwhm_mm:.1f} mm")
    if metrics.flatness_percent is not None:
        subtitle.append(f"Flatness {metrics.flatness_percent:.2f}%")
    if metrics.symmetry_percent is not None:
        subtitle.append(f"Symmetry {metrics.symmetry_percent:.2f}%")

    full_title = title
    if subtitle:
        full_title += "\n" + " | ".join(subtitle)
    ax.set_title(full_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Relative dose, normalized at midpoint of 80% crossings")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    return fig


def plot_heatmap(
    path: Path,
    title: str,
    x_positions_mm: np.ndarray,
    y_positions_mm: np.ndarray,
    dose_matrix: np.ndarray,
    central_x_mm: float,
    central_y_mm: float,
) -> plt.Figure:
    """Plot the aligned 2D dose matrix without display smoothing."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    image = ax.imshow(
        dose_matrix,
        extent=[x_positions_mm[0], x_positions_mm[-1], y_positions_mm[0], y_positions_mm[-1]],
        origin="lower",
        interpolation="bicubic",
        aspect="equal",
    )
    ax.axhline(central_y_mm, color="white", linewidth=0.9, linestyle="--")
    ax.axvline(central_x_mm, color="white", linewidth=0.9, linestyle="--")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Dose (Gy)")
    ax.set_title(title)
    ax.set_xlabel("Cross-plane x (mm)")
    ax.set_ylabel("In-plane y (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    return fig


def svg_polyline(points: Iterable[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_axis_profile_svg(
    path: Path,
    title: str,
    xlabel: str,
    positions_mm: np.ndarray,
    relative_dose: np.ndarray,
    metrics: AxisMetrics,
) -> None:
    """Write a dependency-free SVG profile plot."""
    width = 900
    height = 560
    left = 78
    right = 24
    top = 64
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom

    xmin = float(np.min(positions_mm))
    xmax = float(np.max(positions_mm))
    ymax = max(1.05, float(np.max(relative_dose)) * 1.08)

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return top + (ymax - y) / ymax * plot_h

    polyline = svg_polyline((sx(float(x)), sy(float(y))) for x, y in zip(positions_mm, relative_dose))
    title_text = escape(title)
    xlabel_text = escape(xlabel)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="18" font-weight="700">{title_text}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#222" stroke-width="1"/>',
    ]

    for frac in np.linspace(0, 1.0, 6):
        y = sy(float(frac))
        elements.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>')
        elements.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{frac:.1f}</text>')

    for x in np.linspace(xmin, xmax, 7):
        xp = sx(float(x))
        elements.append(f'<line x1="{xp:.2f}" x2="{xp:.2f}" y1="{top}" y2="{top + plot_h}" stroke="#eeeeee" stroke-width="1"/>')
        elements.append(f'<text x="{xp:.2f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{x:.0f}</text>')

    if metrics.flat_region_left_mm is not None and metrics.flat_region_right_mm is not None:
        x1 = sx(metrics.flat_region_left_mm)
        x2 = sx(metrics.flat_region_right_mm)
        elements.append(f'<rect x="{x1:.2f}" y="{top}" width="{x2 - x1:.2f}" height="{plot_h}" fill="#f2c94c" opacity="0.18"/>')

    norm_x = sx(metrics.normalization_position_mm)
    elements.append(f'<line x1="{norm_x:.2f}" x2="{norm_x:.2f}" y1="{top}" y2="{top + plot_h}" stroke="#1f7a1f" stroke-dasharray="8 4" stroke-width="1.3"/>')
    elements.append(f'<text x="{norm_x + 4:.2f}" y="{top + 16}" font-family="Arial" font-size="10" fill="#1f7a1f">100% norm</text>')

    half_y = sy(metrics.half_dose_level)
    elements.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{half_y:.2f}" y2="{half_y:.2f}" stroke="#555" stroke-dasharray="6 5"/>')

    for value, label, dash in [
        (metrics.left_50_mm, "50", "6 5"),
        (metrics.right_50_mm, "50", "6 5"),
        (metrics.left_20_mm, "20", "2 4"),
        (metrics.left_80_mm, "80", "2 4"),
        (metrics.right_80_mm, "80", "2 4"),
        (metrics.right_20_mm, "20", "2 4"),
    ]:
        if value is None:
            continue
        xp = sx(float(value))
        elements.append(f'<line x1="{xp:.2f}" x2="{xp:.2f}" y1="{top}" y2="{top + plot_h}" stroke="#444" stroke-dasharray="{dash}"/>')
        elements.append(f'<text x="{xp + 4:.2f}" y="{top + plot_h - 8}" font-family="Arial" font-size="10">{label}%</text>')

    for x_start, x_end, width_mm, label in [
        (metrics.left_20_mm, metrics.left_80_mm, metrics.left_penumbra_mm, "Left penumbra"),
        (metrics.right_80_mm, metrics.right_20_mm, metrics.right_penumbra_mm, "Right penumbra"),
    ]:
        if x_start is None or x_end is None or width_mm is None:
            continue
        x1 = sx(float(x_start))
        x2 = sx(float(x_end))
        y = sy(0.8)
        text_x = 0.5 * (x1 + x2)
        text_y = y - 12
        elements.append(f'<line x1="{x1:.2f}" x2="{x2:.2f}" y1="{y:.2f}" y2="{y:.2f}" stroke="#8a4f00" stroke-width="2"/>')
        elements.append(f'<line x1="{x1:.2f}" x2="{x1:.2f}" y1="{y - 5:.2f}" y2="{y + 5:.2f}" stroke="#8a4f00" stroke-width="2"/>')
        elements.append(f'<line x1="{x2:.2f}" x2="{x2:.2f}" y1="{y - 5:.2f}" y2="{y + 5:.2f}" stroke="#8a4f00" stroke-width="2"/>')
        elements.append(f'<text x="{text_x:.2f}" y="{text_y:.2f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#8a4f00">{escape(label)}: {width_mm:.1f} mm</text>')

    elements.append(f'<polyline fill="none" stroke="#0b5cad" stroke-width="2" points="{polyline}"/>')
    for x, y in zip(positions_mm, relative_dose):
        elements.append(f'<circle cx="{sx(float(x)):.2f}" cy="{sy(float(y)):.2f}" r="2.4" fill="#0b5cad"/>')

    metric_parts = []
    if metrics.fwhm_mm is not None:
        metric_parts.append(f"FWHM {metrics.fwhm_mm:.1f} mm")
    if metrics.flatness_percent is not None:
        metric_parts.append(f"Flatness {metrics.flatness_percent:.2f}%")
    if metrics.symmetry_percent is not None:
        metric_parts.append(f"Symmetry {metrics.symmetry_percent:.2f}%")
    if metric_parts:
        elements.append(f'<text x="{left}" y="50" font-family="Arial" font-size="12">{escape(" | ".join(metric_parts))}</text>')

    elements.extend(
        [
            f'<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="13">{xlabel_text}</text>',
            f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="13">Relative dose, normalized at midpoint of 80% crossings</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements), encoding="utf-8")


def viridis_like_color(value: float) -> str:
    """Small three-stop color map for dependency-free heatmap SVGs."""
    rgb = viridis_like_rgb(value)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def viridis_like_rgb(value: float) -> Tuple[int, int, int]:
    """Small three-stop color map as RGB values."""
    stops = [
        (68, 1, 84),
        (33, 145, 140),
        (253, 231, 37),
    ]
    value = min(1.0, max(0.0, value))
    if value <= 0.5:
        lo, hi = stops[0], stops[1]
        t = value / 0.5
    else:
        lo, hi = stops[1], stops[2]
        t = (value - 0.5) / 0.5
    rgb = [round(lo[i] + t * (hi[i] - lo[i])) for i in range(3)]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def write_heatmap_png(
    path: Path,
    title: str,
    x_positions_mm: np.ndarray,
    y_positions_mm: np.ndarray,
    dose_matrix: np.ndarray,
    central_x_mm: float,
    central_y_mm: float,
) -> None:
    """Write a smooth dependency-light PNG heatmap using Pillow."""
    if Image is None or ImageDraw is None:
        write_heatmap_svg(
            path.with_suffix(".svg"),
            title,
            x_positions_mm,
            y_positions_mm,
            dose_matrix,
            central_x_mm,
            central_y_mm,
        )
        return

    width = 900
    height = 760
    left = 92
    right = 96
    top = 64
    bottom = 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    dmin = float(np.min(dose_matrix))
    dmax = float(np.max(dose_matrix))
    denom = dmax - dmin if dmax > dmin else 1.0

    normalized = (dose_matrix - dmin) / denom
    rgb = np.zeros((normalized.shape[0], normalized.shape[1], 3), dtype=np.uint8)
    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            rgb[row, col, :] = viridis_like_rgb(float(normalized[row, col]))

    heatmap = Image.fromarray(np.flipud(rgb), mode="RGB")
    resampling = getattr(Image.Resampling, "BICUBIC", Image.BICUBIC)
    heatmap = heatmap.resize((plot_w, plot_h), resampling)

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(heatmap, (left, top))

    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(35, 35, 35), width=1)

    def sx(x: float) -> float:
        return left + (x - float(x_positions_mm[0])) / (float(x_positions_mm[-1]) - float(x_positions_mm[0])) * plot_w

    def sy(y: float) -> float:
        return top + (float(y_positions_mm[-1]) - y) / (float(y_positions_mm[-1]) - float(y_positions_mm[0])) * plot_h

    center_x = sx(central_x_mm)
    center_y = sy(central_y_mm)
    draw.line([(center_x, top), (center_x, top + plot_h)], fill=(255, 255, 255), width=2)
    draw.line([(left, center_y), (left + plot_w, center_y)], fill=(255, 255, 255), width=2)

    draw.text((left, 26), title, fill=(20, 20, 20))
    draw.text((left + plot_w // 2 - 55, height - 32), "Cross-plane x (mm)", fill=(20, 20, 20))
    draw.text((12, top + plot_h // 2), "In-plane y (mm)", fill=(20, 20, 20))

    for x in np.linspace(float(x_positions_mm[0]), float(x_positions_mm[-1]), 7):
        px = sx(float(x))
        draw.line([(px, top + plot_h), (px, top + plot_h + 5)], fill=(35, 35, 35), width=1)
        draw.text((px - 12, top + plot_h + 9), f"{x:.0f}", fill=(20, 20, 20))

    for y in np.linspace(float(y_positions_mm[0]), float(y_positions_mm[-1]), 7):
        py = sy(float(y))
        draw.line([(left - 5, py), (left, py)], fill=(35, 35, 35), width=1)
        draw.text((left - 42, py - 6), f"{y:.0f}", fill=(20, 20, 20))

    bar_x = left + plot_w + 26
    bar_y = top
    bar_w = 20
    bar_h = plot_h
    for i in range(bar_h):
        value = 1.0 - i / max(1, bar_h - 1)
        draw.line(
            [(bar_x, bar_y + i), (bar_x + bar_w, bar_y + i)],
            fill=viridis_like_rgb(value),
            width=1,
        )
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline=(35, 35, 35), width=1)
    draw.text((bar_x + 28, bar_y - 2), f"{dmax:.4g}", fill=(20, 20, 20))
    draw.text((bar_x + 28, bar_y + bar_h - 10), f"{dmin:.4g}", fill=(20, 20, 20))
    draw.text((bar_x - 4, bar_y + bar_h + 12), "Dose (Gy)", fill=(20, 20, 20))

    canvas.save(path)


def write_heatmap_svg(
    path: Path,
    title: str,
    x_positions_mm: np.ndarray,
    y_positions_mm: np.ndarray,
    dose_matrix: np.ndarray,
    central_x_mm: float,
    central_y_mm: float,
) -> None:
    """Write a dependency-free SVG heatmap of the aligned matrix."""
    width = 760
    height = 700
    left = 78
    right = 28
    top = 58
    bottom = 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    rows, cols = dose_matrix.shape
    cell_w = plot_w / cols
    cell_h = plot_h / rows
    dmin = float(np.min(dose_matrix))
    dmax = float(np.max(dose_matrix))
    denom = dmax - dmin if dmax > dmin else 1.0

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="18" font-weight="700">{escape(title)}</text>',
    ]

    for row in range(rows):
        for col in range(cols):
            x = left + col * cell_w
            y = top + (rows - row - 1) * cell_h
            color = viridis_like_color((float(dose_matrix[row, col]) - dmin) / denom)
            elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w + 0.2:.2f}" height="{cell_h + 0.2:.2f}" fill="{color}"/>')

    def sx(x: float) -> float:
        return left + (x - float(x_positions_mm[0])) / (float(x_positions_mm[-1]) - float(x_positions_mm[0])) * plot_w

    def sy(y: float) -> float:
        return top + (float(y_positions_mm[-1]) - y) / (float(y_positions_mm[-1]) - float(y_positions_mm[0])) * plot_h

    elements.extend(
        [
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222"/>',
            f'<line x1="{sx(central_x_mm):.2f}" x2="{sx(central_x_mm):.2f}" y1="{top}" y2="{top + plot_h}" stroke="white" stroke-dasharray="6 5" stroke-width="1.2"/>',
            f'<line x1="{left}" x2="{left + plot_w}" y1="{sy(central_y_mm):.2f}" y2="{sy(central_y_mm):.2f}" stroke="white" stroke-dasharray="6 5" stroke-width="1.2"/>',
        ]
    )

    for x in np.linspace(float(x_positions_mm[0]), float(x_positions_mm[-1]), 7):
        xp = sx(float(x))
        elements.append(f'<text x="{xp:.2f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{x:.0f}</text>')
    for y in np.linspace(float(y_positions_mm[0]), float(y_positions_mm[-1]), 7):
        yp = sy(float(y))
        elements.append(f'<text x="{left - 10}" y="{yp + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{y:.0f}</text>')

    elements.extend(
        [
            f'<text x="{left + plot_w / 2}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="13">Cross-plane x (mm)</text>',
            f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="13">In-plane y (mm)</text>',
            f'<text x="{left}" y="{height - 42}" font-family="Arial" font-size="11">Dose color scale: {dmin:.4g} to {dmax:.4g} Gy</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements), encoding="utf-8")


def parse_setup_from_filename(stem: str) -> Dict[str, str]:
    """Extract simple setup labels from filenames such as 9MeV_6_two_2cmoff."""
    parts = stem.split("_")
    result = {
        "energy": parts[0] if len(parts) > 0 else "",
        "field_cm": parts[1] if len(parts) > 1 and parts[1].isdigit() else "",
        "plates": "",
        "offset": "",
    }
    for part in parts[2:]:
        if part in {"one", "two"}:
            result["plates"] = part
        elif result["offset"]:
            result["offset"] += "_" + part
        else:
            result["offset"] = part
    return result


def metric_row(
    mcc_file: Path,
    setup: Dict[str, str],
    axis: AxisMetrics,
    central_y_mm: float,
) -> Dict[str, object]:
    """Convert one AxisMetrics object to a flat CSV row."""
    return {
        "file": mcc_file.name,
        "energy": setup["energy"],
        "field_cm": setup["field_cm"],
        "plates": setup["plates"],
        "offset": setup["offset"],
        "axis": axis.axis_name,
        "normalization_x_mm": axis.normalization_position_mm,
        "profile_y_mm": central_y_mm,
        "normalization_dose_gy": axis.normalization_dose_gy,
        "peak_relative_dose": axis.peak_relative_dose,
        "peak_position_mm": axis.peak_position_mm,
        "fwhm_mm": axis.fwhm_mm,
        "left_50_mm": axis.left_50_mm,
        "right_50_mm": axis.right_50_mm,
        "left_penumbra_mm": axis.left_penumbra_mm,
        "right_penumbra_mm": axis.right_penumbra_mm,
        "flatness_percent": axis.flatness_percent,
        "symmetry_percent": axis.symmetry_percent,
    }


def process_mcc_file(mcc_file: Path, output_root: Path) -> List[Dict[str, object]]:
    """Analyze one MCC file and save all generated profile outputs."""
    profiles = read_mcc(mcc_file)
    (
        x_positions_mm,
        y_positions_mm,
        dose_matrix,
        measured_mask,
        interpolation_method,
    ) = build_aligned_dose_matrix(profiles)

    central_y_idx = int(np.argmin(np.abs(y_positions_mm)))
    central_x_idx = int(np.argmin(np.abs(x_positions_mm)))
    central_y_mm = float(y_positions_mm[central_y_idx])
    central_x_mm = float(x_positions_mm[central_x_idx])

    crossplane_dose = dose_matrix[central_y_idx, :]
    crossplane_measured = measured_mask[central_y_idx, :]
    crossplane_method = interpolation_method[central_y_idx, :]

    crossplane_relative, crossplane_metrics = analyze_axis_profile(
        "crossplane_x_at_y0",
        x_positions_mm,
        crossplane_dose,
    )

    scan_output_dir = output_root / mcc_file.stem
    scan_output_dir.mkdir(parents=True, exist_ok=True)

    write_raw_measurements_csv(scan_output_dir / "raw_detector_measurements.csv", profiles)
    write_matrix_csv(scan_output_dir / "aligned_dose_matrix_gy.csv", x_positions_mm, y_positions_mm, dose_matrix, "dose_gy")
    write_matrix_csv(
        scan_output_dir / "measured_mask.csv",
        x_positions_mm,
        y_positions_mm,
        measured_mask.astype(int),
        "directly_measured",
        fmt="%d",
    )
    write_method_matrix_csv(
        scan_output_dir / "interpolation_method.csv",
        x_positions_mm,
        y_positions_mm,
        interpolation_method,
    )
    np.save(scan_output_dir / "x_positions_mm.npy", x_positions_mm)
    np.save(scan_output_dir / "y_positions_mm.npy", y_positions_mm)
    np.save(scan_output_dir / "aligned_dose_matrix_gy.npy", dose_matrix)
    np.save(scan_output_dir / "measured_mask.npy", measured_mask)
    np.save(scan_output_dir / "interpolation_method.npy", interpolation_method)

    write_profile_csv(
        scan_output_dir / "central_crossplane_profile.csv",
        "crossplane_x_mm_at_y0",
        x_positions_mm,
        crossplane_dose,
        crossplane_relative,
        crossplane_measured,
        crossplane_method,
    )

    report_text = write_metrics_report(
        scan_output_dir / "analysis_report.txt",
        mcc_file,
        profiles,
        x_positions_mm,
        y_positions_mm,
        dose_matrix,
        measured_mask,
        central_y_mm,
        crossplane_metrics,
    )

    if plt is not None and PdfPages is not None:
        figures: List[plt.Figure] = []
        figures.append(
            plot_axis_profile(
                scan_output_dir / "central_crossplane_profile.png",
                f"{mcc_file.stem}: central cross-plane profile at y = {central_y_mm:.1f} mm",
                "Cross-plane x (mm)",
                x_positions_mm,
                crossplane_relative,
                crossplane_metrics,
            )
        )
        figures.append(
            plot_heatmap(
                scan_output_dir / "aligned_dose_heatmap.png",
                f"{mcc_file.stem}: aligned dose heatmap",
                x_positions_mm,
                y_positions_mm,
                dose_matrix,
                central_x_mm,
                central_y_mm,
            )
        )
        with PdfPages(scan_output_dir / "profile_report.pdf") as pdf:
            for fig in figures:
                pdf.savefig(fig)
            fig_text, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            ax.text(0.02, 0.98, report_text, va="top", ha="left", family="monospace", fontsize=8.5)
            fig_text.tight_layout()
            pdf.savefig(fig_text)
            plt.close(fig_text)

        for fig in figures:
            plt.close(fig)
    else:
        write_axis_profile_svg(
            scan_output_dir / "central_crossplane_profile.svg",
            f"{mcc_file.stem}: central cross-plane profile at y = {central_y_mm:.1f} mm",
            "Cross-plane x (mm)",
            x_positions_mm,
            crossplane_relative,
            crossplane_metrics,
        )
        write_heatmap_png(
            scan_output_dir / "aligned_dose_heatmap.png",
            f"{mcc_file.stem}: aligned dose heatmap",
            x_positions_mm,
            y_positions_mm,
            dose_matrix,
            central_x_mm,
            central_y_mm,
        )

    setup = parse_setup_from_filename(mcc_file.stem)
    return [
        metric_row(mcc_file, setup, crossplane_metrics, central_y_mm),
    ]


def write_summary_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_pdf(path: Path, rows: List[Dict[str, object]]) -> None:
    """Write a compact multipage summary table."""
    if plt is None or PdfPages is None:
        return

    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("Batch summary: crossplane_x_at_y0", loc="left", fontsize=13, pad=14)

        columns = [
            "file",
            "normalization_dose_gy",
            "fwhm_mm",
            "left_penumbra_mm",
            "right_penumbra_mm",
            "flatness_percent",
            "symmetry_percent",
        ]
        table_data = []
        for row in rows:
            table_data.append(
                [
                    row["file"],
                    f"{float(row['normalization_dose_gy']):.5g}",
                    format_optional(row["fwhm_mm"]),
                    format_optional(row["left_penumbra_mm"]),
                    format_optional(row["right_penumbra_mm"]),
                    format_optional(row["flatness_percent"]),
                    format_optional(row["symmetry_percent"]),
                ]
            )

        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.8)
        table.scale(1.0, 1.2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mcc_files = sorted(INPUT_DIR.glob("*.mcc"))
    if not mcc_files:
        raise FileNotFoundError(f"No .mcc files found in {INPUT_DIR}")

    print(f"Found {len(mcc_files)} MCC files in {INPUT_DIR}")
    print(f"Writing generated profiles to {OUTPUT_DIR}")

    summary_rows: List[Dict[str, object]] = []
    for mcc_file in mcc_files:
        print(f"Processing {mcc_file.name}")
        summary_rows.extend(process_mcc_file(mcc_file, OUTPUT_DIR))

    write_summary_csv(OUTPUT_DIR / "batch_profile_summary.csv", summary_rows)
    write_summary_pdf(OUTPUT_DIR / "batch_profile_summary.pdf", summary_rows)

    print(f"Done. Summary CSV: {OUTPUT_DIR / 'batch_profile_summary.csv'}")
    if (OUTPUT_DIR / "batch_profile_summary.pdf").exists():
        print(f"Done. Summary PDF: {OUTPUT_DIR / 'batch_profile_summary.pdf'}")
    else:
        print(
            "Matplotlib is not installed; wrote SVG profile plots and "
            "Pillow PNG heatmaps instead of matplotlib PNG/PDF plots."
        )


if __name__ == "__main__":
    main()
