from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# where we want results saved, where we want to read from
OUTPUT_DIR = r"/path/to/output/"
MCC_FILE = r"/path/to/mcc/"

# it's okay if the output directory folder already exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# this pulls numerical values out of mcc file - needed because the raw file
# contains lots of plain text
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# define a custom data container named profile
# each profile represents one begin scan to end scan block of the .mcc
@dataclass
class Profile:
    """
    One MCC BEGIN_SCAN block.

    Attributes
    ----------
    scan_number:
        Scan number written in the MCC file.

    inplane_position:
        Physical row coordinate from SCAN_OFFAXIS_INPLANE.

    x:
        Measured cross-plane detector coordinates for this row.

    dose:
        Measured dose values corresponding one-to-one with x.
    """
    scan_number: int
    inplane_position: float
    x: np.ndarray
    dose: np.ndarray

# define a function called read_mcc
def read_mcc(filepath: str) -> List[Profile]:

    # take a file path and return a list of profile objects defined above
    # starts an empty list, each scan profile added here
    profiles: List[Profile] = []

    # inside_scan becomes true after begin_scan
    # inside data becomes true after begin data
    inside_scan = False
    inside_data = False

    # store temporary data while reading one scan
    # once the scan ends this becomes a PROFILE
    current_scan_number: Optional[int] = None
    current_inplane_position: Optional[float] = None
    current_x: List[float] = []
    current_dose: List[float] = []

    # opens the mcc file for reading, ignore errors to prevent from crashing
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        # read the file one line at a time
        for raw_line in f:
            # remove space, tabs, newlines from beginning and end
            stripped = raw_line.strip()

            if not stripped:
                continue

            # BEGIN_SCAN_DATA must not be mistaken for BEGIN_SCAN.
            # checks whether the current line looks EXACTLY like BEGIN_SCAN 1, etc.
            match_scan = re.fullmatch(r"BEGIN_SCAN\s+(\d+)", stripped)
            # if this line is a scan start, reset everything for a new profile
            if match_scan:
                # we are now inside a scan
                # save scan number, clear old x and dose lists, move to next line
                inside_scan = True
                inside_data = False
                current_scan_number = int(match_scan.group(1))
                current_inplane_position = None
                current_x = []
                current_dose = []
                continue

            # if we haven't entered a begin_scan yet, ignore the line
            if not inside_scan:
                continue

            # finds the line that tells us info about the in-plane direction
            if stripped.startswith("SCAN_OFFAXIS_INPLANE="):
                value_text = stripped.split("=", 1)[1].strip()
                # convert "SCAN_OFFAXIS_INPLANE=-125.00" to -125.00
                current_inplane_position = float(value_text)
                continue

            # start reading measurement rows
            if stripped == "BEGIN_DATA":
                inside_data = True
                continue

            # stop reading measurement rows
            if stripped == "END_DATA":
                inside_data = False
                continue

            # detects the end of a scan
            if re.fullmatch(r"END_SCAN\s+\d+", stripped):

                # check whether the scan had all required information
                if current_scan_number is None:
                    raise ValueError("Encountered END_SCAN without a scan number.")

                if current_inplane_position is None:
                    raise ValueError(
                        f"Scan {current_scan_number} is missing "
                        "SCAN_OFFAXIS_INPLANE."
                    )

                if not current_x:
                    raise ValueError(
                        f"Scan {current_scan_number} contains no data points."
                    )

                # the collected x and dose values in python lists are now
                # converted to numpy arrays
                x_array = np.asarray(current_x, dtype=float)
                dose_array = np.asarray(current_dose, dtype=float)

                # there should be one dose value for each x position
                if x_array.size != dose_array.size:
                    raise ValueError(
                        f"Scan {current_scan_number} has mismatched x and dose "
                        f"lengths: {x_array.size} versus {dose_array.size}."
                    )

                # Sort each profile by physical x coordinate. This makes later
                # interpolation safe even if a file stores a scan in reverse.
                # sort x positions from low to high, rearrange the dose values
                # in the same order
                order = np.argsort(x_array)
                x_array = x_array[order]
                dose_array = dose_array[order]

                # check for duplicate x values
                if np.unique(x_array).size != x_array.size:
                    raise ValueError(
                        f"Scan {current_scan_number} contains duplicate "
                        "cross-plane positions."
                    )

                # creates one profile object and adds it to the list
                # at this point, one MCC scan has become one structured
                # Python object, class Profiles
                profiles.append(
                    Profile(
                        scan_number=current_scan_number,
                        inplane_position=float(current_inplane_position),
                        x=x_array,
                        dose=dose_array,
                    )
                )

                # reset after the scan
                # clears temporary variables so next scan can start
                inside_scan = False
                inside_data = False
                current_scan_number = None
                current_inplane_position = None
                current_x = []
                current_dose = []
                continue

            # read only numeric measurement lines between begin data and end data
            if inside_data:
                # A data line has the form:
                #   -130.00    8.2674E-03    #1379
                #
                # Remove the detector-number comment first so "#1379" is not
                # incorrectly treated as a third physical data value.
                data_text = stripped.split("#", 1)[0].strip()
                # extract numbers from the cleaned line
                numbers = NUMBER_PATTERN.findall(data_text)

                # if the line does not contain at least x and dose, skip
                if len(numbers) < 2:
                    continue
                # first number is cross-plane position, second is dose
                current_x.append(float(numbers[0]))
                current_dose.append(float(numbers[1]))

    # if no profiles were found
    if not profiles:
        raise ValueError(
            "No valid MCC profiles were found. Check the file path and format."
        )

    # Put rows into physical in-plane order rather than trusting file order.
    profiles.sort(key=lambda p: p.inplane_position)

    # send list of profiles back to the rest of the script
    return profiles


# function to find common grid spacing
# figures out the smallest x spacing present across all profiles
def estimate_common_grid_spacing(
    profiles: List[Profile],
    decimals: int = 8,
) -> float:
    """
    Estimate the smallest physical x increment represented across all rows.

    For the staggered OCTAVIUS geometry in the supplied MCC excerpt:
      - one row contains ..., -130, -120, -110, ...
      - the next contains ..., -125, -115, -105, ...

    Each individual row has 10 mm detector spacing, but the union of all row
    coordinates has a 5 mm increment. Therefore, 5 mm is the appropriate common
    grid for an aligned rectangular representation.

    Rounding is used only to suppress tiny floating-point differences.
    """
    # collect every x position from every pofile into one big array
    all_positions = np.concatenate([p.x for p in profiles])
    # rounds tiny floating point differences and keeps only unique positions
    unique_positions = np.unique(np.round(all_positions, decimals=decimals))
    # sort low to high
    unique_positions.sort()

    # calculate the spacing between neighboring x positions
    # Octavius 1500 -> [-130, -125, -120] -> [5, 5]
    differences = np.diff(unique_positions)
    # remove zero or tiny fake differences caused by floating point
    differences = differences[differences > 10 ** (-decimals)]

    # if position is truly zero, error
    if differences.size == 0:
        raise ValueError("Could not infer a common cross-plane grid spacing.")

    # pick the smallest real spacing
    spacing = float(np.min(differences))

    # error for negative spacing
    if spacing <= 0:
        raise ValueError(f"Invalid inferred grid spacing: {spacing}")

    # return spacing to the rest of the script
    return spacing

# make a real 2D dose matrix where every column corresponds to the same x
def build_aligned_dose_matrix(
    profiles: List[Profile],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a spatially aligned rectangular dose matrix.

    Why interpolation is necessary
    ------------------------------
    Alternating detector rows are staggered by half a detector pitch. Their
    measured x coordinates are therefore different. A rectangular NumPy array
    requires every column to represent the same physical x coordinate.

    The original code stacked dose values by array index. That falsely treated,
    for example, a measurement at x=-125 mm as though it were at x=-130 mm.

    This function creates a common physical x grid and linearly interpolates
    each measured row onto that grid.

    Important interpretation
    ------------------------
    The resulting matrix is a spatially aligned reconstruction. Values that
    fall between two detectors in a row are interpolated estimates, not new
    independent detector measurements.

    Returns
    -------
    x_positions:
        Common cross-plane grid.

    y_positions:
        Physical in-plane row coordinates from SCAN_OFFAXIS_INPLANE.

    dose_matrix:
        Aligned matrix with shape [n_profiles, n_x_positions].

    measured_mask:
        Boolean matrix of the same shape. True means that grid location was
        directly measured in that row; False means it was interpolated.
    """
    if not profiles:
        raise ValueError("No profiles were supplied.")

    # make an array of all physical row positions
    y_positions = np.asarray(
        [p.inplane_position for p in profiles],
        dtype=float,
    )

    # Restrict the matrix to the x interval covered by every row. This avoids
    # extrapolating beyond the outermost detector in any profile.
    # shared range is only the part covered by every row (-125 to 125, for ex)
    common_min = max(float(np.min(p.x)) for p in profiles)
    common_max = min(float(np.max(p.x)) for p in profiles)

    if common_max <= common_min:
        raise ValueError(
            "The profiles do not share a common cross-plane coordinate range."
        )

    # find the common x spacing, usually 5 mm
    spacing = estimate_common_grid_spacing(profiles)

    # calculate how many spacing intervals fit between common min and max
    n_steps = int(np.floor((common_max - common_min) / spacing + 1e-9))
    # define the common x grid based on the minimum, spacing, and n_steps
    x_positions = common_min + spacing * np.arange(n_steps + 1, dtype=float)
    # so now we have [-125, -120, -115,..., 120, 125]

    # Include common_max if floating-point rounding left it just beyond the
    # final generated point.
    if common_max - x_positions[-1] > spacing * 1e-6:
        x_positions = np.append(x_positions, common_max)

    # collect the aligned dose rows and measured/interpolated masks
    aligned_rows: List[np.ndarray] = []
    measured_rows: List[np.ndarray] = []

    # define how close two coordinates need to be to count as the same
    tolerance = max(1e-8, spacing * 1e-6)

    # loop through every detector row
    for profile in profiles:
        # np.interp performs linear interpolation only inside the measured
        # range. Because x_positions is restricted to the common overlap,
        # no extrapolation is performed here.
        aligned_dose = np.interp(
            x_positions,
            profile.x,
            profile.dose,
        )
        aligned_rows.append(aligned_dose)
        # that is, if a row originally has [-125, --15, -105,...]
        # it will now have [-125, -120, -115, -110, ...]

        # Record which aligned cells correspond to an actual detector
        # coordinate in this particular row
        # this compares every common x position to every measured x position in profile
        # it determines distance between each grid point and nearest real detector
        distance_to_nearest_measurement = np.min(
            np.abs(x_positions[:, None] - profile.x[None, :]),
            axis=1,
        )
        # if a grid point is equal to real detector, mark True
        measured_rows.append(
            distance_to_nearest_measurement <= tolerance
        )

    # stack all rows into 2D arrays
    # shape = (number of in-plane rows) * (number of cross-plane positions)
    dose_matrix = np.vstack(aligned_rows)
    measured_mask = np.vstack(measured_rows)

    # return physical x coords, physical y coords, aligned dose data,
    # and measured or interpolated flags
    return x_positions, y_positions, dose_matrix, measured_mask


# =============================================================================
# 3. SAVE RAW AND ALIGNED DATA
# =============================================================================

# save the original data exactly as measured
def save_raw_measurements_csv(
    csv_path: str,
    profiles: List[Profile],
) -> None:
    """
    Save every original detector measurement without interpolation.

    This long-format file is the audit trail for the raw MCC data.
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scan_number",
                "inplane_position",
                "crossplane_position",
                "dose",
            ]
        )

        for profile in profiles:
            for x, dose in zip(profile.x, profile.dose):
                writer.writerow(
                    [
                        profile.scan_number,
                        profile.inplane_position,
                        x,
                        dose,
                    ]
                )

    print(f"Raw measurement CSV saved to: {csv_path}")

# saves corrected rectangular matrix
def save_aligned_dose_csv(
    csv_path: str,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    dose_matrix: np.ndarray,
) -> None:
    """
    Save the aligned dose matrix.

    Each profile column is labeled with its physical in-plane coordinate rather
    than only a sequential profile number.
    """
    if dose_matrix.shape != (len(y_positions), len(x_positions)):
        raise ValueError(
            "Dose matrix shape does not match x/y coordinate lengths."
        )

    header_columns = ["crossplane_position"]
    header_columns.extend(
        f"dose_at_inplane_{y:g}" for y in y_positions
    )

    output = np.column_stack([x_positions, dose_matrix.T])

    np.savetxt(
        csv_path,
        output,
        delimiter=",",
        header=",".join(header_columns),
        comments="",
    )

    print(f"Aligned dose matrix CSV saved to: {csv_path}")

# save measured/interpolated mask
def save_measured_mask_csv(
    csv_path: str,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    measured_mask: np.ndarray,
) -> None:
    """
    Save a 1/0 mask identifying measured versus interpolated matrix cells.

    1 = directly measured detector location
    0 = linearly interpolated location
    """
    header_columns = ["crossplane_position"]
    header_columns.extend(
        f"measured_at_inplane_{y:g}" for y in y_positions
    )

    output = np.column_stack(
        [x_positions, measured_mask.astype(int).T]
    )

    np.savetxt(
        csv_path,
        output,
        delimiter=",",
        header=",".join(header_columns),
        comments="",
        fmt=["%.8g"] + ["%d"] * measured_mask.shape[0],
    )

    print(f"Measured/interpolated mask CSV saved to: {csv_path}")


# =============================================================================
# 4. PROFILE-LEVEL HELPERS
# =============================================================================

# function finds where profile crosses certain dose level
def interpolate_crossing(
    x: np.ndarray,
    y: np.ndarray,
    level: float,
    side: str,
) -> Optional[float]:
    """
    Find a linearly interpolated level crossing on one side of the field.

    Parameters
    ----------
    side:
        "left" or "right". The crossing nearest the central axis is returned.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")

    # separate left half from right half
    if side == "left":
        mask = x <= 0
    else:
        mask = x >= 0

    # keep only selected side
    xs = x[mask]
    ys = y[mask]

    # need at least two points to find a crossing
    if xs.size < 2:
        return None

    # sort x-values from low to high
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    # store possible crossing positions
    crossings: List[float] = []

    # look at each neighboring pair of points
    for i in range(len(xs) - 1):
        # these are the two dose values around that segment
        y1 = ys[i]
        y2 = ys[i + 1]

        # if both points are above or both below the level, this segment does not cross
        if (y1 - level) * (y2 - level) > 0:
            continue

        if np.isclose(y1, y2):
            if np.isclose(y1, level):
                crossings.append(float(0.5 * (xs[i] + xs[i + 1])))
            continue

        # how far between the two points the crossing occurs
        fraction = (level - y1) / (y2 - y1)

        if 0 <= fraction <= 1:
            # convert that fraction into an x coordinate
            crossing = xs[i] + fraction * (xs[i + 1] - xs[i])
            crossings.append(float(crossing))

    if not crossings:
        return None

    # Choose the edge crossing nearest x=0. This is more robust if noise creates
    # an additional crossing far outside the primary field edge.
    return min(crossings, key=abs)


# =============================================================================
# 5. ANALYZE DOSE
# =============================================================================

# calculate summary metrics
def analyze_dose(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    dose_matrix: np.ndarray,
    report_path: str,
    central_profile_idx: int,
    profile_norm: np.ndarray,
) -> Dict[str, object]:
    """
    Calculate central-profile and two-dimensional summary metrics.

    The matrix supplied here has already been spatially aligned, so every
    matrix column corresponds to the same physical x coordinate in every row.
    """
    lines: List[str] = []
    lines.append("[Basic dose analysis]\n")

    metrics: Dict[str, object] = {
        "central_profile_idx": int(central_profile_idx),
        "central_profile_y": float(y_positions[central_profile_idx]),
        "profile_norm": profile_norm,
    }

    # -------------------------------------------------------------------------
    # Global maximum over the aligned 2D matrix
    # -------------------------------------------------------------------------
    # finds highest dose anywhere in the 2D aligned matrix
    global_max = float(np.max(dose_matrix))
    max_profile_idx, max_pos_idx = np.unravel_index(
        np.argmax(dose_matrix),
        dose_matrix.shape,
    )

    metrics["global_max"] = global_max
    metrics["global_max_profile_idx"] = int(max_profile_idx)
    metrics["global_max_y"] = float(y_positions[max_profile_idx])
    metrics["global_max_x"] = float(x_positions[max_pos_idx])

    lines.append(f"Global max dose: {global_max:.6f}\n")
    lines.append(
        "  -> at "
        f"in-plane y = {y_positions[max_profile_idx]:.3f}, "
        f"cross-plane x = {x_positions[max_pos_idx]:.3f}\n"
    )

    # -------------------------------------------------------------------------
    # Dose along the physical x≈0 column
    # -------------------------------------------------------------------------
    # finds the x position closest to zero
    central_axis_idx = int(np.argmin(np.abs(x_positions)))
    central_axis_x = float(x_positions[central_axis_idx])
    # take the whole vertical column at x = 0
    central_axis_dose = dose_matrix[:, central_axis_idx]

    metrics["central_axis_idx"] = central_axis_idx
    metrics["central_axis_x"] = central_axis_x

    lines.append(
        f"\nCentral-axis column (x ≈ {central_axis_x:.3f}) dose stats:\n"
    )
    lines.append(f"  Min:  {central_axis_dose.min():.6f}\n")
    lines.append(f"  Max:  {central_axis_dose.max():.6f}\n")
    lines.append(f"  Mean: {central_axis_dose.mean():.6f}\n")

    # -------------------------------------------------------------------------
    # FWHM of the physical central in-plane profile
    # -------------------------------------------------------------------------
    max_normalized = float(np.max(profile_norm))
    half_max_level = 0.5 * max_normalized

    x_left50 = interpolate_crossing(
        x_positions,
        profile_norm,
        half_max_level,
        side="left",
    )
    x_right50 = interpolate_crossing(
        x_positions,
        profile_norm,
        half_max_level,
        side="right",
    )

    fwhm: Optional[float] = None

    if x_left50 is not None and x_right50 is not None:
        fwhm = float(x_right50 - x_left50)

        metrics["FWHM"] = fwhm
        metrics["x_left50"] = x_left50
        metrics["x_right50"] = x_right50

        lines.append(
            "\nFWHM of central profile "
            f"(in-plane y = {y_positions[central_profile_idx]:.3f}):\n"
        )
        lines.append(f"  FWHM = {fwhm:.3f}\n")
        lines.append(
            f"  Half-max interval: {x_left50:.3f} to {x_right50:.3f}\n"
        )
    else:
        lines.append(
            "\nCould not compute FWHM because both 50% crossings were not found.\n"
        )

    # -------------------------------------------------------------------------
    # Flatness and symmetry in the central 80% of the FWHM-defined field
    # -------------------------------------------------------------------------
    flatness: Optional[float] = None
    symmetry: Optional[float] = None

    if fwhm is not None and x_left50 is not None and x_right50 is not None:
        field_center = 0.5 * (x_left50 + x_right50)
        field_width = x_right50 - x_left50

        # Central 80% means ±40% of the full field width around field center.
        flat_left = field_center - 0.4 * field_width
        flat_right = field_center + 0.4 * field_width

        metrics["flat_left"] = float(flat_left)
        metrics["flat_right"] = float(flat_right)

        flat_mask = (
            (x_positions >= flat_left)
            & (x_positions <= flat_right)
        )
        profile_flat = profile_norm[flat_mask]

        if profile_flat.size > 0:
            dmax = float(np.max(profile_flat))
            dmin = float(np.min(profile_flat))

            denominator = dmax + dmin
            if denominator > 0:
                flatness = 100.0 * (dmax - dmin) / denominator

                metrics["flatness"] = float(flatness)
                metrics["Dmax_c"] = dmax
                metrics["Dmin_c"] = dmin

                lines.append(
                    "\nFlatness in central 80% of FWHM-defined field:\n"
                )
                lines.append(f"  Dmax: {dmax:.6f}\n")
                lines.append(f"  Dmin: {dmin:.6f}\n")
                lines.append(f"  Flatness: {flatness:.2f} %\n")
            else:
                lines.append(
                    "\nFlatness could not be computed because Dmax + Dmin "
                    "was zero.\n"
                )
        else:
            lines.append(
                "\nFlatness could not be computed because the central "
                "80% region contained no points.\n"
            )

        # Symmetry is evaluated around the measured field center, not blindly
        # around x=0. That accommodates a slightly shifted field.
        maximum_pair_difference = 0.0
        pair_count = 0

        central_indices = np.where(flat_mask)[0]

        for i in central_indices:
            mirrored_x = 2.0 * field_center - x_positions[i]
            j = int(np.argmin(np.abs(x_positions - mirrored_x)))

            if not flat_mask[j]:
                continue

            maximum_pair_difference = max(
                maximum_pair_difference,
                abs(float(profile_norm[i] - profile_norm[j])),
            )
            pair_count += 1

        if pair_count > 0:
            symmetry = 100.0 * maximum_pair_difference
            metrics["symmetry"] = float(symmetry)

            lines.append(
                "\nSymmetry in central 80% of FWHM-defined field:\n"
            )
            lines.append(
                "  Maximum paired normalized-dose difference: "
                f"{symmetry:.2f} %\n"
            )
        else:
            lines.append(
                "\nSymmetry could not be computed because no mirrored "
                "point pairs were available.\n"
            )
    else:
        lines.append(
            "\nFlatness and symmetry could not be computed without a "
            "valid FWHM.\n"
        )

    # -------------------------------------------------------------------------
    # 80%-20% penumbra
    # -------------------------------------------------------------------------
    x20_left = interpolate_crossing(
        x_positions,
        profile_norm,
        0.2,
        side="left",
    )
    x80_left = interpolate_crossing(
        x_positions,
        profile_norm,
        0.8,
        side="left",
    )
    x20_right = interpolate_crossing(
        x_positions,
        profile_norm,
        0.2,
        side="right",
    )
    x80_right = interpolate_crossing(
        x_positions,
        profile_norm,
        0.8,
        side="right",
    )

    metrics["x20_left"] = x20_left
    metrics["x80_left"] = x80_left
    metrics["x20_right"] = x20_right
    metrics["x80_right"] = x80_right

    lines.append(
        "\n80%-20% penumbra, relative to central-axis normalization:\n"
    )

    if x20_left is not None and x80_left is not None:
        penumbra_left = abs(x80_left - x20_left)
        metrics["penumbra_left"] = float(penumbra_left)
        lines.append(
            f"  Left:  x20 = {x20_left:.3f}, "
            f"x80 = {x80_left:.3f}, "
            f"width = {penumbra_left:.3f}\n"
        )
    else:
        lines.append(
            "  Left: could not determine both 20% and 80% crossings.\n"
        )

    if x20_right is not None and x80_right is not None:
        penumbra_right = abs(x80_right - x20_right)
        metrics["penumbra_right"] = float(penumbra_right)
        lines.append(
            f"  Right: x20 = {x20_right:.3f}, "
            f"x80 = {x80_right:.3f}, "
            f"width = {penumbra_right:.3f}\n"
        )
    else:
        lines.append(
            "  Right: could not determine both 20% and 80% crossings.\n"
        )

    report_text = "".join(lines)
    metrics["report_text"] = report_text

    print(report_text)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Analysis report saved to: {report_path}")

    return metrics


# =============================================================================
# 6. PLOT CENTRAL PROFILE
# =============================================================================

def plot_central_profile_with_annotations(
    x_positions: np.ndarray,
    profile_norm: np.ndarray,
    metrics: Dict[str, object],
    out_path_png: str,
) -> plt.Figure:
    """Plot the selected physical central profile and its beam metrics."""
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        x_positions,
        profile_norm,
        marker="o",
        markersize=3,
        label="Central in-plane profile",
    )

    flat_left = metrics.get("flat_left")
    flat_right = metrics.get("flat_right")

    if flat_left is not None and flat_right is not None:
        ax.axvspan(
            float(flat_left),
            float(flat_right),
            alpha=0.15,
            label="Central 80% field region",
        )

    x_left50 = metrics.get("x_left50")
    x_right50 = metrics.get("x_right50")

    if x_left50 is not None:
        ax.axvline(
            float(x_left50),
            linestyle="--",
            linewidth=1,
            label="50% field edges",
        )

    if x_right50 is not None:
        ax.axvline(
            float(x_right50),
            linestyle="--",
            linewidth=1,
        )

    level_markers = [
        (metrics.get("x20_left"), "20% left"),
        (metrics.get("x80_left"), "80% left"),
        (metrics.get("x20_right"), "20% right"),
        (metrics.get("x80_right"), "80% right"),
    ]

    for x_value, label in level_markers:
        if x_value is None:
            continue

        x_value = float(x_value)
        ax.axvline(x_value, linestyle=":", linewidth=1)
        ax.text(
            x_value,
            0.08,
            label,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=7,
        )

    subtitle_parts: List[str] = []

    if metrics.get("FWHM") is not None:
        subtitle_parts.append(f"FWHM = {float(metrics['FWHM']):.1f}")

    if metrics.get("flatness") is not None:
        subtitle_parts.append(
            f"Flatness = {float(metrics['flatness']):.1f}%"
        )

    if metrics.get("symmetry") is not None:
        subtitle_parts.append(
            f"Symmetry = {float(metrics['symmetry']):.1f}%"
        )

    central_y = float(metrics["central_profile_y"])
    subtitle = " | ".join(subtitle_parts)

    title = f"Central profile at in-plane y = {central_y:.1f}"
    if subtitle:
        title += "\n" + subtitle

    ax.set_xlabel("Cross-plane position")
    ax.set_ylabel("Relative dose normalized at x ≈ 0")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path_png, dpi=300)

    print(f"Annotated central profile saved to: {out_path_png}")

    return fig


# =============================================================================
# 7. PLOT PHYSICALLY ALIGNED HEATMAP
# =============================================================================

def plot_and_save_dose_heatmap(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    dose_matrix: np.ndarray,
    out_path_png: str,
    title: str = "Dose heatmap: in-plane versus cross-plane position",
) -> plt.Figure:
    """
    Plot the aligned matrix using actual physical x and y coordinates.

    pcolormesh is used instead of labeling rows merely by profile index.
    It respects the supplied coordinate arrays and does not introduce an
    additional display-only smoothing step.
    :type y_positions: np.ndarray
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        dose_matrix,
        extent=[
            x_positions[0],
            x_positions[-1],
            y_positions[0],
            y_positions[-1],
        ],
        origin="lower",
        interpolation="bicubic",
        aspect="equal",
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Dose")

    ax.set_xlabel("Cross-plane position")
    ax.set_ylabel("In-plane position")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(out_path_png, dpi=300)

    print(f"Heatmap plot saved to: {out_path_png}")

    return fig


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> None:
    # -------------------------------------------------------------------------
    # Read the MCC file as explicit scan blocks
    # -------------------------------------------------------------------------
    profiles = read_mcc(MCC_FILE)

    print(f"Number of profiles found: {len(profiles)}")
    print(
        "First profile: "
        f"scan={profiles[0].scan_number}, "
        f"in-plane={profiles[0].inplane_position:.3f}, "
        f"points={len(profiles[0].x)}"
    )
    print(
        "Last profile: "
        f"scan={profiles[-1].scan_number}, "
        f"in-plane={profiles[-1].inplane_position:.3f}, "
        f"points={len(profiles[-1].x)}"
    )

    # -------------------------------------------------------------------------
    # Build a common, physically aligned x grid
    # -------------------------------------------------------------------------
    # turns staggered detector profiles into a coordinate-corrected 2D matrix
    (
        x_positions,
        y_positions,
        dose_matrix,
        measured_mask,
    ) = build_aligned_dose_matrix(profiles)

    print(
        "\nAligned dose matrix shape [n_inplane_rows, n_crossplane_points]: "
        f"{dose_matrix.shape}"
    )
    print(
        f"Cross-plane grid: {x_positions[0]:.3f} to "
        f"{x_positions[-1]:.3f}"
    )
    print(
        f"In-plane grid: {y_positions[0]:.3f} to "
        f"{y_positions[-1]:.3f}"
    )
    print(
        "Common cross-plane spacing: "
        f"{np.median(np.diff(x_positions)):.3f}"
    )
    print(
        "Directly measured aligned cells: "
        f"{measured_mask.sum()} / {measured_mask.size}"
    )

    # -------------------------------------------------------------------------
    # Select the profile physically nearest y=0
    # -------------------------------------------------------------------------
    central_profile_idx = int(np.argmin(np.abs(y_positions)))
    central_profile_y = float(y_positions[central_profile_idx])
    central_profile = dose_matrix[central_profile_idx, :]

    central_axis_idx = int(np.argmin(np.abs(x_positions)))
    central_axis_x = float(x_positions[central_axis_idx])
    central_dose = float(central_profile[central_axis_idx])

    if not np.isfinite(central_dose) or np.isclose(central_dose, 0.0):
        raise ValueError(
            "The selected central-axis dose is zero or non-finite, so the "
            "central profile cannot be normalized."
        )

    profile_norm = central_profile / central_dose

    print(f"\nCentral profile index: {central_profile_idx}")
    print(f"Central profile y: {central_profile_y:.3f}")
    print(
        f"Central axis x ≈ {central_axis_x:.3f}, "
        f"central dose = {central_dose:.6f}"
    )

    # -------------------------------------------------------------------------
    # Save NumPy arrays
    # -------------------------------------------------------------------------
    x_npy_path = os.path.join(OUTPUT_DIR, "x_positions_aligned.npy")
    y_npy_path = os.path.join(OUTPUT_DIR, "y_positions.npy")
    dose_npy_path = os.path.join(OUTPUT_DIR, "dose_matrix_aligned.npy")
    mask_npy_path = os.path.join(OUTPUT_DIR, "measured_mask.npy")

    np.save(x_npy_path, x_positions)
    np.save(y_npy_path, y_positions)
    np.save(dose_npy_path, dose_matrix)
    np.save(mask_npy_path, measured_mask)

    print(f"x positions saved to: {x_npy_path}")
    print(f"y positions saved to: {y_npy_path}")
    print(f"Aligned dose matrix saved to: {dose_npy_path}")
    print(f"Measured/interpolated mask saved to: {mask_npy_path}")

    # -------------------------------------------------------------------------
    # Save both raw and aligned CSV data
    # -------------------------------------------------------------------------
    raw_csv_path = os.path.join(
        OUTPUT_DIR,
        "raw_detector_measurements_Aluminum_midline_105SSD.csv",
    )
    aligned_csv_path = os.path.join(
        OUTPUT_DIR,
        "dose_matrix_aligned_Aluminum_midline_105SSD.csv",
    )
    mask_csv_path = os.path.join(
        OUTPUT_DIR,
        "measured_mask_Aluminum_midline_105SSD.csv",
    )

    save_raw_measurements_csv(raw_csv_path, profiles)
    save_aligned_dose_csv(
        aligned_csv_path,
        x_positions,
        y_positions,
        dose_matrix,
    )
    save_measured_mask_csv(
        mask_csv_path,
        x_positions,
        y_positions,
        measured_mask,
    )

    # -------------------------------------------------------------------------
    # Analyze the aligned dose data
    # -------------------------------------------------------------------------
    report_path = os.path.join(
        OUTPUT_DIR,
        "dose_analysis_Aluminum_midline_105SSD.txt",
    )

    metrics = analyze_dose(
        x_positions=x_positions,
        y_positions=y_positions,
        dose_matrix=dose_matrix,
        report_path=report_path,
        central_profile_idx=central_profile_idx,
        profile_norm=profile_norm,
    )

    # -------------------------------------------------------------------------
    # Create plots
    # -------------------------------------------------------------------------
    central_png = os.path.join(
        OUTPUT_DIR,
        "central_profile_annotated.png",
    )
    fig_profile = plot_central_profile_with_annotations(
        x_positions=x_positions,
        profile_norm=profile_norm,
        metrics=metrics,
        out_path_png=central_png,
    )

    heatmap_png = os.path.join(
        OUTPUT_DIR,
        "dose_heatmap_aligned.png",
    )
    fig_heatmap = plot_and_save_dose_heatmap(
        x_positions=x_positions,
        y_positions=y_positions,
        dose_matrix=dose_matrix,
        out_path_png=heatmap_png,
        title="Aligned dose heatmap: in-plane vs cross-plane position",
    )

    # -------------------------------------------------------------------------
    # Save the PDF report
    # -------------------------------------------------------------------------
    pdf_path = os.path.join(
        OUTPUT_DIR,
        "Aluminum_midline_105SSD_report.pdf",
    )

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig_profile)
        pdf.savefig(fig_heatmap)

        fig_text, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            str(metrics["report_text"]),
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
        )
        fig_text.tight_layout()
        pdf.savefig(fig_text)
        plt.close(fig_text)

    print(f"PDF report saved to: {pdf_path}")

    # Close figures after they have been written to PNG and PDF.
    plt.close(fig_profile)
    plt.close(fig_heatmap)

    print(np.unique(np.diff(x_positions)))
    print(np.unique(np.diff(y_positions)))

if __name__ == "__main__":
    main()
