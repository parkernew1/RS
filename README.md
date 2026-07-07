# RS Project

Analysis scripts and data for comparing film, RayStation DICOM/CSV exports, and
OCTAVIUS MCC detector measurements.

## Code

Project Python scripts live in [`python/`](python/). See
[`python/README.md`](python/README.md) for the current script inventory and the
remaining gaps from `list_of_scripts.txt`.

## Data

- `Actual Runs/`: manuscript measurement workspace.
  - `6MeV_calibration_films_07062026/`: 0, 50, 100, 150, 200, 250, and
    300 cGy scanned calibration films for the current 6 MeV film workflow.
  - `Profiles/`: profile DICOM exports, scan files, converted CSVs, and results.
  - `PDDs/`: PDD DICOM exports, scan files, converted CSVs, and results.
  - `Output factors/`: manuscript output-factor measurements.
  - `measurement_list_manuscript.xlsx`: manuscript measurement tracking workbook.
- `Trial Runs/`: archived trial/antiquated measurement work, including the prior
  OCTAVIUS/RayStation comparison data and generated outputs.

Local duplicate OCTAVIUS exports at the repo root (`Octavius_1500_copper/` and
`Octavius_1500_copper.zip`) are ignored. The archived copies live under
`Trial Runs/`.
