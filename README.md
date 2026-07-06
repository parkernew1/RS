# RS Project

Analysis scripts and data for comparing film, RayStation DICOM/CSV exports, and
OCTAVIUS MCC detector measurements.

## Code

Project Python scripts live in [`python/`](python/). See
[`python/README.md`](python/README.md) for the current script inventory and the
remaining gaps from `list_of_scripts.txt`.

## Data

- `Octavius_Raystation_comparison_copper/All_CSV/`: RayStation profile CSV exports.
- `Octavius_Raystation_comparison_copper/All_MCC/`: OCTAVIUS MCC measurements.
- `Octavius_Raystation_comparison_copper/comparison_results/`: normalized profile
  comparison outputs.
- `Octavius_Raystation_comparison_copper/raw_gy_comparison_results/`: raw Gy
  profile comparison outputs.
- `octavius_1500_copper_crossplane_profiles/`: generated MCC profile CSVs and QA
  outputs.

Local duplicate OCTAVIUS exports (`Octavius_1500_copper/` and
`Octavius_1500_copper.zip`) are ignored because the tracked MCC data already lives
under `Octavius_Raystation_comparison_copper/All_MCC/`.
