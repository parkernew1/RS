# Python Script Inventory

This folder contains the Python code needed for the current analysis work.

| Needed script | Current status | File |
| --- | --- | --- |
| Film scan to dose profiles | Present | `film_scan_profile_analysis.py` |
| Film scan to PDDs | Present | `film_scan_pdd_analysis.py` |
| DICOM dose file to dose profile CSV | Present | `dicom_dose_profile_to_csv.py` |
| DICOM dose file to PDD CSV | Present | `dicom_dose_pdd_to_csv.py` |
| OCTAVIUS MCC file to dose profile CSV | Present | `octavius_mcc_profile_to_csv.py` |
| Measured-vs-computed profile comparison | Present | `profile_comparison.py` |
| Measured-vs-computed PDD comparison | Missing dedicated script | Not present |

Additional helper:

- `film_calibration.py`: shared 6 MeV film calibration curve loading,
  interpolation, and calibration output helpers.
- `raw_dose_profile_comparison.py`: compares paired RayStation CSV and OCTAVIUS
  MCC profiles in absolute Gy, without dose normalization.
