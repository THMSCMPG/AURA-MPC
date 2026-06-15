# `data/sandia_validation/`

Sandia National Laboratories validation inputs for paper-scenario
regression.

## Files

| File | Purpose | Columns | Units |
|------|---------|---------|-------|
| `pvmc_validation_template.csv` | 1 000-row time-series template used by the Day-9 orchestrator and retained unchanged for Day 10. | `timestamp, G_poa, T_amb, T_module, WS, ird` | ISO-8601, W/m², °C, °C, m/s, W/m² |
| `IEC61853_1-2.xlsx` | IEC 61853-1/2 reference characterisation workbook (Sandia distribution). | XLSX | – |
| `pvmc_validation_source.txt` | Provenance pointers. | – | – |

## Provenance

* **Standard:** Sandia PV Module Characterisation PVMC programme; IEC 61853.
* **Distribution:** Sandia National Laboratories public datasets (see
  `Sandia_National_Laboratories_Datasets/` one level up).
* **Day-10 status:** this file was populated in Day 9 and is retained
  unchanged in Day 10. Verification command:
  `cksum data/sandia_validation/pvmc_validation_template.csv`.
* **Last verified:** 2026-04-17.
