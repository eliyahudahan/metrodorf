# Day 2 — Training Data Distribution (05.10.2026)

## Source
- Infrabel Open Data, Monthly raw punctuality data files
- Period: January 2023 – December 2024 (24 months)
- Files: 24 CSVs, ~7.7 GB total
- License: CC0

## Train Shape
- 45,582,334 rows × 3 columns (DELAY_ARR, RELATION, THOP1_COD)

## DELAY_ARR Distribution (2023-2024)
| Quantile | Value (seconds) |
|---|---|
| P50 | 52 |
| P75 | 167 |
| P90 | 390 |
| P95 | 642 |

## Prevalence by Threshold (2023-2024)
| Threshold | Percent |
|---|---|
| ≥180s (3m00s) | 22.33% |
| ≥300s (5m00s) | 13.19% |
| **≥360s (6m00s)** | **10.56%** |
| ≥420s (7m00s) | 8.65% |

## Target Decision
- Primary target: **≥360 seconds**
- Anchored in Infrabel KPI P.II1 (6 minutes at 111 measuring points)
- Empirical prevalence: 10.56% — a stable class balance for binary classification
- Threshold defined **before** model evaluation — not optimized against F1

## Line Selected
- **IC 16-1**
- Train rows: 1,067,816
- Delayed (≥360s): 155,545 (14.57%)

## Test (untouched)
- January 2025
- 1,965,004 rows (58,554 for IC 16-1)

## Artifacts
- `data/processed/ic16_train_2023_2024.parquet`
- `data/process_train.py`
