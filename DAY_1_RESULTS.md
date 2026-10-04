# Day 1 — Data Feasibility Results (04.10.2026)

## Source
- Infrabel Open Data, Monthly raw punctuality data files
- File: Data_raw_punctuality_202501.csv (~321 MB)
- Period: January 2025

## Shape
- 1,965,004 rows × 21 columns
- Note: 8GB RAM machine — read in chunks if needed

## GT
- `DELAY_ARR` (already computed by Infrabel, in seconds)
- NaN: 95,940 (4.9%) — mostly origin stations

## Target
- `DELAY_ARR > 300` seconds (>5 minutes)
- Count: 197,167
- Percent: 10.03%

## DELAY_ARR Distribution
- mean: 119.4 sec
- median: 40 sec
- std: 352.7 sec
- min: -85,789 sec (outlier — 23.8h early)
- max: 26,154 sec (outlier — 7.3h late)

## Top Relations by Delayed Count
| RELATION | Total | Delayed | % |
|---|---|---|---|
| P | 81,835 | 9,801 | 12.0% |
| EURST | 22,180 | 8,990 | 40.5% |
| IC 16-1 | 58,554 | 7,277 | 12.4% |
| IC 14 | 35,944 | 7,215 | 20.1% |
| IC 18 | 42,673 | 6,261 | 14.7% |
| IC 25 | 63,644 | 5,211 | 8.2% |

## Decision
- ✅ GT works. Continue with Metrodorf v2.
- Focus line: IC 16-1 (58K records, 12.4% delayed)
- Alternative: IC 14 (35K, 20.1%)
- One month of data is sufficient for training.

## Constraints
- Machine: 8GB RAM
- Strategy: read CSV in chunks (chunksize=100,000)
- Filter to one line early, before loading full dataset
