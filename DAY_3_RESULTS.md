# Day 3 — LightGBM (2023-2024)

## Setup
- Train: 2023 (542,211 rows)
- Validation: 2024 (525,605 rows)
- Features: hour, day_of_week, month, is_weekend, is_stop, prev_delay_arr, prev_delay_dep, station_enc
- Target: DELAY_ARR >= 360 seconds

## Leakage fix
- Removed DELAY_DEP (measured at same station as target).
- Added prev_delay_arr, prev_delay_dep (previous station's delay — NOT leakage).

## Empirical validation of prev_delay
- Correlation prev_delay_dep vs DELAY_ARR: 0.9636
- Correlation prev_delay_arr vs DELAY_ARR: 0.9437
- P(delayed | prev NOT delayed): 1.74%
- P(delayed | prev delayed): 89.81%
- Ratio: 51.66x
- Mutual Information: 0.2922

## Baseline vs Model (Validation 2024)
| Model | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (persistence t-1) | 0.8948 | 0.8948 | 0.8948 |
| LightGBM | 0.9393 | 0.9063 | **0.9225** |
| Improvement | +4.45 pp | +1.15 pp | **+2.77 pp** |

## Confusion Matrix (LightGBM, Validation 2024)
- TP=62,864 · FN=6,502 · FP=4,059 · TN=452,180

## Overfitting check
- Train F1: 0.9474
- Val F1: 0.9225
- Gap: 0.0249 (healthy — <0.05)

## Feature Importance
| Feature | Importance |
|---|---|
| station_enc | 12,898 |
| prev_delay_arr | 12,628 |
| prev_delay_dep | 12,185 |
| hour | 9,334 |
| month | 7,870 |
| day_of_week | 5,542 |
| is_stop | 1,372 |
| is_weekend | 47 |

## Example (25.09.2024, hour=8)
- Train 2128 at BRUSSEL-CENTRAAL, planned 08:23:00
- Actual 08:29:36, DELAY=396s (6.6 min)
- Model prob=0.9410 → predicted DELAYED
- Actual: DELAYED → TRUE POSITIVE ✅

## Next
- Day 4: final evaluation on January 2025 test set
