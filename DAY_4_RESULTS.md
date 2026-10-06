# Day 4 — Final Evaluation on Test Set (January 2025)

## Setup
- Train: 2023 (542,211 rows)
- Validation: 2024 (525,605 rows)
- Test: January 2025 (46,574 rows of line IC 16-1) — untouched until now

## Test Set Results — LightGBM
- Rows: 46,574
- TP = 4,347 (predicted delayed, actually delayed)
- FN = 456 (predicted on-time, actually delayed)
- FP = 247 (predicted delayed, actually on-time)
- TN = 41,524 (predicted on-time, actually on-time)
- Precision: 0.9462
- Recall: 0.9051
- **F1: 0.9252**

## Baseline (persistence t-1)
- TP = 4,264 · FN = 539 · FP = 530 · TN = 41,241
- Precision: 0.8894
- Recall: 0.8878
- F1: 0.8886

## Improvement
- F1: 0.8886 -> 0.9252 (+3.66 pp)

## Cross-dataset consistency
| Dataset | Period | F1 |
|---|---|---|
| Train | 2023 | 0.9474 |
| Validation | 2024 | 0.9225 |
| Test | Jan 2025 | 0.9252 |

Test F1 ≈ Validation F1 -> no overfitting. Model generalizes.

## Next
- Challenger: LSTM (limited budget, stopping rule)
- Dashboard + Deploy + Map
