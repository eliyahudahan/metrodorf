# Day 4 — LSTM Challenger vs LightGBM

## Setup
- Train: 2023, Validation: 2024, Test: Jan 2025 (untouched)
- LSTM: sequence length = 5 previous stations per train
- Features per step: DELAY_ARR, DELAY_DEP, prev_delay_arr, prev_delay_dep, hour

## LSTM Results (Validation 2024)
| Config | Hidden | Layers | F1 |
|---|---|---|---|
| 1 | 64 | 2 | 0.9107 |
| 2 | 128 | 2 | 0.9110 |
| 3 | 64 | 1 | 0.8989 |

Best: F1 = 0.9110

## Comparison
| Model | Val F1 (2024) | Test F1 (Jan 2025) |
|---|---|---|
| Baseline (persistence t-1) | 0.8948 | 0.8886 |
| LightGBM | **0.9228** | **0.9252** |
| LSTM (best) | 0.9110 | not run |

## Decision
**LightGBM is the primary model.**

Rationale:
- LightGBM F1 (0.9228) > LSTM F1 (0.9110) — +1.18 pp on validation.
- Tabular data + engineered features (`prev_delay`) → tree-based models are the standard.
- LSTM tried to learn the sequential dependency from scratch, but it was already captured by `prev_delay`.
- LightGBM is faster to train, faster to infer, and simpler to deploy.

## What this teaches
- Time (hour, day) is a feature, not a sequence.
- Feature engineering beats representation learning when data is tabular.
- The literature (LSTM is good for sequential tasks) was a hypothesis.
  The data was the judge.

## Next
- Dashboard + Deploy + Map (Day 5)
- README with 3 Nir Etzion questions
