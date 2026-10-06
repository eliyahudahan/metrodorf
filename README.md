# Metrodorf v2 — Train Delay Prediction (Infrabel, Belgium)

> **Portfolio project** — *not deployed in production.*

**Live dashboard:** [metrodorf-w98cpew9yrmcyxhetqgntc.streamlit.app](https://metrodorf-w98cpew9yrmcyxhetqgntc.streamlit.app)

---

## 1. What is the problem?

**Question:** *Will a train on line IC 16-1 arrive **more than 6 minutes late**
(≥360 seconds) at a given station, on a given day and hour?*

**Target audience:** A rail analyst at a European railway operator.

**Why this matters:**
- Trains delayed by ≥6 minutes disrupt passenger connections.
- Rail analysts need advance warning to plan rolling stock and operational flexibility.
- A delay prediction model gives the analyst time to react — reallocate trains, notify staff, adjust schedules.

---

## 2. Data

| Field | Value |
|---|---|
| **Source** | [Infrabel Open Data](https://opendata.infrabel.be/) |
| **Dataset** | Monthly raw punctuality data files |
| **License** | CC0 (public domain) |
| **Period (train)** | January 2023 – December 2024 |
| **Period (test)** | January 2025 (untouched until final evaluation) |
| **Scale** | 45.6M rows total · 1.07M rows for line IC 16-1 |
| **Resolution** | 1 second |
| **Line** | IC 16-1 (Brussels ↔ Luxembourg corridor) |

### Ground Truth

- **Field:** `DELAY_ARR` — already computed by Infrabel, in seconds.
- **Definition:** `REAL_TIME_ARR − PLANNED_TIME_ARR`.
- **Target:** `DELAY_ARR >= 360` seconds (6 minutes).
- **Why 360:** Aligns with Infrabel's official KPI **P.II1** — *"National passenger punctuality rate at 6 minutes at 111 measuring points."*

> **Note:** Infrabel's official metric measures at the terminus station (and at the first station after the Brussels North-South junction). This project measures delay at **every station observation** on the line. It is **not** a reproduction of the official metric.

---

## 3. Method

### 3.1 Baseline

**Persistence t-1:** predict "delayed" if the same train's *previous* station was delayed ≥360 seconds.

This is a strong baseline — and it must be beaten by the model, not just by a random guess.

### 3.2 Features (8)

| # | Feature | Description |
|---|---|---|
| 1 | `hour` | hour of planned arrival |
| 2 | `day_of_week` | 0 = Monday, 6 = Sunday |
| 3 | `month` | 1–12 |
| 4 | `is_weekend` | binary |
| 5 | `is_stop` | stop type (`=` vs `D`) |
| 6 | **`prev_delay_arr`** | previous station's arrival delay |
| 7 | **`prev_delay_dep`** | previous station's departure delay |
| 8 | `station_enc` | station target encoding (train only) |

### 3.3 Models

- **Primary:** LightGBM.
- **Challenger:** LSTM (sequence of 5 previous stations).
- **Threshold:** 0.80 (selected on validation set, not on test).

### 3.4 Empirical validation of `prev_delay`

The choice of `prev_delay` as a feature is **not an assumption** — it was measured on the training data:

| Metric | Value |
|---|---|
| Correlation `prev_delay_dep` vs `DELAY_ARR` | **0.9636** |
| Correlation `prev_delay_arr` vs `DELAY_ARR` | **0.9437** |
| P(delayed \| prev NOT delayed) | **1.74%** |
| P(delayed \| prev delayed) | **89.81%** |
| Ratio | **51.66×** |
| Mutual Information | **0.2922** |

### 3.5 Leakage detection

An initial run included `DELAY_DEP` (the departure delay at the *same* station) as a feature.
It was removed because it is measured **after** the target, and it leaked the outcome
(F1 was implausibly high at 0.97). The corrected feature is `prev_delay_arr/dep` —
the previous station's delay, which is measured **before** the prediction is made.

---

## 4. Results

### 4.1 Validation (2024)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (persistence t-1) | 0.8948 | 0.8948 | 0.8948 |
| **LightGBM** | **0.9393** | **0.9063** | **0.9225** |
| Improvement | +4.45 pp | +1.15 pp | **+2.77 pp** |

### 4.2 Test (January 2025 — untouched until final evaluation)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (persistence t-1) | 0.8894 | 0.8878 | 0.8886 |
| **LightGBM** | **0.9462** | **0.9051** | **0.9252** |
| Improvement | +5.68 pp | +1.73 pp | **+3.66 pp** |

**Confusion matrix (Test, LightGBM):**
- TP = 4,347 · FN = 456 · FP = 247 · TN = 41,524

### 4.3 Overfitting check

| Dataset | F1 |
|---|---|
| Train (2023) | 0.9474 |
| Validation (2024) | 0.9225 |
| **Test (Jan 2025)** | **0.9252** |
| **Gap (Train − Test)** | **0.0222** (healthy — <0.05) |

### 4.4 Challenger — LSTM

| Config | Hidden | Layers | Val F1 |
|---|---|---|---|
| 1 | 64 | 2 | 0.9107 |
| 2 | 128 | 2 | 0.9110 |
| 3 | 64 | 1 | 0.8989 |

**Best LSTM: 0.9110.** **LightGBM wins: 0.9225 (Val), 0.9252 (Test).**

**Decision:** LightGBM is the primary model. The data is tabular, and the engineered features
(`prev_delay`) already capture the sequential dependency. The LSTM tried to learn the same
relationship from raw sequences, but with lower accuracy — the literature (LSTM is effective
for sequential tasks) was a hypothesis; the data was the judge.

---

## 5. Prediction vs Reality — Real Examples

Examples from **25.09.2024, hour 8:00** (line IC 16-1):

| Train | Station | Planned | Actual | Delay (s) | Model prob. | Predicted | Actual | Verdict |
|---|---|---|---|---|---|---|---|---|
| 2128 | BRUSSEL-CENTRAAL | 08:23:00 | 08:29:36 | 396 | 0.9410 | DELAYED | DELAYED | TP ✅ |
| 2128 | BRUSSEL-KAPELLEKERK | 08:26:00 | 08:32:00 | 360 | 0.8785 | DELAYED | DELAYED | TP ✅ |
| 2108 | BRUSSEL-LUXEMBURG | 08:54:00 | 09:01:08 | 428 | 0.9982 | DELAYED | DELAYED | TP ✅ |
| 2128 | BRUSSEL-CONGRES | 08:21:00 | 08:27:02 | 362 | 0.6540 | ON TIME | DELAYED | FN ❌ |
| 2131 | ARLON | 08:31:00 | 08:30:08 | −51 | 0.7341 | ON TIME | ON TIME | TN ✅ |
| 2107 | LA HULPE | 08:08:00 | 08:09:04 | 64 | 0.0058 | ON TIME | ON TIME | TN ✅ |

**What this tells us:**
- **TP:** When the previous station was ≥6 min late, the model confidently predicts delay (0.88–0.99).
- **FN:** Train 2128 at BRUSSEL-CONGRES — the previous station was only slightly under the threshold (358s vs 360s), so the model gave 0.65 — just below the 0.80 cutoff. The train arrived 6:02 late.
- **TN with high `prev_delay`:** Train 2131 at ARLON — the previous station was 410s late, but this train recovered and arrived on time. The model **correctly** did not predict a delay.

→ The model is **not** simply copying `prev_delay`. It learned when recovery happens.

---

## 6. What This Project Is

- A demonstration of a complete data science workflow — from raw data to a deployed dashboard.
- A portfolio piece for a rail analyst.
- A documented case study of **leakage detection, baseline comparison, and model choice**.

## 7. What This Project Is Not

- **Not a production system.**
- **Not a real-time dispatcher tool.**
- **Not generalizable** to other lines without retraining.
- **Not including weather, incidents, holidays, or cancellations.**

---

## 8. Limitations

1. **Official metric difference:** Infrabel measures at the terminus + Brussels junction. This project measures at every station observation.
2. **Cancellations not in the dataset** — Infrabel reports them separately.
3. **Single line (IC 16-1)** — not generalized to other lines.
4. **No external features** — weather, incidents, holidays are not used.
5. **Threshold 360s is a domain decision** — anchored in Infrabel's KPI P.II1, not optimized for F1.
6. **Test period:** January 2025 only (one month).
7. **Data drift not evaluated.**

---

## 9. What I learned

- **Empirical validation beats assumption.** The `prev_delay` relationship was measured (correlation = 0.96, P = 89.8%), not assumed.
- **Feature engineering beats representation learning** when data is tabular and small. LightGBM (0.9252) beat LSTM (0.9110) with a fraction of the training time.
- **A proper baseline is essential.** The persistence t-1 baseline achieved F1=0.89 — the model must beat that, not just a random guess.
- **Leakage can look like success.** An early run with `DELAY_DEP` (same-station) achieved F1=0.97 — implausibly high. Removing it dropped the score to the correct level.
- **Test set is sacred.** January 2025 was untouched until the final evaluation.

---

## 10. Repository structure
metrodorf/
├── app.py # Streamlit dashboard (5 tabs)
├── requirements.txt
├── README.md
├── DAY_1_RESULTS.md # Data feasibility
├── DAY_2_RESULTS.md # Distribution + prevalence
├── DAY_3_RESULTS.md # LightGBM training
├── DAY_4_RESULTS.md # Test set evaluation
├── DAY_4_LSTM_RESULTS.md # LSTM challenger
├── data/
│ ├── raw/ # Infrabel CSV (gitignored)
│ ├── processed/ # Parquet (gitignored)
│ ├── process_train.py # Pipeline: raw → parquet
│ ├── verify_prev_delay.py # Empirical validation
│ └── show_example.py # Real examples
├── models/
│ ├── train_lightgbm.py # Primary model
│ ├── tune_lightgbm.py # Grid search
│ ├── train_lstm.py # Challenger
│ ├── evaluate_test.py # Final test evaluation
│ └── saved/ # Artifacts (metrics JSON, model)
├── features/ # Feature engineering
├── evaluation/ # Metrics
└── tests/

text

---

## 11. How to reproduce

```bash
# 1. Clone
git clone https://github.com/eliyahudahan/metrodorf.git
cd metrodorf

# 2. Install
pip install -r requirements.txt

# 3. Download data
# Infrabel Open Data: https://opendata.infrabel.be/
# Download monthly punctuality files for 2023, 2024, and Jan 2025.
# Place them in data/raw/.

# 4. Process + train
python3 data/process_train.py
python3 data/verify_prev_delay.py
python3 models/train_lightgbm.py
python3 models/evaluate_test.py

# 5. Run dashboard
streamlit run app.py
12. License
Data: Infrabel Open Data — CC0 (public domain).

Code: Portfolio project — see LICENSE file.

13. Contact
Eliyahu Dahan

GitHub: @eliyahudahan

LinkedIn: eliyahu-dahan-684b22294
