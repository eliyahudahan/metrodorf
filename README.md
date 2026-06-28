# Metrodorf – Train Delay Prediction for Rhine-Ruhr

Real-time delay prediction for polycentric regional rail networks, built for a dispatcher managing connections at Dortmund Hbf.

---

## Problem

Polycentric regions like Rhine-Ruhr have dense, interdependent rail traffic. A delay at one station cascades into missed connections elsewhere. Dispatchers need early signal, not just live status.

---

## Research Basis

Built on three core papers:

- **Al Ghamdi (2022)** – Heterogeneous Ensembles for Regression → Weighted averaging (R²), optimal ensemble size 3–4
- **Rondini et al. (2025)** – Power Laws in Railway Delays → Heavy tails, Laplacian noise, exponential distribution for German high-speed
- **Kämpere & Alsahag (2025)** – Network Features for Delay Prediction (UvA) → Baseline 0.65 balanced accuracy, 21% threshold

---

## Results

| Model | R² | Ensemble Weight |
|-------|-----|-----------------|
| Gaussian | 0.167 | 47.4% |
| XGBoost | 0.075 | 31.3% |
| Random Forest | 0.051 | 21.2% |
| **Ensemble** | **0.145** | 100% |

- MAE: 3.35 minutes
- Beats UvA baseline

---

## Decision Logic

| Delay | Action |
|-------|--------|
| >10 min | Hold connections |
| 5–10 min | Monitor |
| <5 min | Normal |

---

## Stack

Streamlit, scikit-learn (Random Forest, XGBoost, Gaussian Processes), Plotly

---

## Author

Eliyahu Dahan