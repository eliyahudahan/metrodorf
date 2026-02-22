# Summary of TU Darmstadt Research (2023)

## 1. Core Problem
Train delay prediction is complex because delays propagate through networks. Single models (Random Forest, XGBoost, etc.) have limited accuracy and consistency.

## 2. Proposed Solution
**Heterogeneous Ensembles** – combining different types of regression models improves both accuracy and consistency.

## 3. Framework (5 Phases)
- **Data preprocessing** – cleaning, feature extraction, standardization
- **Data partitioning** – train / validation / test splits
- **Modelling** – generate multiple models using different algorithms
- **Model selection** – choose best models based on accuracy + diversity
- **Ensemble building** – combine models using:
  - **Averaging (AE)** – simple mean of all model outputs
  - **Weighted Averaging (WE)** – weights based on model performance (e.g., R²)

## 4. Key Metrics Used
- R², MAE, RMSE, MSE
- % correct prediction after rounding
- % within 1 minute

## 5. Main Findings
- ✅ **Ensembles outperform single models** (higher R², lower error)
- ✅ **Diversity between models is crucial** – models must make different errors
- ✅ **Coincident Failure Diversity (CFD)** – best measure for regression diversity
- ✅ **Deep Learning ensembles** (Tabnet + CNN) also improved, but ML ensembles were often better on tabular data

## 6. Why It Matters for Metrodorf
This research proves that combining multiple models (like Gaussian Process + XGBoost + Random Forest) in a **weighted ensemble** can significantly improve delay prediction accuracy – exactly our approach for Rhine-Ruhr.
