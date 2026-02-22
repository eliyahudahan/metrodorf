# Summary of Al Ghamdi (2022) – PhD Thesis

## 1. Core Problem
Train delay prediction is critical but existing methods (single models or homogeneous ensembles like Random Forest) have limited accuracy and consistency.

## 2. Proposed Solution
**Heterogeneous Ensembles** – combining different types of regression models (not just one algorithm) to improve both accuracy and reliability.

## 3. Key Framework (5 Phases)
- **Data preprocessing** – cleaning, feature extraction, standardization
- **Data partitioning** – train (70%) / validation (15%) / test (15%)
- **Modelling** – generate multiple models using different algorithms (GBR, EN, KNN, MLP, SGD, XGB, DT, BR, RF, Ridge, LR, Lasso)
- **Model selection** – two methods:
  - **MSM1** – selects by accuracy only
  - **MSM2** – selects by accuracy + diversity (CFD, COR, COV, DIS)
- **Ensemble building** – combine models using:
  - **Averaging (AE)** – simple mean
  - **Weighted Averaging (WE)** – weights based on R² performance

## 4. Main Findings
- ✅ **MSM1 (accuracy only) outperformed MSM2** – diversity measures didn't help in regression
- ✅ **WE slightly better than AE** – weighting mechanism helps
- ✅ **Optimal ensemble size: 3–4 models** – beyond that, accuracy drops
- ✅ **Ensembles outperform single models** (including RF and XGBoost)
- ✅ **More consistent** – lower standard deviation across runs
- ✅ **Deep Learning (Tabnet + CNN)** also improved with ensembles, but ML ensembles still better on tabular data

## 5. Key Metrics Used
- R², MAE, RMSE, MSE
- % correct prediction after rounding
- Critical Difference (CD) diagrams for statistical comparison

## 6. Why It Matters for Metrodorf
This thesis proves that combining multiple models (like Gaussian Process + XGBoost + Random Forest) in a **weighted ensemble** significantly improves delay prediction – exactly our approach for Rhine-Ruhr.
