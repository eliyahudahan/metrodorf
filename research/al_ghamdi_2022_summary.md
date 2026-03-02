# Al Ghamdi (2022) – Heterogeneous Ensembles for Regression

**Title:** "Developing Heterogeneous Ensembles for Regression Problems"  
**Author:** Al Ghamdi, A. (PhD Thesis)  
**Year:** 2022  
**Focus:** Combining different regression models to improve accuracy and consistency.

## Core Problem
Single models or homogeneous ensembles (e.g., Random Forest) have limited accuracy and high variance in complex prediction tasks like train delays.

## Proposed Solution
**Heterogeneous ensembles** – combine models from different families (GBR, EN, KNN, MLP, SGD, XGB, DT, BR, RF, Ridge, LR, Lasso).

## Key Findings
- ✅ **MSM1 (accuracy only)** outperformed MSM2 (accuracy + diversity)
- ✅ **Weighted Averaging (WE)** slightly better than simple averaging (AE)
- ✅ **Optimal ensemble size: 3–4 models**
- ✅ Ensembles outperform single models (including RF and XGBoost)
- ✅ More consistent – lower standard deviation across runs

## Relevance to Metrodorf
Directly justifies our **3-model ensemble** (Gaussian + XGBoost + Random Forest) with **weighted averaging based on R²** – exactly the 0.474, 0.313, 0.212 weights we use.
