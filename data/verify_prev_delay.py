"""
Metrodorf v2 — Day 3
Empirical check: does prev_delay predict DELAY_ARR?

NOT an assumption — a measurement from the data.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, mutual_info_score,
)

print("Loading data...")
df = pd.read_parquet("data/processed/ic16_train_2023_2024.parquet")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Filter rows with prev_delay
df_clean = df.dropna(subset=['prev_delay_arr']).copy()
print(f"\nRows with prev_delay_arr: {len(df_clean)}")

# Target
df_clean['target'] = (df_clean['DELAY_ARR'] >= 360).astype(int)
df_clean['prev_delayed'] = (df_clean['prev_delay_arr'] >= 360).astype(int)

# 1. Correlation
print("\n=== Correlation ===")
print(f"prev_delay_arr vs DELAY_ARR: "
      f"{df_clean[['prev_delay_arr', 'DELAY_ARR']].corr().iloc[0,1]:.4f}")
print(f"prev_delay_dep vs DELAY_ARR: "
      f"{df_clean[['prev_delay_dep', 'DELAY_ARR']].corr().iloc[0,1]:.4f}")

# 2. Conditional probability
print("\n=== Conditional Probability ===")
p_not = df_clean[df_clean['prev_delayed'] == 0]['target'].mean()
p_yes = df_clean[df_clean['prev_delayed'] == 1]['target'].mean()
print(f"P(delayed | prev NOT delayed) = {p_not:.4f}")
print(f"P(delayed | prev delayed)     = {p_yes:.4f}")
if p_not > 0:
    print(f"Ratio: {p_yes / p_not:.2f}x")

# 3. Baseline — persistence t-1
print("\n=== Baseline: persistence t-1 ===")
y_true = df_clean['target'].values
y_pred = df_clean['prev_delayed'].values

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")

# 4. Mutual information
mi = mutual_info_score(y_true, y_pred)
print(f"\n=== Mutual Information ===\nMI = {mi:.4f}")
