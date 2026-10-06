"""
Metrodorf v2 — Day 3
Baseline (persistence t-1) + LightGBM.

Compares:
  - Baseline: prev_delay_arr >= 360 (F1~0.898)
  - LightGBM: with prev_delay + time + station features
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)
import json
import time
import os

print("Loading data...")
df = pd.read_parquet("data/processed/ic16_train_2023_2024.parquet")
print(f"Shape: {df.shape}")

# Time features
df['hour'] = df['planned_dt'].dt.hour
df['day_of_week'] = df['planned_dt'].dt.dayofweek
df['month'] = df['planned_dt'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Target
df['target'] = (df['DELAY_ARR'] >= 360).astype(int)

# Chronological split
df['year'] = df['planned_dt'].dt.year
train_mask = (df['year'] == 2023)
val_mask = (df['year'] == 2024)

# Features
feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend']

df['is_stop'] = (df['THOP1_COD'] == '=').astype(int)
feature_cols.append('is_stop')

df['prev_delay_arr'] = df['prev_delay_arr'].fillna(0).clip(-3600, 36000)
df['prev_delay_dep'] = df['prev_delay_dep'].fillna(0).clip(-3600, 36000)
feature_cols.append('prev_delay_arr')
feature_cols.append('prev_delay_dep')

df['PTCAR_NO'] = df['PTCAR_NO'].fillna('UNKNOWN').astype(str)
station_mean = df[train_mask].groupby('PTCAR_NO')['target'].mean()
df['station_enc'] = df['PTCAR_NO'].map(station_mean).fillna(0.1)
feature_cols.append('station_enc')

print(f"\nFeatures: {feature_cols}")

df = df.dropna(subset=['hour', 'day_of_week', 'month'])

X_train = df.loc[train_mask, feature_cols]
y_train = df.loc[train_mask, 'target']
X_val = df.loc[val_mask, feature_cols]
y_val = df.loc[val_mask, 'target']

print(f"Train: {len(X_train)} rows ({y_train.mean()*100:.2f}% positive)")
print(f"Val:   {len(X_val)} rows ({y_val.mean()*100:.2f}% positive)")

# Baseline
print("\n" + "=" * 60)
print("BASELINE: persistence t-1 (prev_delay_arr >= 360)")
print("=" * 60)
y_pred_b = (X_val['prev_delay_arr'] >= 360).astype(int)
cm_b = confusion_matrix(y_val, y_pred_b)
tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
p_b = precision_score(y_val, y_pred_b, zero_division=0)
r_b = recall_score(y_val, y_pred_b, zero_division=0)
f_b = f1_score(y_val, y_pred_b, zero_division=0)
print(f"TP={tp_b}, FN={fn_b}, FP={fp_b}, TN={tn_b}")
print(f"Precision: {p_b:.4f}")
print(f"Recall:    {r_b:.4f}")
print(f"F1:        {f_b:.4f}")

# LightGBM
print("\n" + "=" * 60)
print("LightGBM")
print("=" * 60)

t0 = time.time()
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    is_unbalance=True,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='binary_logloss',
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
print(f"Trained in {time.time()-t0:.1f}s")
print(f"Best iteration: {model.best_iteration_}")

y_prob = model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.10, 0.90, 0.05)
best_f1 = 0.0
best_thr = 0.5
for thr in thresholds:
    y_pred = (y_prob >= thr).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"\nBest threshold: {best_thr:.2f} (F1={best_f1:.4f})")

y_pred_lgb = (y_prob >= best_thr).astype(int)
cm = confusion_matrix(y_val, y_pred_lgb)
tn, fp, fn, tp = cm.ravel()
p = precision_score(y_val, y_pred_lgb)
r = recall_score(y_val, y_pred_lgb)
f = f1_score(y_val, y_pred_lgb)

print(f"\nLightGBM (threshold={best_thr:.2f}):")
print(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
print(f"Precision: {p:.4f}")
print(f"Recall:    {r:.4f}")
print(f"F1:        {f:.4f}")

print(f"\nImprovement over baseline:")
print(f"  F1: {f_b:.4f} -> {f:.4f}  ({(f-f_b)*100:+.2f} pp)")
print(f"  Precision: {p_b:.4f} -> {p:.4f}  ({(p-p_b)*100:+.2f} pp)")
print(f"  Recall: {r_b:.4f} -> {r:.4f}  ({(r-r_b)*100:+.2f} pp)")

print("\n" + "=" * 60)
print("Feature Importance")
print("=" * 60)
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False)
print(importance.to_string(index=False))

os.makedirs("models/saved", exist_ok=True)
model.booster_.save_model("models/saved/lgbm_v1.txt")

metrics = {
    'baseline': {
        'tp': int(tp_b), 'fn': int(fn_b), 'fp': int(fp_b), 'tn': int(tn_b),
        'precision': float(p_b), 'recall': float(r_b), 'f1': float(f_b),
    },
    'lightgbm': {
        'threshold': float(best_thr),
        'tp': int(tp), 'fn': int(fn), 'fp': int(fp), 'tn': int(tn),
        'precision': float(p), 'recall': float(r), 'f1': float(f),
        'best_iteration': int(model.best_iteration_),
    },
    'features': feature_cols,
    'train_size': int(len(X_train)),
    'val_size': int(len(X_val)),
}

with open("models/saved/day3_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print("  models/saved/lgbm_v1.txt")
print("  models/saved/day3_metrics.json")
