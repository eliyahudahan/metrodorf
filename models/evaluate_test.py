"""
Metrodorf v2 — Day 4
Final evaluation on January 2025 test set (untouched).

This is the first and only time the model sees the test set.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)
import json
import glob

# ---- Load the test set (January 2025) ----
print("Loading January 2025 test set...")

# The test file is Data_raw_punctuality_202501.csv (already in data/raw/)
test_file = "data/raw/Data_raw_punctuality_202501.csv"

chunks = []
for chunk in pd.read_csv(test_file, chunksize=100_000, usecols=[
    'DATDEP', 'TRAIN_NO', 'RELATION', 'PTCAR_NO', 'THOP1_COD',
    'PTCAR_LG_NM_NL', 'LINE_NO_DEP', 'LINE_NO_ARR',
    'REAL_TIME_ARR', 'PLANNED_TIME_ARR', 'DELAY_ARR',
    'REAL_DATE_ARR', 'PLANNED_DATE_ARR',
    'REAL_TIME_DEP', 'PLANNED_TIME_DEP', 'DELAY_DEP',
    'REAL_DATE_DEP', 'PLANNED_DATE_DEP',
]):
    chunk = chunk[chunk['RELATION'] == 'IC 16-1']
    chunk = chunk[chunk['THOP1_COD'].isin(['D', '='])]
    chunk = chunk[chunk['DELAY_ARR'].between(-3600, 36000)]
    if len(chunk) > 0:
        chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print(f"Test rows (IC 16-1): {len(df)}")

# ---- Rebuild the same features ----
df['planned_dt'] = pd.to_datetime(
    df['PLANNED_DATE_ARR'] + ' ' + df['PLANNED_TIME_ARR'].fillna('00:00:00'),
    format='%d%b%Y %H:%M:%S',
    errors='coerce',
)
df = df.sort_values(['TRAIN_NO', 'planned_dt']).reset_index(drop=True)

df['prev_delay_arr'] = df.groupby('TRAIN_NO')['DELAY_ARR'].shift(1)
df['prev_delay_dep'] = df.groupby('TRAIN_NO')['DELAY_DEP'].shift(1)

df['hour'] = df['planned_dt'].dt.hour
df['day_of_week'] = df['planned_dt'].dt.dayofweek
df['month'] = df['planned_dt'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_stop'] = (df['THOP1_COD'] == '=').astype(int)
df['target'] = (df['DELAY_ARR'] >= 360).astype(int)

df['prev_delay_arr'] = df['prev_delay_arr'].fillna(0).clip(-3600, 36000)
df['prev_delay_dep'] = df['prev_delay_dep'].fillna(0).clip(-3600, 36000)

# station_enc — use the SAME encoding computed from train (2023)
# Load the station means from training period
train_files = sorted(glob.glob("data/raw/infrabel_2023-*.csv"))
chunks = []
for f in train_files:
    for chunk in pd.read_csv(f, chunksize=100_000, usecols=[
        'RELATION', 'THOP1_COD', 'PTCAR_NO', 'DELAY_ARR'
    ]):
        chunk = chunk[chunk['RELATION'] == 'IC 16-1']
        chunk = chunk[chunk['THOP1_COD'].isin(['D', '='])]
        chunk = chunk[chunk['DELAY_ARR'].between(-3600, 36000)]
        if len(chunk) > 0:
            chunks.append(chunk)

train_df = pd.concat(chunks, ignore_index=True)
train_df['target'] = (train_df['DELAY_ARR'] >= 360).astype(int)
train_df['PTCAR_NO'] = train_df['PTCAR_NO'].fillna('UNKNOWN').astype(str)
station_mean = train_df.groupby('PTCAR_NO')['target'].mean()

df['PTCAR_NO'] = df['PTCAR_NO'].fillna('UNKNOWN').astype(str)
df['station_enc'] = df['PTCAR_NO'].map(station_mean).fillna(0.1)

# ---- Predict ----
booster = lgb.Booster(model_file='models/saved/lgbm_v1.txt')
feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend',
                'is_stop', 'prev_delay_arr', 'prev_delay_dep', 'station_enc']

X_test = df[feature_cols].fillna(0)
y_test = df['target']

y_prob = booster.predict(X_test)
y_pred = (y_prob >= 0.80).astype(int)

# ---- Metrics ----
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
p = precision_score(y_test, y_pred)
r = recall_score(y_test, y_pred)
f = f1_score(y_test, y_pred)

print("\n" + "="*60)
print("TEST SET EVALUATION — January 2025")
print("="*60)
print(f"Rows: {len(y_test)}")
print(f"\nTP={tp}, FN={fn}, FP={fp}, TN={tn}")
print(f"Precision: {p:.4f}")
print(f"Recall:    {r:.4f}")
print(f"F1:        {f:.4f}")

# Baseline
y_pred_b = (X_test['prev_delay_arr'] >= 360).astype(int)
cm_b = confusion_matrix(y_test, y_pred_b)
tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
p_b = precision_score(y_test, y_pred_b, zero_division=0)
r_b = recall_score(y_test, y_pred_b, zero_division=0)
f_b = f1_score(y_test, y_pred_b, zero_division=0)

print(f"\nBaseline (persistence t-1):")
print(f"TP={tp_b}, FN={fn_b}, FP={fp_b}, TN={tn_b}")
print(f"Precision: {p_b:.4f}")
print(f"Recall:    {r_b:.4f}")
print(f"F1:        {f_b:.4f}")

print(f"\nImprovement:")
print(f"  F1: {f_b:.4f} -> {f:.4f}  ({(f-f_b)*100:+.2f} pp)")

# Save
results = {
    'test_rows': int(len(y_test)),
    'lightgbm': {'tp': int(tp), 'fn': int(fn), 'fp': int(fp), 'tn': int(tn),
                 'precision': float(p), 'recall': float(r), 'f1': float(f)},
    'baseline': {'tp': int(tp_b), 'fn': int(fn_b), 'fp': int(fp_b), 'tn': int(tn_b),
                 'precision': float(p_b), 'recall': float(r_b), 'f1': float(f_b)},
}
with open("models/saved/test_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved test_metrics.json")
