"""
Metrodorf v2 — Show a real example:
25.09.2024, ~08:00, line IC 16-1.

What did the model predict?
What actually happened?
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import json

# ---- Load ----
print("Loading data...")
df = pd.read_parquet("data/processed/ic16_train_2023_2024.parquet")

# ---- Rebuild features (same as training) ----
df['hour'] = df['planned_dt'].dt.hour
df['day_of_week'] = df['planned_dt'].dt.dayofweek
df['month'] = df['planned_dt'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_stop'] = (df['THOP1_COD'] == '=').astype(int)
df['prev_delay_arr'] = df['prev_delay_arr'].fillna(0).clip(-3600, 36000)
df['prev_delay_dep'] = df['prev_delay_dep'].fillna(0).clip(-3600, 36000)

df['year'] = df['planned_dt'].dt.year
train_mask = (df['year'] == 2023)

df['PTCAR_NO'] = df['PTCAR_NO'].fillna('UNKNOWN').astype(str)
station_mean = df[train_mask].groupby('PTCAR_NO')['target'].mean() if 'target' in df.columns else None

# Build target
df['target'] = (df['DELAY_ARR'] >= 360).astype(int)
station_mean = df[train_mask].groupby('PTCAR_NO')['target'].mean()
df['station_enc'] = df['PTCAR_NO'].map(station_mean).fillna(0.1)

feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend',
                'is_stop', 'prev_delay_arr', 'prev_delay_dep', 'station_enc']

# ---- Find the specific row: 25.09.2024, ~08:00 ----
target_date = pd.Timestamp('2024-09-25')
mask = (
    (df['planned_dt'].dt.date == target_date.date()) &
    (df['hour'] == 8)
)
example_df = df[mask].sort_values('planned_dt')

print(f"\n=== Found {len(example_df)} records on {target_date.date()}, hour=8 ===")
print("\nAll rows:")
print(example_df[[
    'TRAIN_NO', 'PTCAR_LG_NM_NL', 'planned_dt',
    'PLANNED_TIME_ARR', 'REAL_TIME_ARR', 'DELAY_ARR',
    'prev_delay_arr', 'prev_delay_dep', 'target'
]].to_string(index=False))

# ---- Load the trained model ----
print("\nLoading trained model...")
booster = lgb.Booster(model_file='models/saved/lgbm_v1.txt')

# ---- Predict ----
X = example_df[feature_cols].fillna(0)
y_prob = booster.predict(X)
y_pred = (y_prob >= 0.80).astype(int)

# ---- Show the result ----
print("\n" + "=" * 70)
print(f"PREDICTION vs ACTUAL — 25.09.2024, hour=8")
print("=" * 70)

for i, (idx, row) in enumerate(example_df.iterrows()):
    print(f"\n--- Train {row['TRAIN_NO']} at {row['PTCAR_LG_NM_NL']} ---")
    print(f"  Planned arrival:   {row['PLANNED_TIME_ARR']}")
    print(f"  Actual arrival:    {row['REAL_TIME_ARR']}")
    print(f"  DELAY_ARR:         {row['DELAY_ARR']} seconds ({row['DELAY_ARR']/60:.1f} min)")
    print(f"  prev_delay_arr:    {row['prev_delay_arr']} seconds")
    print(f"  prev_delay_dep:    {row['prev_delay_dep']} seconds")
    print(f"  Model probability: {y_prob[i]:.4f}")
    print(f"  Model prediction:  {'DELAYED (>=6 min)' if y_pred[i]==1 else 'ON TIME'}")
    print(f"  Actual outcome:    {'DELAYED (>=6 min)' if row['target']==1 else 'ON TIME'}")

    # Confusion classification
    if y_pred[i] == 1 and row['target'] == 1:
        verdict = "TRUE POSITIVE ✅"
    elif y_pred[i] == 1 and row['target'] == 0:
        verdict = "FALSE POSITIVE ❌"
    elif y_pred[i] == 0 and row['target'] == 1:
        verdict = "FALSE NEGATIVE ❌"
    else:
        verdict = "TRUE NEGATIVE ✅"
    print(f"  Verdict:           {verdict}")
