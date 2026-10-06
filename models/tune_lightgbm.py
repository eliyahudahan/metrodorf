"""
Metrodorf v2 — Day 4
Grid search for LightGBM — fair tuning budget (30 min max).

Compares 6 configurations. Picks best by validation F1.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import time
import itertools

print("Loading data...")
df = pd.read_parquet("data/processed/ic16_train_2023_2024.parquet")
df['hour'] = df['planned_dt'].dt.hour
df['day_of_week'] = df['planned_dt'].dt.dayofweek
df['month'] = df['planned_dt'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_stop'] = (df['THOP1_COD'] == '=').astype(int)
df['prev_delay_arr'] = df['prev_delay_arr'].fillna(0).clip(-3600, 36000)
df['prev_delay_dep'] = df['prev_delay_dep'].fillna(0).clip(-3600, 36000)
df['target'] = (df['DELAY_ARR'] >= 360).astype(int)

df['year'] = df['planned_dt'].dt.year
train_mask = (df['year'] == 2023)
val_mask = (df['year'] == 2024)

df['PTCAR_NO'] = df['PTCAR_NO'].fillna('UNKNOWN').astype(str)
station_mean = df[train_mask].groupby('PTCAR_NO')['target'].mean()
df['station_enc'] = df['PTCAR_NO'].map(station_mean).fillna(0.1)

feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend',
                'is_stop', 'prev_delay_arr', 'prev_delay_dep', 'station_enc']

X_train = df.loc[train_mask, feature_cols].fillna(0)
y_train = df.loc[train_mask, 'target']
X_val = df.loc[val_mask, feature_cols].fillna(0)
y_val = df.loc[val_mask, 'target']

print(f"Train: {len(X_train)} · Val: {len(X_val)}")

# --- Grid search ---
configs = [
    {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 20},
    {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 63, 'min_child_samples': 50},
    {'n_estimators': 1000, 'learning_rate': 0.03, 'num_leaves': 63, 'min_child_samples': 50},
    {'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 100},
    {'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 63, 'min_child_samples': 50},
    {'n_estimators': 800, 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 30},
]

results = []
t0 = time.time()
for i, cfg in enumerate(configs):
    print(f"\n--- Config {i+1}/{len(configs)}: {cfg} ---")
    model = lgb.LGBMClassifier(
        **cfg,
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
    
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # Find best threshold
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    
    y_pred = (y_prob >= best_thr).astype(int)
    p = precision_score(y_val, y_pred)
    r = recall_score(y_val, y_pred)
    
    print(f"  F1={best_f1:.4f} · P={p:.4f} · R={r:.4f} · thr={best_thr:.2f} · iter={model.best_iteration_}")
    
    results.append({
        'config': cfg,
        'f1': best_f1,
        'precision': p,
        'recall': r,
        'threshold': best_thr,
        'best_iteration': int(model.best_iteration_),
    })

print(f"\nTotal tuning time: {time.time()-t0:.1f}s")

# Best config
results.sort(key=lambda x: x['f1'], reverse=True)
print("\n" + "="*60)
print("BEST CONFIG:")
print("="*60)
print(json.dumps(results[0], indent=2))

# Save
with open("models/saved/tuning_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved tuning_results.json")
