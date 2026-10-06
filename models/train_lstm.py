"""
Metrodorf v2 — Day 4
LSTM challenger to LightGBM.

Sequence: 5 previous stations per train.
Architecture: 2-layer LSTM (64 units) + Dense.
Loss: BCE with class weights.
Budget: max 3 configurations. Stopping rule: if F1 doesn't beat 0.92 by config 3, stop.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import json

# ---- Reproducibility ----
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ---- Load ----
print("Loading data...")
df = pd.read_parquet("data/processed/ic16_train_2023_2024.parquet")
df = df.sort_values(['TRAIN_NO', 'planned_dt']).reset_index(drop=True)

df['hour'] = df['planned_dt'].dt.hour
df['day_of_week'] = df['planned_dt'].dt.dayofweek
df['target'] = (df['DELAY_ARR'] >= 360).astype(int)
df['year'] = df['planned_dt'].dt.year

df['prev_delay_arr'] = df['prev_delay_arr'].fillna(0).clip(-3600, 36000)
df['prev_delay_dep'] = df['prev_delay_dep'].fillna(0).clip(-3600, 36000)
df['DELAY_ARR'] = df['DELAY_ARR'].clip(-3600, 36000)
df['DELAY_DEP'] = df['DELAY_DEP'].fillna(0).clip(-3600, 36000)

# Normalize features
for col in ['DELAY_ARR', 'DELAY_DEP', 'prev_delay_arr', 'prev_delay_dep']:
    df[col] = df[col] / 3600.0  # normalize by 1 hour

# ---- Build sequences ----
SEQ_LEN = 5
print(f"Building sequences (len={SEQ_LEN})...")

sequences = []
targets = []
years = []

# Group by train
for train_no, group in df.groupby('TRAIN_NO'):
    group = group.sort_values('planned_dt').reset_index(drop=True)
    feats = group[['DELAY_ARR', 'DELAY_DEP', 'prev_delay_arr', 'prev_delay_dep', 'hour']].values
    targs = group['target'].values
    yrs = group['year'].values
    
    for i in range(SEQ_LEN, len(group)):
        seq = feats[i-SEQ_LEN:i]
        sequences.append(seq)
        targets.append(targs[i])
        years.append(yrs[i])

X = np.array(sequences, dtype=np.float32)
y = np.array(targets, dtype=np.float32)
years = np.array(years)

print(f"Sequences: {X.shape}, Targets: {y.shape}")

# Split
train_mask = (years == 2023)
val_mask = (years == 2024)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"Train positive rate: {y_train.mean()*100:.2f}%")
print(f"Val positive rate: {y_val.mean()*100:.2f}%")

# ---- Dataset ----
class DelayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_ds = DelayDataset(X_train, y_train)
val_ds = DelayDataset(X_val, y_val)

# ---- Model ----
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out).squeeze(-1)

# ---- Training ----
def train_model(config):
    print(f"\n--- Config: {config} ---")
    model = LSTMModel(
        input_size=X.shape[2],
        hidden=config['hidden'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    ).to(DEVICE)
    
    # Class weight
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
    
    best_val_f1 = 0
    patience = 3
    no_improve = 0
    
    t0 = time.time()
    for epoch in range(config['max_epochs']):
        model.train()
        train_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_probs = []
        with torch.no_grad():
            for Xb, _ in val_loader:
                Xb = Xb.to(DEVICE)
                out = torch.sigmoid(model(Xb))
                val_probs.append(out.cpu().numpy())
        val_probs = np.concatenate(val_probs)
        
        # Best threshold
        best_f1_e = 0
        for thr in np.arange(0.1, 0.9, 0.05):
            y_pred = (val_probs >= thr).astype(int)
            f1 = f1_score(y_val, y_pred)
            if f1 > best_f1_e:
                best_f1_e = f1
        
        print(f"  Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f} · val_F1={best_f1_e:.4f} · elapsed={time.time()-t0:.1f}s")
        
        if best_f1_e > best_val_f1:
            best_val_f1 = best_f1_e
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    return best_val_f1, model

# ---- Grid search — max 3 configs ----
configs = [
    {'hidden': 64, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001, 'batch_size': 512, 'max_epochs': 10},
    {'hidden': 128, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.001, 'batch_size': 512, 'max_epochs': 10},
    {'hidden': 64, 'num_layers': 1, 'dropout': 0.2, 'lr': 0.0005, 'batch_size': 1024, 'max_epochs': 10},
]

results = []
for cfg in configs:
    f1, model = train_model(cfg)
    results.append({'config': cfg, 'val_f1': f1})
    if f1 > 0.92:
        print(f"\n✅ LSTM beats 0.92 — stopping rule satisfied")
        break

print(f"\n{'='*60}")
print("LSTM Results")
print(f"{'='*60}")
for r in results:
    print(f"  {r['config']} -> F1={r['val_f1']:.4f}")

# Best config
best = max(results, key=lambda x: x['val_f1'])
print(f"\nBest Val F1: {best['val_f1']:.4f}")

with open("models/saved/lstm_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved lstm_results.json")
