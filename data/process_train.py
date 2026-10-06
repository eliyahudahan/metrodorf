"""
Metrodorf v2 — Day 3
Process Infrabel raw data for line IC 16-1 (2023-2024).

Reads: data/raw/infrabel_2023-*.csv, data/raw/infrabel_2024-*.csv
Writes: data/processed/ic16_train_2023_2024.parquet

Features added:
  - planned_dt: planned arrival timestamp
  - prev_delay_arr: previous station's arrival delay (same train)
  - prev_delay_dep: previous station's departure delay (same train)

prev_delay_arr/dep are NOT leakage — they were measured BEFORE the current
station's arrival. This matches Nir Etzion's task: predict delay in advance.

Memory strategy: chunksize=100_000 (for 8GB RAM machines).
"""
import pandas as pd
import glob
import time

LINE = 'IC 16-1'
OUTPUT = "data/processed/ic16_train_2023_2024.parquet"

files = sorted(
    glob.glob("data/raw/infrabel_2023-*.csv")
    + glob.glob("data/raw/infrabel_2024-*.csv")
)
print(f"Files: {len(files)}")

chunks = []
t0 = time.time()
for f in files:
    for chunk in pd.read_csv(
        f,
        chunksize=100_000,
        usecols=[
            'DATDEP', 'TRAIN_NO', 'RELATION', 'PTCAR_NO', 'THOP1_COD',
            'PTCAR_LG_NM_NL', 'LINE_NO_DEP', 'LINE_NO_ARR',
            'REAL_TIME_ARR', 'PLANNED_TIME_ARR', 'DELAY_ARR',
            'REAL_DATE_ARR', 'PLANNED_DATE_ARR',
            'REAL_TIME_DEP', 'PLANNED_TIME_DEP', 'DELAY_DEP',
            'REAL_DATE_DEP', 'PLANNED_DATE_DEP',
        ],
    ):
        chunk = chunk[chunk['RELATION'] == LINE]
        chunk = chunk[chunk['THOP1_COD'].isin(['D', '='])]
        chunk = chunk[chunk['DELAY_ARR'].between(-3600, 36000)]
        if len(chunk) > 0:
            chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal: {len(df)} rows for {LINE}")

df['planned_dt'] = pd.to_datetime(
    df['PLANNED_DATE_ARR'] + ' ' + df['PLANNED_TIME_ARR'].fillna('00:00:00'),
    format='%d%b%Y %H:%M:%S',
    errors='coerce',
)

df = df.sort_values(['TRAIN_NO', 'planned_dt']).reset_index(drop=True)

print("Computing prev_delay...")
df['prev_delay_arr'] = df.groupby('TRAIN_NO')['DELAY_ARR'].shift(1)
df['prev_delay_dep'] = df.groupby('TRAIN_NO')['DELAY_DEP'].shift(1)

print(f"prev_delay_arr non-null: {df['prev_delay_arr'].notna().sum()}")
print(f"prev_delay_dep non-null: {df['prev_delay_dep'].notna().sum()}")

print(f"\nDelayed (>=360s): {(df['DELAY_ARR'] >= 360).sum()}")
print(f"Percent: {(df['DELAY_ARR'] >= 360).mean() * 100:.2f}%")
print(f"Elapsed: {time.time() - t0:.1f}s")

df.to_parquet(OUTPUT, index=False)
print(f"\nSaved to {OUTPUT}")
