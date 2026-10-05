import pandas as pd
import glob
import time

LINE = 'IC 16-1'
OUTPUT = "data/processed/ic16_train_2023_2024.parquet"

files = sorted(glob.glob("data/raw/infrabel_2023-*.csv") + 
                glob.glob("data/raw/infrabel_2024-*.csv"))
print(f"Files: {len(files)}")

chunks = []
t0 = time.time()
for f in files:
    for chunk in pd.read_csv(f, chunksize=100_000,
                              usecols=['DATDEP', 'TRAIN_NO', 'RELATION', 'PTCAR_NO',
                                       'THOP1_COD', 'PTCAR_LG_NM_NL', 'LINE_NO_ARR',
                                       'REAL_TIME_ARR', 'PLANNED_TIME_ARR', 'DELAY_ARR',
                                       'REAL_DATE_ARR', 'PLANNED_DATE_ARR']):
        chunk = chunk[chunk['RELATION'] == LINE]
        chunk = chunk[chunk['THOP1_COD'].isin(['D', '='])]
        chunk = chunk[chunk['DELAY_ARR'].between(-3600, 36000)]
        if len(chunk) > 0:
            chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal: {len(df)} rows for {LINE}")
print(f"Delayed (≥360s): {(df['DELAY_ARR'] >= 360).sum()}")
print(f"Percent: {(df['DELAY_ARR'] >= 360).mean() * 100:.2f}%")
print(f"Elapsed: {time.time() - t0:.1f}s")

df.to_parquet(OUTPUT, index=False)
print(f"\nSaved to {OUTPUT}")
