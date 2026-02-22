import requests
import pandas as pd
from pathlib import Path

# Create data folders
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Download free station data from DB Open Data
print("📥 Downloading station data...")

# Try multiple sources
sources = [
    "https://download-data.deutschebahn.com/static/datasets/betriebsstellen/DBNetz-Betriebsstellenverzeichnis-Stand2025-01.csv",
    "https://opendata.ecdc.europa.eu/transport/data/rail/station_codes.csv"
]

for url in sources:
    try:
        df = pd.read_csv(url)
        df.to_csv("data/raw/stations_backup.csv", index=False)
        print(f"✅ Downloaded {len(df)} stations")
        break
    except:
        continue
