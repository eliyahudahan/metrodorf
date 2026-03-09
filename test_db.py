from database.db_manager import DatabaseManager
import logging

logging.basicConfig(level=logging.INFO)

# Create tables
db = DatabaseManager()
db.create_tables()

# Insert a test station
station_id = db.insert_station(
    'Dortmund Hbf',
    '8000080',
    'EDO',
    51.517899,
    7.459294
)
print(f"✅ Station ID: {station_id}")

# Insert a test delay
test_delay = {
    'distance_km': 70.0,
    'time_of_day': 17,
    'day_of_week': 2,
    'is_peak_hour': 1,
    'is_cologne_bottleneck': 1,
    'delay_minutes': 8.3,
    'source': 'test',
    'timestamp': '2026-03-09T15:30:00'
}
db.insert_real_delay(station_id, test_delay)
print("✅ Test delay inserted")

# Retrieve data
df = db.get_training_data()
print(f"\n📊 Total records: {len(df)}")
print(df.head())

db.close()
