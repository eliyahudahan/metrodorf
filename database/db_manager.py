"""
PostgreSQL Database Manager for Metrodorf
Handles all database operations with SQLAlchemy
Gracefully falls back to CSV when database is unavailable (Streamlit Cloud)
"""

import pandas as pd
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Safely import DATABASE_URL — fall back to None if config is missing
try:
    from .db_config import DATABASE_URL
except (ImportError, ModuleNotFoundError):
    DATABASE_URL = None

class DatabaseManager:
    """
    Manages all PostgreSQL operations for Metrodorf
    Falls back gracefully when .env is not available (Streamlit Cloud)
    """
    
    def __init__(self):
        self.engine = None
        self.available = False
        
        if DATABASE_URL:
            try:
                self.engine = create_engine(
                    DATABASE_URL,
                    pool_size=5,
                    max_overflow=10,
                    echo=False
                )
                self.available = True
                logger.info("✅ Connected to PostgreSQL database")
            except Exception as e:
                logger.warning(f"⚠️ Database not available: {e}")
                logger.info("📁 Will use CSV fallback for all operations")
        else:
            logger.info("📁 No database config found — using CSV fallback")
    
    @contextmanager
    def get_connection(self):
        if not self.available or not self.engine:
            raise RuntimeError("Database not available — use CSV fallback")
        conn = self.engine.connect()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def create_tables(self):
        """Create all tables if they don't exist"""
        if not self.available:
            logger.info("📁 Skipping table creation (no database)")
            return
        
        try:
            with self.get_connection() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS stations (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) UNIQUE NOT NULL,
                        eva VARCHAR(10) UNIQUE,
                        ds100 VARCHAR(5),
                        latitude DECIMAL(10, 8),
                        longitude DECIMAL(11, 8),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS real_delays (
                        id SERIAL PRIMARY KEY,
                        station_id INTEGER REFERENCES stations(id) ON DELETE CASCADE,
                        distance_km DECIMAL(6, 2),
                        time_of_day INTEGER CHECK (time_of_day >= 0 AND time_of_day <= 23),
                        day_of_week INTEGER CHECK (day_of_week >= 0 AND day_of_week <= 6),
                        is_peak_hour BOOLEAN,
                        is_cologne_bottleneck BOOLEAN,
                        delay_minutes DECIMAL(6, 2),
                        source VARCHAR(20),
                        api_timestamp TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_real_delays_station ON real_delays(station_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_real_delays_timestamp ON real_delays(api_timestamp)"))
                
                logger.info("✅ Tables created/verified")
        except RuntimeError:
            logger.info("📁 Cannot create tables — no database connection")
    
    def insert_station(self, name, eva=None, ds100=None, lat=None, lon=None):
        """Insert or update a station"""
        if not self.available:
            logger.info("📁 Skipping station insert (no database)")
            return None
        
        try:
            with self.get_connection() as conn:
                result = conn.execute(
                    text("SELECT id FROM stations WHERE name = :name OR eva = :eva"),
                    {"name": name, "eva": eva}
                ).first()
                
                if result:
                    conn.execute(
                        text("""
                            UPDATE stations 
                            SET eva = COALESCE(:eva, eva),
                                ds100 = COALESCE(:ds100, ds100),
                                latitude = COALESCE(:lat, latitude),
                                longitude = COALESCE(:lon, longitude),
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = :id
                        """),
                        {"id": result[0], "eva": eva, "ds100": ds100, 
                         "lat": lat, "lon": lon}
                    )
                    return result[0]
                else:
                    result = conn.execute(
                        text("""
                            INSERT INTO stations (name, eva, ds100, latitude, longitude)
                            VALUES (:name, :eva, :ds100, :lat, :lon)
                            RETURNING id
                        """),
                        {"name": name, "eva": eva, "ds100": ds100, 
                         "lat": lat, "lon": lon}
                    )
                    return result.scalar()
        except RuntimeError:
            return None
    
    def insert_real_delay(self, station_id, delay_data):
        if not self.available:
            logger.info("📁 Skipping delay insert (no database)")
            return False
        
        try:
            with self.get_connection() as conn:
                conn.execute(
                    text("""
                        INSERT INTO real_delays (
                            station_id, distance_km, time_of_day, day_of_week,
                            is_peak_hour, is_cologne_bottleneck, delay_minutes,
                            source, api_timestamp
                        ) VALUES (
                            :station_id, :distance_km, :time_of_day, :day_of_week,
                            :is_peak_hour, :is_cologne_bottleneck, :delay_minutes,
                            :source, :api_timestamp
                        )
                    """),
                    {
                        "station_id": station_id,
                        "distance_km": delay_data['distance_km'],
                        "time_of_day": delay_data['time_of_day'],
                        "day_of_week": delay_data['day_of_week'],
                        "is_peak_hour": bool(delay_data['is_peak_hour']),  
                        "is_cologne_bottleneck": bool(delay_data['is_cologne_bottleneck']), 
                        "delay_minutes": delay_data['delay_minutes'],
                        "source": delay_data.get('source', 'real'),
                        "api_timestamp": delay_data.get('timestamp', datetime.now().isoformat())
                    }
                )
            return True
        except RuntimeError:
            return False
    
    def get_training_data(self, limit=None):
        """Get all real delays for training"""
        if not self.available or self.engine is None:
           logger.info("📁 Database not available for training data retrieval")
           return pd.DataFrame()
    
        try:
            query = "SELECT * FROM real_delays ORDER BY api_timestamp"
            if limit:
               query += f" LIMIT {limit}"
        
            df = pd.read_sql(query, self.engine)
            logger.info(f"📊 Loaded {len(df)} real delay records from database")
            return df
        except Exception as e:
             logger.info(f"📁 Failed to load from database: {e}")
             return pd.DataFrame()

    
    def close(self):
        """Close all connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("✅ Database connections closed")