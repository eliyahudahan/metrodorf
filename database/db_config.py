"""
Database configuration for PostgreSQL
Loads settings from .env file
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'metrodorf'),
    'user': os.getenv('DB_USER', 'framg'),
    'password': os.getenv('DB_PASSWORD', '')
}

# Connection string for SQLAlchemy
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
