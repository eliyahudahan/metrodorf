"""
Database configuration for PostgreSQL
Loads settings from .env file securely
Gracefully handles missing .env for Streamlit Cloud
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Get the absolute path to the project root (where .env is located)
PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Load .env file from project root
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    # Fallback: try current directory (for development)
    load_dotenv()

# Database configuration with secure defaults
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

# Check which variables are missing
_missing = [k for k, v in DB_CONFIG.items() if v is None]

# Only raise error if SOME but not ALL are missing
# If ALL are missing → Streamlit Cloud, use fallback
# If SOME are missing → misconfigured .env, warn
if _missing:
    if len(_missing) == len(DB_CONFIG):
        # All missing — likely Streamlit Cloud, no .env file
        DATABASE_URL = None
    else:
        # Partial config — warn but don't crash
        import logging
        logging.warning(
            f"⚠️ Missing database config: {_missing}. "
            "Database features will be disabled."
        )
        DATABASE_URL = None
else:
    # Connection string for SQLAlchemy (never printed!)
    DATABASE_URL = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )

# Optional: helper function to get config without exposing password
def get_config_safe():
    """Return database config with password masked for logging"""
    return {
        'host': DB_CONFIG['host'],
        'port': DB_CONFIG['port'],
        'database': DB_CONFIG['database'],
        'user': DB_CONFIG['user'],
        'password': '***MASKED***'
    }