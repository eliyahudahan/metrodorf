"""
Database configuration for PostgreSQL
Loads settings from .env file securely
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

# Validate required configuration
_missing = [k for k, v in DB_CONFIG.items() if v is None]
if _missing:
    raise ValueError(
        f"Missing required environment variables: {_missing}\n"
        "Please ensure .env file exists with DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD"
    )

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