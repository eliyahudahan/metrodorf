import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BasePredictor:
    """Base class with data loading and feature preparation"""
    
    def __init__(self):
        self.training_data = pd.read_csv("data/processed/training_data.csv")
        self.zone_matrix = pd.read_csv("data/processed/zone_interaction_matrix.csv", index_col=0)
        self.zone_features = pd.read_csv("data/processed/zone_features.csv")
        self.models = {}
        self.weights = {}
        
        logger.info(f"✅ Loaded {len(self.training_data)} training samples")
        logger.info(f"✅ Loaded {len(self.zone_matrix)} zones")
    
    def prepare_features(self):
        """Create feature matrix (Levy's Gaussian decay)"""
        df = self.training_data.copy()
        
        # Levy's polycentric features
        df['cologne_effect'] = df['is_cologne_bottleneck'] * 2.0
        df['peak_effect'] = df['is_peak_hour'] * 1.5
        df['distance_decay'] = np.exp(-(df['distance_km']**2) / (2 * 50**2))
        df['cologne_peak_interaction'] = df['cologne_effect'] * df['peak_effect']
        
        feature_columns = [
            'distance_km', 'time_of_day', 'day_of_week',
            'is_peak_hour', 'is_cologne_bottleneck',
            'cologne_effect', 'peak_effect', 'distance_decay',
            'cologne_peak_interaction'
        ]
        
        logger.info(f"✅ Created {len(feature_columns)} features")
        return df[feature_columns], df['delay_minutes']