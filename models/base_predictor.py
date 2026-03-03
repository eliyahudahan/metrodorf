"""
Base predictor class for Metrodorf
Loads data and creates features based on core research papers:
- Bologna 2025: Heavy tails, Laplacian noise, priority rules
- UvA 2025: External factors (time, events) essential for beating baseline
- Al Ghamdi 2022: Features prepared for ensemble methods
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BasePredictor:
    """Base class with data loading and feature preparation"""
    
    def __init__(self):
        # Load preprocessed training data
        self.training_data = pd.read_csv("data/processed/training_data.csv")
        
        # Load zone interaction matrix (city-to-city influence)
        self.zone_matrix = pd.read_csv("data/processed/zone_interaction_matrix.csv", index_col=0)
        
        # Load station features
        self.zone_features = pd.read_csv("data/processed/zone_features.csv")
        
        # Initialize model storage
        self.models = {}      # Will store trained models: xgb, rf, gaussian
        self.weights = {}     # Will store R²-based weights for ensemble
        
        logger.info(f"✅ Loaded {len(self.training_data)} training samples")
        logger.info(f"✅ Loaded {len(self.zone_matrix)} zones")
    
    def prepare_features(self):
        """
        Create feature matrix for model training
        Features based on research findings:
        - Bologna 2025: Heavy tails require distance-based decay (Laplacian noise)
        - Bologna 2025: Priority rules explain Cologne bottleneck (2.0x multiplier)
        - UvA 2025: External factors (peak hour) essential for beating 0.65 baseline
        - Al Ghamdi 2022: Features engineered for ensemble learning
        """
        df = self.training_data.copy()
        
        # Bologna 2025: Priority rules - Cologne bottleneck effect
        # Trains passing through Cologne suffer 2x delay impact
        df['cologne_effect'] = df['is_cologne_bottleneck'] * 2.0
        
        # UvA 2025: External factors - peak hour impact
        # Time-of-day crucial for beating baseline (0.65 BA)
        df['peak_effect'] = df['is_peak_hour'] * 1.5
        
        # Bologna 2025: Laplacian noise - distance-based decay
        # Station-to-station fluctuations modeled as Gaussian decay
        # Sigma=50km matches Rhine-Ruhr regional scale
        df['distance_decay'] = np.exp(-(df['distance_km']**2) / (2 * 50**2))
        
        # Bologna + UvA: Non-linear interaction effects
        # When both Cologne AND peak hour occur, impact > sum of parts
        df['cologne_peak_interaction'] = df['cologne_effect'] * df['peak_effect']
        
        # Feature set for model training
        feature_columns = [
            'distance_km',           # Travel distance
            'time_of_day',           # Hour of day (UvA external factor)
            'day_of_week',           # Day of week (UvA external factor)
            'is_peak_hour',          # Peak hour flag (UvA)
            'is_cologne_bottleneck', # Cologne passage flag (Bologna)
            'cologne_effect',        # 2.0x multiplier (Bologna priority)
            'peak_effect',           # 1.5x multiplier (UvA external)
            'distance_decay',        # Laplacian noise (Bologna)
            'cologne_peak_interaction' # Combined effect
        ]
        
        logger.info(f"✅ Created {len(feature_columns)} features")
        logger.info(f"   Features based on: Bologna 2025 (priority + Laplacian), UvA 2025 (external factors), Al Ghamdi 2022 (ensemble)")
        
        # Return features (X) and target (y)
        return df[feature_columns], df['delay_minutes']