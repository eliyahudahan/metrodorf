"""
Gaussian-inspired model for delay prediction
Based on Bologna 2025 findings:
- Heavy-tailed delay distributions
- Laplacian noise for station-to-station fluctuations
- Priority rules (Cologne bottleneck = 2.0x multiplier)
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

class GaussianInspiredModel:
    """
    Model that captures Bologna 2025 insights:
    • Heavy tails → Ridge regularization prevents overfitting to extremes
    • Laplacian noise → distance_decay feature models uncertainty
    • Priority rules → cologne_kernel applies 2.0x multiplier
    """
    
    def __init__(self, zone_matrix=None):
        self.zone_matrix = zone_matrix
        self.model = None
        self.coefficients = None
    
    def fit(self, X, y):
        """
        Train model with Bologna-inspired features
        - L2 regularization (alpha=1.0) handles heavy tails
        - Cologne bottleneck gets 2.0x priority multiplier
        - Distance decay models Laplacian noise
        """
        # Create copy to avoid modifying original
        X_with_kernel = X.copy()
        
        # Bologna 2025: Priority rules - Cologne bottleneck effect
        if self.zone_matrix is not None and 'is_cologne_bottleneck' in X.columns:
            X_with_kernel['cologne_kernel'] = X['is_cologne_bottleneck'] * 2.0
        
        # Bologna 2025: Laplacian noise - distance-based decay
        if 'distance_km' in X.columns:
            sigma = 50  # Regional scale (km)
            X_with_kernel['distance_decay'] = np.exp(-(X['distance_km']**2) / (2 * sigma**2))
        
        # Ridge regression with L2 regularization prevents overfitting to heavy tails
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_with_kernel, y)
        return self
    
    def predict(self, X):
        """Make predictions using trained model"""
        X_with_kernel = X.copy()
        
        if self.zone_matrix is not None and 'is_cologne_bottleneck' in X.columns:
            X_with_kernel['cologne_kernel'] = X['is_cologne_bottleneck'] * 2.0
        
        if 'distance_km' in X.columns:
            sigma = 50
            X_with_kernel['distance_decay'] = np.exp(-(X['distance_km']**2) / (2 * sigma**2))
        
        return self.model.predict(X_with_kernel)
    
    def score(self, X, y):
        """Calculate R² score"""
        return r2_score(y, self.predict(X))