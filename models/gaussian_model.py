"""
Gaussian-inspired model for delay prediction
Based on Bologna 2025 power laws and heavy tails
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

class GaussianInspiredModel:
    """
    Gaussian-inspired model for delay prediction
    
    Based on Bologna 2025 (Rondini et al.):
    • Station-to-station delays follow Asymmetric Laplace distribution
    • Local trains show power-law tails (heavy tails)
    • Priority rules explain cut-offs at 30/60 minutes
    
    Our model captures these through:
    • distance_decay: Gaussian decay with sigma=50
    • cologne_kernel: 2.0 multiplier for bottleneck
    • Ridge regression with L2 regularization
    """
    
    def __init__(self, zone_matrix=None):
        self.zone_matrix = zone_matrix
        self.model = None
        self.coefficients = None
    
    def fit(self, X, y):
        # Create copy to avoid modifying original
        X_with_kernel = X.copy()
        
        # Add Cologne bottleneck effect (2.0x multiplier)
        if self.zone_matrix is not None and 'is_cologne_bottleneck' in X.columns:
            X_with_kernel['cologne_kernel'] = X['is_cologne_bottleneck'] * 2.0
        
        # Add Gaussian distance decay (Levy's insight)
        if 'distance_km' in X.columns:
            sigma = 50  # Typical zone size in Rhine-Ruhr
            X_with_kernel['distance_decay'] = np.exp(-(X['distance_km']**2) / (2 * sigma**2))
        
        # Train Ridge regression
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_with_kernel, y)
        return self
    
    def predict(self, X):
        X_with_kernel = X.copy()
        
        if self.zone_matrix is not None and 'is_cologne_bottleneck' in X.columns:
            X_with_kernel['cologne_kernel'] = X['is_cologne_bottleneck'] * 2.0
        
        if 'distance_km' in X.columns:
            sigma = 50
            X_with_kernel['distance_decay'] = np.exp(-(X['distance_km']**2) / (2 * sigma**2))
        
        return self.model.predict(X_with_kernel)
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))