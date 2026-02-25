"""
Gaussian-inspired model for delay prediction
Based on Levy's polycentric region theory
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

class GaussianInspiredModel:
    """
    Custom model inspired by Gaussian processes
    Incorporates distance decay (Levy's insight)
    """
    
    def __init__(self, zone_matrix=None):
        self.zone_matrix = zone_matrix
        self.model = None
        self.coefficients = None
    
    def fit(self, X, y):
        # Simple linear model with Gaussian prior
        # Add Gaussian kernel features
        X_with_kernel = X.copy()
        
        # Add interaction terms based on zone matrix
        if self.zone_matrix is not None and 'is_cologne_bottleneck' in X.columns:
            X_with_kernel['cologne_kernel'] = X['is_cologne_bottleneck'] * 2.0
        
        # Add distance decay feature
        if 'distance_km' in X.columns:
            sigma = 50  # Typical zone size in Rhine-Ruhr
            X_with_kernel['distance_decay'] = np.exp(-(X['distance_km']**2) / (2 * sigma**2))
        
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