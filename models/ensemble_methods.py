"""
Ensemble prediction methods based on Al Ghamdi (2022)
- Weighted Averaging (WE) outperforms simple averaging
- Weights based on R² scores from validation
"""

import numpy as np
import pandas as pd

class EnsembleMethods:
    """Ensemble prediction methods (Al Ghamdi's WE)"""
    
    def predict_ensemble(self, X):
        """
        Weighted averaging (Al Ghamdi 2022, Section 3.2.6)
        WE method: final = Σ(weight_i * prediction_i) / Σ(weights)
        """
        predictions = np.zeros(len(X))
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            weight = self.weights[name] / total_weight
            predictions += weight * model.predict(X)
        
        return predictions
    
    def predict_delay(self, distance, time_of_day, day_of_week, is_peak, is_cologne):
        """
        Predict delay for a single journey
        Combines all research insights:
        - Bologna 2025: priority rules (cologne_effect)
        - UvA 2025: external factors (peak_effect)
        - Bologna 2025: Laplacian noise (distance_decay)
        - Al Ghamdi 2022: ensemble averaging
        """
        features = pd.DataFrame([{
            'distance_km': distance,
            'time_of_day': time_of_day,
            'day_of_week': day_of_week,
            'is_peak_hour': is_peak,
            'is_cologne_bottleneck': is_cologne,
            'cologne_effect': is_cologne * 2.0,      # Bologna priority
            'peak_effect': is_peak * 1.5,            # UvA external
            'distance_decay': np.exp(-(distance**2) / (2 * 50**2)),  # Bologna Laplacian
            'cologne_peak_interaction': (is_cologne * 2.0) * (is_peak * 1.5)  # Combined
        }])
        return self.predict_ensemble(features)[0]