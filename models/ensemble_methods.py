"""
Ensemble prediction methods based on Al Ghamdi (2022)
- Weighted Averaging (WE) outperforms simple averaging
- Weights based on R² scores from validation
- Added prediction intervals for uncertainty quantification (Dr. Oscar's advice)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Union

class EnsembleMethods:
    """Ensemble prediction methods (Al Ghamdi's WE)"""
    
    # Type hints to help Pylance understand inherited attributes
    models: Dict[str, Any]
    weights: Dict[str, float]
    
    def predict_ensemble(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
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
    
    def predict_with_uncertainty(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns prediction with confidence interval.
        Based on Dr. Oscar's advice: use variance across ensemble members.
        
        Args:
            X: Feature matrix
            confidence_level: 0.68 for 1σ, 0.95 for 2σ, 0.99 for 3σ
        
        Returns:
            mean: Predicted delay (minutes)
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
        """
        # Get predictions from all models (unweighted, for variance)
        all_predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            all_predictions.append(pred.flatten())
        
        all_predictions = np.array(all_predictions)
        
        # Weighted mean (same as predict_ensemble)
        total_weight = sum(self.weights.values())
        weighted_mean = np.zeros(len(X))
        for name, model in self.models.items():
            weight = self.weights[name] / total_weight
            weighted_mean += weight * model.predict(X).flatten()
        
        # Calculate standard deviation across models
        std_dev = np.std(all_predictions, axis=0)
        
        # Z-score for confidence level
        z_scores = {0.68: 1.0, 0.95: 1.96, 0.99: 2.58}
        z = z_scores.get(confidence_level, 1.96)
        
        lower_bound = weighted_mean - z * std_dev
        upper_bound = weighted_mean + z * std_dev
        
        return weighted_mean, lower_bound, upper_bound
    
    def predict_delay(
        self, 
        distance: float, 
        time_of_day: int, 
        day_of_week: int, 
        is_peak: int, 
        is_cologne: int
    ) -> float:
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
            'cologne_effect': is_cologne * 2.0,
            'peak_effect': is_peak * 1.5,
            'distance_decay': np.exp(-(distance**2) / (2 * 50**2)),
            'cologne_peak_interaction': (is_cologne * 2.0) * (is_peak * 1.5)
        }])
        return self.predict_ensemble(features)[0]
    
    def predict_delay_with_ci(
        self, 
        distance: float, 
        time_of_day: int, 
        day_of_week: int, 
        is_peak: int, 
        is_cologne: int, 
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Predict delay with confidence interval for a single journey.
        Useful for showing uncertainty to dispatchers.
        """
        features = pd.DataFrame([{
            'distance_km': distance,
            'time_of_day': time_of_day,
            'day_of_week': day_of_week,
            'is_peak_hour': is_peak,
            'is_cologne_bottleneck': is_cologne,
            'cologne_effect': is_cologne * 2.0,
            'peak_effect': is_peak * 1.5,
            'distance_decay': np.exp(-(distance**2) / (2 * 50**2)),
            'cologne_peak_interaction': (is_cologne * 2.0) * (is_peak * 1.5)
        }])
        
        mean, lower, upper = self.predict_with_uncertainty(features, confidence_level)
        return mean[0], lower[0], upper[0]