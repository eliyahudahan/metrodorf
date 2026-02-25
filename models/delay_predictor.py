"""
Main Delay Predictor - Integrating all components
Based on Levy (2024), TU Darmstadt (2023), Al Ghamdi (2022)
"""

import logging
from pathlib import Path
import pandas as pd
import joblib

from .base_predictor import BasePredictor
from .ensemble_methods import EnsembleMethods
from .training_pipeline import TrainingPipeline
from .evaluation import ModelEvaluation

logger = logging.getLogger(__name__)

class DelayPredictor(BasePredictor, EnsembleMethods, TrainingPipeline, ModelEvaluation):
    """
    Complete delay prediction system for Rhine-Ruhr region
    Combines: data loading (Base), ensemble (EnsembleMethods), 
              training (TrainingPipeline), evaluation (ModelEvaluation)
    """
    
    def __init__(self):
        super().__init__()
        
    def train_ensemble(self):
        """Override to add evaluation and saving"""
        X_test, y_test = super().train_ensemble()
        
        # Step C: Evaluate
        self.evaluate_models(X_test, y_test)
        
        # Step D: Research connections
        self.connect_to_research()
        
        # Save models
        self.save_models()
        
        return {
            'xgb_score': self.weights.get('xgb', 0),
            'rf_score': self.weights.get('rf', 0),
            'gaussian_score': self.weights.get('gaussian', 0),
            'ensemble_weights': self.weights
        }
    
    def save_models(self):
        """Save trained models"""
        print("\n💾 SAVING MODELS...")
        Path("models/saved").mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f"models/saved/{name}_model.pkl")
        
        pd.Series(self.weights).to_csv("models/saved/model_weights.csv")
        print(f"✅ Saved weights: {self.weights}")
    
    def load_models(self):
        """Load trained models"""
        for name in ['xgb', 'rf', 'gaussian']:
            self.models[name] = joblib.load(f"models/saved/{name}_model.pkl")
        self.weights = pd.read_csv("models/saved/model_weights.csv", index_col=0)[0].to_dict()
        logger.info("✅ Models loaded")


# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚆 DELAY PREDICTION MODEL (Steps B→C→D)")
    print("="*60 + "\n")
    
    predictor = DelayPredictor()
    results = predictor.train_ensemble()
    
    print("\n🔮 SAMPLE PREDICTIONS:")
    print("-"*40)
    
    # Test cases
    test_cases = [
        ("Cologne → Dortmund, peak", 70, 17, 2, 1, 1),
        ("Essen → Bochum, off-peak", 15, 10, 2, 0, 0),
        ("Cologne → Düsseldorf, peak", 40, 8, 2, 1, 1),
    ]
    
    for desc, dist, tod, dow, peak, cologne in test_cases:
        delay = predictor.predict_delay(dist, tod, dow, peak, cologne)
        print(f"{desc:30}: {delay:.1f} min")