import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
from .gaussian_model import GaussianInspiredModel

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Training pipeline with smartphone-optimized parameters"""
    
    def train_ensemble(self):
        """Train heterogeneous ensemble (Al Ghamdi 2022)"""
        X, y = self.prepare_features()
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # XGBoost with smartphone learnings (LR=0.114 principle!)
        logger.info("Training XGBoost with optimized parameters...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,                    # Shallower = general patterns
            learning_rate=0.03,              # Slower = stable (like LR=0.114!)
            subsample=0.7,                    # Prevent overfitting
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgb'] = xgb_model
        self.weights['xgb'] = max(0, xgb_model.score(X_val, y_val))
        
        # Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)
        self.models['rf'] = rf_model
        self.weights['rf'] = max(0, rf_model.score(X_val, y_val))
        
        # Gaussian model (Levy's insight!)
        logger.info("Training Gaussian-inspired model...")
        self.models['gaussian'] = GaussianInspiredModel(self.zone_matrix)
        self.models['gaussian'].fit(X_train, y_train)
        self.weights['gaussian'] = max(0, self.models['gaussian'].score(X_val, y_val))
        
        # Normalize weights (like your 0.114 optimization!)
        total = sum(self.weights.values())
        if total > 0:
            for name in self.weights:
                self.weights[name] /= total
        
        logger.info(f"\n🔢 Ensemble weights: {self.weights}")
        return X_test, y_test