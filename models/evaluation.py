import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

class ModelEvaluation:
    """Evaluation and research connections (Steps C+D)"""
    
    def evaluate_models(self, X_test, y_test):
        """Step C: Compare all models"""
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON (Step C)")
        print("="*60)
        
        results = []
        for name, model in self.models.items():
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            results.append({'Model': name, 'R²': r2, 'MAE': mae})
            print(f"{name:10} R²={r2:.3f}, MAE={mae:.2f} min")
        
        # Ensemble performance
        ensemble_pred = self.predict_ensemble(X_test)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        print(f"\n{'Ensemble':10} R²={ensemble_r2:.3f}, MAE={ensemble_mae:.2f} min")
        
        pd.DataFrame(results).to_csv("models/saved/model_comparison.csv", index=False)
    
    def connect_to_research(self):
        """Step D: Connect to Levy, TU Darmstadt, Al Ghamdi"""
        print("\n" + "="*60)
        print("📚 RESEARCH CONNECTIONS (Step D)")
        print("="*60)
        
        print("\n📖 Levy (2024) - Polycentric Networks:")
        print("   • Gaussian decay models zone interactions")
        print("   • 8 cities in Rhine-Ruhr connected via interaction matrix")
        
        print("\n📖 TU Darmstadt (2023) - Cologne Bottleneck:")
        print(f"   • 67% delays start in Cologne (captured by is_cologne_bottleneck)")
        print(f"   • Impact weight: {self.weights.get('cologne_effect', 2.0):.1f}x")
        
        print("\n📖 Al Ghamdi (2022) - Heterogeneous Ensembles:")
        print(f"   • XGBoost + RF + Gaussian weighted at {self.weights}")
        
        # Save summary
        with open("models/saved/research_connections.txt", "w") as f:
            f.write(f"Levy (2024): Gaussian decay\n")
            f.write(f"TU Darmstadt (2023): Cologne bottleneck\n")
            f.write(f"Al Ghamdi (2022): Ensemble weights: {self.weights}\n")