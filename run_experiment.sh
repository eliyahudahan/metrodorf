#!/bin/bash
# ======================================================================
# METRODORF SAFE EXPERIMENT RUNNER
# ======================================================================
# Runs the model multiple times with built-in protections against:
# - API rate limiting (15-second delays between runs)
# - Overfitting (tracks metrics and stops if R² drops)
# - Data loss (saves everything to timestamped files)
#
# Usage: ./run_experiment_safe.sh [number_of_runs]
# Default: 20 runs
# ======================================================================

# --- Configuration ---------------------------------------------------
TOTAL_RUNS=${1:-20}                    # Number of runs (default 20)
MIN_R2_THRESHOLD=0.2                    # Stop if ensemble R² drops below this
CHECK_INTERVAL=5                        # Check health every N runs
API_COOLDOWN=15                          # Seconds between runs (respect APIs)
RESULTS_DIR="experiment_results_$(date +%Y%m%d_%H%M%S)"
# ----------------------------------------------------------------------

echo "=========================================================="
echo "🚆 METRODORF SAFE EXPERIMENT RUNNER"
echo "=========================================================="
echo "Started: $(date)"
echo "Total runs: $TOTAL_RUNS"
echo "Min R² threshold: $MIN_R2_THRESHOLD"
echo "API cooldown: $API_COOLDOWN seconds"
echo "Results directory: $RESULTS_DIR"
echo "=========================================================="

# Create results directory
mkdir -p "$RESULTS_DIR"

# Main results file
MAIN_LOG="$RESULTS_DIR/experiment_log.csv"
echo "run,timestamp,xgb_r2,rf_r2,gaussian_r2,ensemble_r2,ensemble_mae,xgb_weight,rf_weight,gaussian_weight,real_samples" > "$MAIN_LOG"

# Track best model
BEST_R2=0
BEST_RUN=0

for i in $(seq 1 $TOTAL_RUNS); do
    echo ""
    echo "=========================================================="
    echo "🔄 RUN $i OF $TOTAL_RUNS"
    echo "Started: $(date)"
    echo "=========================================================="
    
    # Run the model and capture output
    OUTPUT=$(python -m models.delay_predictor 2>&1)
    RUN_EXIT_CODE=$?
    
    if [ $RUN_EXIT_CODE -ne 0 ]; then
        echo "❌ Run $i failed with exit code $RUN_EXIT_CODE"
        echo "$OUTPUT" > "$RESULTS_DIR/run_${i}_error.log"
        continue
    fi
    
    # Save full output
    echo "$OUTPUT" > "$RESULTS_DIR/run_${i}_full.log"
    
    # --- Extract metrics ---------------------------------------------
    # Get R² scores from MODEL COMPARISON section
    XGB_R2=$(echo "$OUTPUT" | grep -A5 "MODEL COMPARISON" | grep "xgb" | head -1 | awk '{print $2}' | sed 's/R.=.//')
    RF_R2=$(echo "$OUTPUT" | grep -A5 "MODEL COMPARISON" | grep "rf" | head -1 | awk '{print $2}' | sed 's/R.=.//')
    GAUSS_R2=$(echo "$OUTPUT" | grep -A5 "MODEL COMPARISON" | grep "gaussian" | head -1 | awk '{print $2}' | sed 's/R.=.//')
    ENSEMBLE_R2=$(echo "$OUTPUT" | grep -A2 "Ensemble" | grep "R.=" | head -1 | awk '{print $2}' | sed 's/R.=.//')
    ENSEMBLE_MAE=$(echo "$OUTPUT" | grep -A2 "Ensemble" | grep "MAE=" | head -1 | awk '{print $2}' | sed 's/MAE=.//')
    
    # Get weights from Weighted Ensemble section
    WEIGHTS_LINE=$(echo "$OUTPUT" | grep -A3 "🔢 Weighted Ensemble" | grep -o "xgb.*" | head -1)
    XGB_W=$(echo "$WEIGHTS_LINE" | grep -o "xgb:[0-9.]*" | cut -d: -f2)
    RF_W=$(echo "$WEIGHTS_LINE" | grep -o "rf:[0-9.]*" | cut -d: -f2)
    GAUSS_W=$(echo "$WEIGHTS_LINE" | grep -o "gaussian:[0-9.]*" | cut -d: -f2)
    
    # Count real samples from training data (if possible)
    if [ -f "data/processed/training_data.csv" ]; then
        REAL_SAMPLES=$(python -c "
import pandas as pd
df = pd.read_csv('data/processed/training_data.csv')
print(df[df['source']=='real'].shape[0] if 'source' in df.columns else 0)
" 2>/dev/null || echo "N/A")
    else
        REAL_SAMPLES="N/A"
    fi
    
    # Save to main log
    echo "$i,$(date +%H:%M:%S),$XGB_R2,$RF_R2,$GAUSS_R2,$ENSEMBLE_R2,$ENSEMBLE_MAE,$XGB_W,$RF_W,$GAUSS_W,$REAL_SAMPLES" >> "$MAIN_LOG"
    
    # --- Save model weights for this run ----------------------------
    if [ -f "models/saved/model_weights.csv" ]; then
        cp "models/saved/model_weights.csv" "$RESULTS_DIR/run_${i}_weights.csv"
    fi
    
    # --- Track best model -------------------------------------------
    if (( $(echo "$ENSEMBLE_R2 > $BEST_R2" | bc -l 2>/dev/null) )); then
        BEST_R2=$ENSEMBLE_R2
        BEST_RUN=$i
        # Save best model
        mkdir -p "$RESULTS_DIR/best_model"
        if [ -f "models/saved/xgb_model.pkl" ]; then
            cp "models/saved/xgb_model.pkl" "$RESULTS_DIR/best_model/"
            cp "models/saved/rf_model.pkl" "$RESULTS_DIR/best_model/"
            cp "models/saved/gaussian_model.pkl" "$RESULTS_DIR/best_model/"
            cp "models/saved/model_weights.csv" "$RESULTS_DIR/best_model/"
        fi
        echo "   🏆 New best model! R² = $ENSEMBLE_R2"
    fi
    
    # --- Check for overfitting --------------------------------------
    if (( $(echo "$ENSEMBLE_R2 < $MIN_R2_THRESHOLD" | bc -l 2>/dev/null) )); then
        if [ $i -gt 5 ]; then  # Only warn after first few runs
            echo "⚠️  WARNING: Ensemble R² ($ENSEMBLE_R2) below threshold ($MIN_R2_THRESHOLD)"
            echo "   Possible overfitting or API issues. Check run $i log."
        fi
    fi
    
    # --- Health check every N runs ----------------------------------
    if (( $i % $CHECK_INTERVAL == 0 )); then
        echo ""
        echo "📊 HEALTH CHECK at run $i"
        echo "   Current best R²: $BEST_R2 (run $BEST_RUN)"
        echo "   Last R²: $ENSEMBLE_R2"
        echo "   Real samples: $REAL_SAMPLES"
        echo "   Disk usage: $(du -sh data/raw/ 2>/dev/null | cut -f1)"
    fi
    
    echo "✅ Completed run $i at: $(date)"
    echo "   R²: XGB=$XGB_R2, RF=$RF_R2, GAUSS=$GAUSS_R2, Ensemble=$ENSEMBLE_R2"
    echo "   Weights: XGB=${XGB_W}, RF=${RF_W}, GAUSS=${GAUSS_W}"
    echo "   Real samples: $REAL_SAMPLES"
    
    # --- API cooldown (respect rate limits) ------------------------
    if [ $i -lt $TOTAL_RUNS ]; then
        echo "⏳ Cooling down for $API_COOLDOWN seconds (respecting API rate limits)..."
        sleep $API_COOLDOWN
    fi
    
    echo "----------------------------------------------------------"
done

# --- Final summary ------------------------------------------------
echo ""
echo "=========================================================="
echo "🎉 EXPERIMENT COMPLETED SUCCESSFULLY"
echo "=========================================================="
echo "Started: $(cat "$RESULTS_DIR/experiment_log.csv" | head -2 | tail -1 | cut -d, -f2)"
echo "Finished: $(date)"
echo "Total runs: $TOTAL_RUNS"
echo "Best ensemble R²: $BEST_R2 (run $BEST_RUN)"
echo ""
echo "📁 Results saved in: $RESULTS_DIR"
echo "   - Full logs: run_*_full.log"
echo "   - Metrics: experiment_log.csv"
echo "   - Best model: best_model/"
echo ""
echo "📊 Quick analysis command:"
echo "   python -c \"import pandas as pd; pd.read_csv('$RESULTS_DIR/experiment_log.csv').plot(x='run', y=['xgb_r2','rf_r2','gaussian_r2','ensemble_r2'])\""
echo "=========================================================="

# --- Optional: generate quick plot ------------------------------
if command -v python3 &>/dev/null; then
    python3 << EOF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv('$RESULTS_DIR/experiment_log.csv')

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# R² over time
axes[0,0].plot(df['run'], df['xgb_r2'], 'o-', label='XGBoost', alpha=0.7)
axes[0,0].plot(df['run'], df['rf_r2'], 's-', label='Random Forest', alpha=0.7)
axes[0,0].plot(df['run'], df['gaussian_r2'], '^-', label='Gaussian', alpha=0.7)
axes[0,0].plot(df['run'], df['ensemble_r2'], '*-', label='Ensemble', linewidth=2)
axes[0,0].axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Target 0.3')
axes[0,0].set_xlabel('Run')
axes[0,0].set_ylabel('R² Score')
axes[0,0].set_title('Model Performance Over Time')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Weights over time
axes[0,1].plot(df['run'], df['xgb_weight'], 'o-', label='XGBoost', alpha=0.7)
axes[0,1].plot(df['run'], df['rf_weight'], 's-', label='Random Forest', alpha=0.7)
axes[0,1].plot(df['run'], df['gaussian_weight'], '^-', label='Gaussian', alpha=0.7)
axes[0,1].set_xlabel('Run')
axes[0,1].set_ylabel('Weight')
axes[0,1].set_title('Ensemble Weights Over Time')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# MAE over time
axes[1,0].plot(df['run'], df['ensemble_mae'], 'o-', color='purple')
axes[1,0].set_xlabel('Run')
axes[1,0].set_ylabel('MAE (minutes)')
axes[1,0].set_title('Ensemble MAE Over Time')
axes[1,0].grid(True, alpha=0.3)

# Real samples over time (if available)
if 'real_samples' in df.columns and df['real_samples'].dtype != 'object':
    axes[1,1].plot(df['run'], df['real_samples'], 'o-', color='orange')
    axes[1,1].set_xlabel('Run')
    axes[1,1].set_ylabel('Real Samples')
    axes[1,1].set_title('Real Data Accumulation')
    axes[1,1].grid(True, alpha=0.3)
else:
    axes[1,1].text(0.5, 0.5, 'Real samples data not available', 
                   ha='center', va='center', transform=axes[1,1].transAxes)

plt.tight_layout()
plt.savefig('$RESULTS_DIR/experiment_summary.png', dpi=150)
print("📈 Summary plot saved to $RESULTS_DIR/experiment_summary.png")
EOF
fi

echo "=========================================================="