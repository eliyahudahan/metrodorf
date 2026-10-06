"""
Metrodorf v2 — Streamlit Dashboard
Portfolio project · Not deployed in production.

Target audience: rail analyst at a European railway operator.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

st.set_page_config(
    page_title="Metrodorf — Train Delay Prediction",
    page_icon="🚆",
    layout="wide",
)

# ============================================
# Load artifacts
# ============================================
@st.cache_data
def load_metrics():
    with open("models/saved/day3_metrics.json") as f:
        val = json.load(f)
    with open("models/saved/test_metrics.json") as f:
        test = json.load(f)
    with open("models/saved/lstm_results.json") as f:
        lstm = json.load(f)
    return val, test, lstm

try:
    val_metrics, test_metrics, lstm_results = load_metrics()
except Exception as e:
    st.error(f"Could not load metrics: {e}")
    st.stop()

# ============================================
# Header
# ============================================
st.title("🚆 Metrodorf v2 — Train Delay Prediction")
st.markdown("**Portfolio project** · *Not deployed in production*")
st.markdown("Line **IC 16-1** · Infrabel (Belgium) · 2023–2024 train, Jan 2025 test")
st.divider()

# ============================================
# Tabs
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["📋 Problem", "🔬 Method", "📊 Results", "🗺️ Map"])

# ============================================
# TAB 1 — Problem
# ============================================
with tab1:
    st.header("1. What is the problem?")
    st.markdown("""
    **Question:** *"Will a train on line IC 16-1 arrive **more than 6 minutes late**
    (≥360 seconds) at a given station, on a given day and hour?"*
    
    **Target audience:** A rail analyst at a European railway operator.
    
    **Why this matters:**
    - Trains delayed by ≥6 minutes disrupt passenger connections.
    - Rail analysts need advance warning to plan rolling stock and operational flexibility.
    - A delay prediction model gives the analyst time to react — reallocate trains, notify staff, adjust schedules.
    """)
    
    st.subheader("Ground Truth (GT)")
    st.markdown("""
    - **Source:** Infrabel Open Data (`DELAY_ARR`, computed by Infrabel in seconds).
    - **Definition:** `REAL_TIME_ARR − PLANNED_TIME_ARR`.
    - **Resolution:** 1 second.
    - **Target:** `DELAY_ARR >= 360` seconds (6 minutes).
    - **Why 360:** Aligns with Infrabel's KPI P.II1 (*"National passenger punctuality rate at 6 minutes at 111 measuring points"*).
    
    **Note:** Infrabel's official metric measures at terminus + Brussels junction.
    This project measures delay at **every station observation** on the line.
    It is **not** a reproduction of the official metric.
    """)
    
    st.subheader("Not included")
    st.markdown("""
    - **Cancelled trains** are not in the dataset (Infrabel reports them separately).
    - **Weather** features are not used (out of scope for this portfolio project).
    """)

# ============================================
# TAB 2 — Method
# ============================================
with tab2:
    st.header("2. How is the solution built?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data")
        st.markdown("""
        - **Source:** Infrabel Open Data (CC0 license).
        - **Period:** January 2023 – December 2024 (train), January 2025 (test).
        - **Scale:** 45.6M rows total, 1.07M rows for line IC 16-1.
        - **Split:** 2023 → train, 2024 → validation, Jan 2025 → test.
        """)
        
        st.subheader("Features (8)")
        st.markdown("""
        1. `hour` — hour of planned arrival
        2. `day_of_week`
        3. `month`
        4. `is_weekend`
        5. `is_stop` — stop type
        6. **`prev_delay_arr`** — previous station's arrival delay
        7. **`prev_delay_dep`** — previous station's departure delay
        8. `station_enc` — station target encoding (train only)
        """)
    
    with col2:
        st.subheader("Models")
        st.markdown("""
        - **Baseline:** persistence t-1 (predict delayed if previous station delayed).
        - **Primary:** LightGBM (tabular data, engineered features).
        - **Challenger:** LSTM (sequence of 5 previous stations).
        """)
        
        st.subheader("Empirical validation of `prev_delay`")
        st.markdown("""
        - Correlation `prev_delay_dep` vs `DELAY_ARR`: **0.9636**
        - P(delayed | prev NOT delayed): **1.74%**
        - P(delayed | prev delayed): **89.81%**
        - Ratio: **51.66×**
        - Mutual Information: **0.2922**
        
        → This is a **measured** relationship, not an assumption.
        """)
        
        st.subheader("Overfitting check")
        st.markdown("""
        - Train F1: **0.9474**
        - Validation F1: **0.9225**
        - Test F1: **0.9252**
        - Gap: **0.0249** (healthy — < 0.05)
        """)

# ============================================
# TAB 3 — Results
# ============================================
with tab3:
    st.header("3. How well did it work?")
    
    st.subheader("Validation (2024) — LightGBM vs Baseline")
    
    v_lgb = val_metrics['lightgbm']
    v_base = val_metrics['baseline']
    
    df_val = pd.DataFrame({
        'Model': ['Baseline (persistence t-1)', 'LightGBM'],
        'TP': [v_base['tp'], v_lgb['tp']],
        'FN': [v_base['fn'], v_lgb['fn']],
        'FP': [v_base['fp'], v_lgb['fp']],
        'TN': [v_base['tn'], v_lgb['tn']],
        'Precision': [v_base['precision'], v_lgb['precision']],
        'Recall': [v_base['recall'], v_lgb['recall']],
        'F1': [v_base['f1'], v_lgb['f1']],
    })
    st.dataframe(df_val.style.format({
        'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1': '{:.4f}'
    }), use_container_width=True)
    
    st.subheader("Test (January 2025) — LightGBM vs Baseline")
    
    t_lgb = test_metrics['lightgbm']
    t_base = test_metrics['baseline']
    
    df_test = pd.DataFrame({
        'Model': ['Baseline (persistence t-1)', 'LightGBM'],
        'TP': [t_base['tp'], t_lgb['tp']],
        'FN': [t_base['fn'], t_lgb['fn']],
        'FP': [t_base['fp'], t_lgb['fp']],
        'TN': [t_base['tn'], t_lgb['tn']],
        'Precision': [t_base['precision'], t_lgb['precision']],
        'Recall': [t_base['recall'], t_lgb['recall']],
        'F1': [t_base['f1'], t_lgb['f1']],
    })
    st.dataframe(df_test.style.format({
        'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1': '{:.4f}'
    }), use_container_width=True)
    
    st.success(f"**Improvement on test set: +{(t_lgb['f1'] - t_base['f1'])*100:.2f} pp F1**")
    
    st.subheader("LSTM Challenger (Validation 2024)")
    df_lstm = pd.DataFrame(lstm_results)
    df_lstm['config'] = df_lstm['config'].astype(str)
    df_lstm = df_lstm[['config', 'val_f1']].rename(columns={'val_f1': 'F1'})
    st.dataframe(df_lstm, use_container_width=True)
    
    st.markdown(f"""
    **Decision:** LightGBM wins. LSTM best F1 = {max(r['val_f1'] for r in lstm_results):.4f},
    vs LightGBM = {v_lgb['f1']:.4f}. The data is tabular; engineered features
    (`prev_delay`) already capture the sequential dependency.
    """)

# ============================================
# TAB 4 — Map
# ============================================
with tab4:
    st.header("4. Where do delays happen?")
    st.markdown("Line **IC 16-1** (Belgium) — Brussels ↔ Luxembourg corridor.")
    
    # Approximate station coordinates for IC 16-1
    stations = [
        {"name": "BRUSSEL-ZUID", "lat": 50.8355, "lon": 4.3355},
        {"name": "BRUSSEL-CENTRAAL", "lat": 50.8455, "lon": 4.3570},
        {"name": "BRUSSEL-NOORD", "lat": 50.8600, "lon": 4.3610},
        {"name": "BRUSSEL-SCHUMAN", "lat": 50.8434, "lon": 4.3810},
        {"name": "OTTIGNIES", "lat": 50.6653, "lon": 4.5695},
        {"name": "GEMBLOUX", "lat": 50.5703, "lon": 4.6910},
        {"name": "NAMUR", "lat": 50.4674, "lon": 4.8718},
        {"name": "CINEY", "lat": 50.2944, "lon": 5.1002},
        {"name": "MARLOIE", "lat": 50.2076, "lon": 5.3325},
        {"name": "LIBRAMONT", "lat": 49.9169, "lon": 5.3725},
        {"name": "ARLON", "lat": 49.6836, "lon": 5.8167},
    ]
    
    df_map = pd.DataFrame(stations)
    
    fig = px.scatter_mapbox(
        df_map,
        lat="lat", lon="lon", text="name",
        zoom=7, height=600,
        title="IC 16-1 line — main stations",
        mapbox_style="open-street-map",
    )
    fig.update_traces(marker=dict(size=12, color='#1f77b4'))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("Station positions are approximate. Map shows the corridor, not all stops.")

# ============================================
# Footer — Limitations
# ============================================
st.divider()
st.markdown("""
### Limitations

- **Official metric difference:** Infrabel measures at terminus + Brussels junction; this project measures at every station observation.
- **Cancellations** are not in the dataset.
- **Single line:** IC 16-1 only. Not generalizable to other lines without retraining.
- **No external features:** weather, incidents, and holidays are not used.
- **Threshold 360s** is a domain decision, not an optimization. Sensitivity at 180/300/420 is available.

### Links

- **GitHub:** [github.com/eliyahudahan/metrodorf](https://github.com/eliyahudahan/metrodorf)
- **License:** Infrabel Open Data (CC0).
""")
