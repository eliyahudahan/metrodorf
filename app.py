"""
Metrodorf v2 — Streamlit Dashboard
Portfolio project · Not deployed in production.

Target audience: rail analyst at a European railway operator.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Problem", "🔬 Method", "📊 Results", "🗺️ Map", "🎯 Examples"
])

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
    - A delay prediction model gives the analyst time to react.
    """)
    
    st.subheader("Ground Truth (GT)")
    st.markdown("""
    - **Source:** Infrabel Open Data (`DELAY_ARR`, computed by Infrabel in seconds).
    - **Definition:** `REAL_TIME_ARR − PLANNED_TIME_ARR`.
    - **Resolution:** 1 second.
    - **Target:** `DELAY_ARR >= 360` seconds (6 minutes).
    - **Why 360:** Aligns with Infrabel's KPI P.II1.
    
    **Note:** Infrabel's official metric measures at terminus + Brussels junction.
    This project measures delay at **every station observation** on the line.
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
        - **Source:** Infrabel Open Data (CC0).
        - **Period:** 2023–2024 (train), Jan 2025 (test).
        - **Scale:** 45.6M rows total, 1.07M rows for IC 16-1.
        - **Split:** 2023 → train, 2024 → validation, Jan 2025 → test.
        """)
        
        st.subheader("Features (8)")
        st.markdown("""
        1. `hour`
        2. `day_of_week`
        3. `month`
        4. `is_weekend`
        5. `is_stop`
        6. **`prev_delay_arr`**
        7. **`prev_delay_dep`**
        8. `station_enc`
        """)
    
    with col2:
        st.subheader("Models")
        st.markdown("""
        - **Baseline:** persistence t-1.
        - **Primary:** LightGBM.
        - **Challenger:** LSTM (sequence of 5 stations).
        """)
        
        st.subheader("Empirical validation of `prev_delay`")
        st.markdown("""
        - Correlation `prev_delay_dep` vs `DELAY_ARR`: **0.9636**
        - P(delayed | prev NOT delayed): **1.74%**
        - P(delayed | prev delayed): **89.81%**
        - Ratio: **51.66×**
        - Mutual Information: **0.2922**
        """)
        
        st.subheader("Overfitting check")
        st.markdown("""
        - Train F1: **0.9474**
        - Validation F1: **0.9225**
        - Test F1: **0.9252**
        - Gap: **0.0249** (healthy)
        """)

# ============================================
# TAB 3 — Results
# ============================================
with tab3:
    st.header("3. How well did it work?")
    
    st.subheader("Validation (2024)")
    v_lgb = val_metrics['lightgbm']
    v_base = val_metrics['baseline']
    df_val = pd.DataFrame({
        'Model': ['Baseline', 'LightGBM'],
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
    
    st.subheader("Test (January 2025)")
    t_lgb = test_metrics['lightgbm']
    t_base = test_metrics['baseline']
    df_test = pd.DataFrame({
        'Model': ['Baseline', 'LightGBM'],
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
    
    st.subheader("LSTM Challenger")
    df_lstm = pd.DataFrame(lstm_results)
    df_lstm['config'] = df_lstm['config'].astype(str)
    df_lstm = df_lstm[['config', 'val_f1']].rename(columns={'val_f1': 'F1'})
    st.dataframe(df_lstm, use_container_width=True)
    
    st.markdown(f"""
    **Decision:** LightGBM wins. LSTM best F1 = {max(r['val_f1'] for r in lstm_results):.4f},
    vs LightGBM = {v_lgb['f1']:.4f}.
    """)

# ============================================
# TAB 4 — Map
# ============================================
with tab4:
    st.header("4. Where do delays happen?")
    st.markdown("Line **IC 16-1** (Belgium) — Brussels ↔ Luxembourg corridor.")
    
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
    
    # Plotly 6.x API + explicit center
    fig = px.scatter_map(
        df_map,
        lat="lat", lon="lon", text="name",
        zoom=7, height=600,
        title="IC 16-1 line — main stations",
        center={"lat": 50.5, "lon": 5.0},
    )
    fig.update_traces(marker=dict(size=12, color='#1f77b4'))
    fig.update_layout(
        map_style="open-street-map",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("Station positions are approximate. Map shows the corridor, not all stops.")

# ============================================
# TAB 5 — Examples (Prediction vs Reality)
# ============================================
with tab5:
    st.header("5. Prediction vs Reality — Real Examples")
    st.markdown("""
    Examples from **25.09.2024, hour 8:00** — line IC 16-1.
    The model predicts the **probability** that a train arrives ≥6 minutes late.
    Below: what the model predicted vs what actually happened.
    """)
    
    # Real examples from the data
    examples = [
        {
            "Train": 2128, "Station": "BRUSSEL-CENTRAAL",
            "Planned": "08:23:00", "Actual": "08:29:36",
            "Delay_sec": 396, "prev_delay_arr": 362, "prev_delay_dep": 362,
            "Probability": 0.9410, "Prediction": "DELAYED",
            "Actual_outcome": "DELAYED", "Verdict": "TP ✅",
            "Note": "Strong signal: prev station was 6+ min late."
        },
        {
            "Train": 2128, "Station": "BRUSSEL-KAPELLEKERK",
            "Planned": "08:26:00", "Actual": "08:32:00",
            "Delay_sec": 360, "prev_delay_arr": 396, "prev_delay_dep": 415,
            "Probability": 0.8785, "Prediction": "DELAYED",
            "Actual_outcome": "DELAYED", "Verdict": "TP ✅",
            "Note": "Exactly on threshold; correctly predicted."
        },
        {
            "Train": 2108, "Station": "BRUSSEL-LUXEMBURG",
            "Planned": "08:54:00", "Actual": "09:01:08",
            "Delay_sec": 428, "prev_delay_arr": 276, "prev_delay_dep": 448,
            "Probability": 0.9982, "Prediction": "DELAYED",
            "Actual_outcome": "DELAYED", "Verdict": "TP ✅",
            "Note": "High confidence — very strong signal."
        },
        {
            "Train": 2128, "Station": "BRUSSEL-CONGRES",
            "Planned": "08:21:00", "Actual": "08:27:02",
            "Delay_sec": 362, "prev_delay_arr": 267, "prev_delay_dep": 358,
            "Probability": 0.6540, "Prediction": "ON TIME",
            "Actual_outcome": "DELAYED", "Verdict": "FN ❌",
            "Note": "Missed. Probability 0.65 < threshold 0.80."
        },
        {
            "Train": 2131, "Station": "ARLON",
            "Planned": "08:31:00", "Actual": "08:30:08",
            "Delay_sec": -51, "prev_delay_arr": 410, "prev_delay_dep": 410,
            "Probability": 0.7341, "Prediction": "ON TIME",
            "Actual_outcome": "ON TIME", "Verdict": "TN ✅",
            "Note": "Model correctly did NOT over-react — train recovered."
        },
        {
            "Train": 2107, "Station": "LA HULPE",
            "Planned": "08:08:00", "Actual": "08:09:04",
            "Delay_sec": 64, "prev_delay_arr": 146, "prev_delay_dep": 146,
            "Probability": 0.0058, "Prediction": "ON TIME",
            "Actual_outcome": "ON TIME", "Verdict": "TN ✅",
            "Note": "Clear signal — correctly predicted."
        },
    ]
    
    df_ex = pd.DataFrame(examples)
    st.dataframe(
        df_ex.style.format({
            'Probability': '{:.4f}',
            'Delay_sec': '{:,.0f}',
        }),
        use_container_width=True,
    )
    
    st.subheader("What this tells us")
    st.markdown("""
    - **TP cases (✅):** When the previous station was ≥6 minutes late, the model correctly predicted the delay. Probabilities are 0.88–0.99.
    - **FN cases (❌):** Train 2128 at BRUSSEL-CONGRES — the previous station was only slightly under the threshold (358s vs 360s), so the model gave probability 0.65 — just below the 0.80 threshold. The train arrived 6:02 late.
    - **TN with high prev_delay (✅):** Train 2131 at ARLON — previous station was 410s late, but this train recovered and arrived on time. The model **correctly** did not predict a delay (probability 0.73).
    
    → The model is **not** simply "copying prev_delay". It learned when recovery happens.
    """)
    
    st.subheader("Overall counts (Test — January 2025)")
    st.markdown("""
    - **TP = 4,347** — correctly predicted delays
    - **FN = 456** — missed delays
    - **FP = 247** — false alarms
    - **TN = 41,524** — correctly predicted on-time
    - **Precision = 94.6% · Recall = 90.5% · F1 = 92.5%**
    """)

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

### Links

- **GitHub:** [github.com/eliyahudahan/metrodorf](https://github.com/eliyahudahan/metrodorf)
- **License:** Infrabel Open Data (CC0).
""")
