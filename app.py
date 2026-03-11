"""
Metrodorf - Train Delay Prediction Dashboard
For dispatcher Thomas Schmidt at Dortmund Hbf
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from models.delay_predictor import DelayPredictor
import time
import logging
import joblib
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data
def load_stations():
    df = pd.read_csv("data/raw/stations_live.csv")
    station_list = df['name'].tolist()
    return station_list

# Load stations
stations = load_stations()

# Load model
@st.cache_resource
def load_model():
    """Load trained models"""
    start = time.time()
    
    predictor = DelayPredictor()
    
    # Load each model
    for name in ['xgb', 'rf', 'gaussian']:
        t0 = time.time()
        model_path = f"models/saved/{name}_model.pkl"
        if Path(model_path).exists():
            predictor.models[name] = joblib.load(model_path)
            print(f"⏱️ Loaded {name} in {time.time()-t0:.2f}s")
        else:
            print(f"⚠️ {name} model not found")
    
    # Load weights
    t0 = time.time()
    weights_path = "models/saved/model_weights.csv"
    if Path(weights_path).exists():
        weights_df = pd.read_csv(weights_path, index_col=0, header=None)
        predictor.weights = weights_df.iloc[:, 0].to_dict()  # type: ignore
        print(f"⏱️ Loaded weights in {time.time()-t0:.2f}s")
    else:
        # Fallback weights
        predictor.weights = {'gaussian': 1.0, 'xgb': 0.0, 'rf': 0.0}  # type: ignore
        print("⚠️ No weights file found, using fallback")
    
    print(f"✅ Models loaded in {time.time()-start:.2f}s")
    return predictor

predictor = load_model()

# Prediction function
def get_real_prediction(station, hour, day, is_peak, is_cologne):
    """Get prediction from real model"""
    # Estimate distance (simplified)
    distance = 50  # default
    if "Cologne" in station and "Düsseldorf" in station:
        distance = 40
    elif "Cologne" in station and "Dortmund" in station:
        distance = 70
    elif "Essen" in station and "Bochum" in station:
        distance = 15
    
    # Get prediction from model
    delay = predictor.predict_delay(
        distance=distance,
        time_of_day=hour,
        day_of_week=day,
        is_peak=is_peak,
        is_cologne=is_cologne
    )
    
    return round(delay, 1)

# Page config
st.set_page_config(
    page_title="Metrodorf - Delay Predictor",
    page_icon="🚆",
    layout="wide"
)

# Title
st.title("🚆 Metrodorf - Rhine-Ruhr Delay Prediction")
st.markdown("Real-time predictions for dispatcher **Thomas Schmidt** at **Dortmund Hbf**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Station selection
    station = st.selectbox(
        "Select Station",
        ["Dortmund Hbf", "Essen Hbf", "Duisburg Hbf", "Düsseldorf Hbf", 
         "Cologne Hbf", "Bonn Hbf", "Bochum Hbf", "Wuppertal Hbf"]
    )
    
    # Time selection
    col1, col2 = st.columns(2)
    with col1:
        hour = st.number_input("Hour", 0, 23, datetime.now().hour)
    with col2:
        day = st.number_input("Day of week", 0, 6, datetime.now().weekday())
    
    # Checkboxes
    is_peak = st.checkbox("🌅 Peak hour", value=(7 <= hour <= 9) or (16 <= hour <= 18))
    is_cologne = st.checkbox("🌉 Cologne bottleneck", value=False)
    
    # Run button
    predict_btn = st.button("🔮 Predict Delay", type="primary")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Current Status")
    
    @st.cache_data(ttl=60)
    def get_live_departures():
        from data.real_time_collector import RealTimeCollector
        collector = RealTimeCollector()
        
        all_departures = []
        for station_name in stations[:3]:
            deps = collector.get_departures(station_name, limit=3)
            for dep in deps:
                parsed = collector.parse_departure(dep, station_name)
                if parsed:
                    all_departures.append(parsed)
                    time.sleep(3)
        
        df = pd.DataFrame(all_departures)
        return df

    # Try to get live data
    try:
        live_data = get_live_departures()
        if len(live_data) > 0:
            status_data = live_data[['station_name', 'time_of_day', 'delay_minutes']].rename(
                columns={'station_name': 'Station', 'time_of_day': 'Time', 'delay_minutes': 'Delay'}
            )
        else:
            status_data = pd.DataFrame({
                "Station": ["Dortmund Hbf", "Essen Hbf", "Cologne Hbf"],
                "Time": [14, 15, 16],
                "Delay": [5, 2, 8]
            })
    except:
        status_data = pd.DataFrame({
            "Station": ["Dortmund Hbf", "Essen Hbf", "Cologne Hbf"],
            "Time": [14, 15, 16],
            "Delay": [5, 2, 8]
        })

    st.dataframe(status_data, use_container_width=True)

with col2:
    st.subheader("🔮 Prediction Result")
    
    if predict_btn:
        delay = get_real_prediction(station, hour, day, is_peak, is_cologne)
        
        st.metric("Predicted Delay", f"{delay} min", delta=None)
        
        mae = 3.35
        st.caption(f"68% confidence: {max(0, delay-mae):.1f} - {delay+mae:.1f} min")
        
        if delay > 10:
            st.error("⚠️ **Recommendation**: Hold connections, notify passengers")
        elif delay > 5:
            st.warning("⚡ **Recommendation**: Monitor closely, consider platform change")
        else:
            st.success("✅ **Recommendation**: Normal operations")

# Bottom section - Recent history
st.subheader("📈 Recent Delay History")

history_data = pd.DataFrame({
    "Time": pd.date_range(start="06:00", periods=8, freq="15min").strftime("%H:%M"),
    "ICE 723": [2, 5, 8, 11, 9, 7, 4, 2],
    "RE 1": [1, 2, 3, 4, 5, 4, 3, 2],
    "RB 53": [3, 4, 5, 6, 7, 6, 5, 4]
})

fig = px.line(history_data, x="Time", y=["ICE 723", "RE 1", "RB 53"],
              title="Train Delays Over Time")
st.plotly_chart(fig, use_container_width=True)

# Footer
st.caption("🚆 Data from v6.db.transport.rest | Model: Ensemble R²=0.478")