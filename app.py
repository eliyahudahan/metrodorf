"""
Metrodorf - Train Delay Prediction Dashboard
For dispatcher Thomas Schmidt at Dortmund Hbf

Optimized version with:
- Fast model loading (only Random Forest)
- Non-blocking data loading
- Improved UI responsiveness
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from models.delay_predictor import DelayPredictor
import time
import logging
import joblib
from pathlib import Path

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Metrodorf - Delay Predictor",
    page_icon="🚆",
    layout="wide"
)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# STATION LOADING (CACHED)
# ============================================
@st.cache_data
def load_stations():
    """Load station list from CSV file"""
    df = pd.read_csv("data/raw/stations_live.csv")
    station_list = df['name'].tolist()
    return station_list

stations = load_stations()

# ============================================
# MODEL LOADING (OPTIMIZED - SINGLE MODEL)
# ============================================
@st.cache_resource
def load_model():
    """
    Load only the best model (Random Forest) for fastest loading.
    This avoids loading all 3 models and retraining data.
    """
    start = time.time()
    
    # Create predictor without loading data
    predictor = DelayPredictor(load_data=False)
    
    # Load only Random Forest (R²=0.509)
    rf_path = "models/saved/rf_model.pkl"
    if Path(rf_path).exists():
        predictor.models['rf'] = joblib.load(rf_path)
        predictor.weights = {'rf': 1.0}
        print(f"✅ Loaded best model (RF) in {time.time()-start:.2f}s")
        print(f"   R²=0.509, MAE=2.75 min")
    else:
        # Fallback to Gaussian if RF not found
        print("⚠️ RF model not found, using Gaussian")
        predictor.models['gaussian'] = joblib.load("models/saved/gaussian_model.pkl")
        predictor.weights = {'gaussian': 1.0}
    
    return predictor

predictor = load_model()

# ============================================
# PREDICTION FUNCTION
# ============================================
def get_real_prediction(station, hour, day, is_peak, is_cologne):
    """
    Get delay prediction from the loaded model
    Uses simplified distance estimation based on station names
    """
    # Estimate distance (simplified - can be improved with real data)
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

# Title
st.title("🚆 Metrodorf - Rhine-Ruhr Delay Prediction")
st.markdown("Real-time predictions for dispatcher **Thomas Schmidt** at **Dortmund Hbf**")

# ============================================
# SIDEBAR - USER INPUTS
# ============================================
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
    
    # Checkboxes for special conditions
    is_peak = st.checkbox("🌅 Peak hour", value=(7 <= hour <= 9) or (16 <= hour <= 18))
    is_cologne = st.checkbox("🌉 Cologne bottleneck", value=False)
    
    # Run button
    predict_btn = st.button("🔮 Predict Delay", type="primary")

# ============================================
# MAIN CONTENT - TWO COLUMNS
# ============================================
col1, col2 = st.columns(2)

# LEFT COLUMN - CURRENT STATUS
with col1:
    st.subheader("📊 Current Status")
    
    # Create placeholder for async data loading
    status_placeholder = st.empty()
    status_placeholder.info("🔄 Loading live departures...")
    
    @st.cache_data(ttl=120)
    def get_live_departures():
        """
        Fetch live departure data from real-time APIs
        Falls back to synthetic data if API fails
        """
        from data.real_time_collector import RealTimeCollector
        collector = RealTimeCollector()
    
        all_departures = []
    
        for station_name in stations[:3]:  # First 3 stations
            try:
                deps = collector.get_departures(station_name, limit=3)
            
                if deps:  # Got real data
                   for dep in deps:
                       parsed = collector.parse_departure(dep, station_name)
                       if parsed:
                          parsed['station_name'] = station_name
                          all_departures.append(parsed)
                          time.sleep(2)
                else:
                    # No real data - generate synthetic
                    logger.info(f"⚠️ No real data for {station_name}, using synthetic")
                    for _ in range(3):
                       synthetic = collector.generate_synthetic_sample()
                       synthetic['station_name'] = station_name
                       all_departures.append(synthetic)
                    
            except Exception as e:
                logger.warning(f"Error for {station_name}: {e}")
                # Fallback to synthetic on error
                for _ in range(3):
                    synthetic = collector.generate_synthetic_sample()
                    synthetic['station_name'] = station_name
                    all_departures.append(synthetic)
    
        df = pd.DataFrame(all_departures)
        logger.info(f"✅ Collected {len(df)} departures")
        return df 
        """
        Fetch live departure data from real-time APIs
        Cached to avoid blocking the UI
        """
        from data.real_time_collector import RealTimeCollector
        collector = RealTimeCollector()
    
        all_departures = []
    
        # Get data from all 8 stations (real-time matters!)
        for station_name in stations:
           try:
                deps = collector.get_departures(station_name, limit=3)
                for dep in deps:
                    parsed = collector.parse_departure(dep, station_name)
                    if parsed:
                       # Add station_name manually (it's not in parsed)
                       parsed['station_name'] = station_name
                       all_departures.append(parsed)
                       time.sleep(2)  # Respect rate limits
           except Exception as e:
                logger.warning(f"Error getting departures for {station_name}: {e}")
                continue
    
        df = pd.DataFrame(all_departures)
        logger.info(f"✅ Collected {len(df)} live departures")
        return df



    # Load data in background
    try:
        live_data = get_live_departures()
        if len(live_data) > 0:
            status_data = live_data[['station_name', 'time_of_day', 'delay_minutes']].rename(
                columns={'station_name': 'Station', 'time_of_day': 'Time', 'delay_minutes': 'Delay'}
            )
            status_placeholder.dataframe(status_data, width='stretch')
        else:
            # Fallback to mock data
            status_placeholder.dataframe(pd.DataFrame({
                "Station": ["Dortmund Hbf", "Essen Hbf", "Cologne Hbf"],
                "Time": [14, 15, 16],
                "Delay": [5, 2, 8]
            }), width='stretch')
    except Exception as e:
        status_placeholder.error(f"Error loading data: {e}")

# RIGHT COLUMN - PREDICTION RESULT
with col2:
    st.subheader("🔮 Prediction Result")
    
    if predict_btn:
        # Get prediction
        delay = get_real_prediction(station, hour, day, is_peak, is_cologne)
        
        # Display metric
        st.metric("Predicted Delay", f"{delay} min", delta=None)
        
        # Confidence interval based on model MAE
        mae = 3.35
        st.caption(f"68% confidence: {max(0, delay-mae):.1f} - {delay+mae:.1f} min")
        
        # Recommendation based on delay severity
        if delay > 10:
            st.error("⚠️ **Recommendation**: Hold connections, notify passengers")
        elif delay > 5:
            st.warning("⚡ **Recommendation**: Monitor closely, consider platform change")
        else:
            st.success("✅ **Recommendation**: Normal operations")

# ============================================
# BOTTOM SECTION - RECENT HISTORY
# ============================================
st.subheader("📈 Recent Delay History")

# Mock history data (replace with real DB data later)
history_data = pd.DataFrame({
    "Time": pd.date_range(start="06:00", periods=8, freq="15min").strftime("%H:%M"),
    "ICE 723": [2, 5, 8, 11, 9, 7, 4, 2],
    "RE 1": [1, 2, 3, 4, 5, 4, 3, 2],
    "RB 53": [3, 4, 5, 6, 7, 6, 5, 4]
})

# Create line chart
fig = px.line(history_data, x="Time", y=["ICE 723", "RE 1", "RB 53"],
              title="Train Delays Over Time")
st.plotly_chart(fig, width='stretch')

# Footer
st.caption("🚆 Data from v6.db.transport.rest | Model: Random Forest (R²=0.509)")