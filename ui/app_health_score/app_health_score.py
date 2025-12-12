# app.py - Final Version with Live Sensor Editing + Fast Trend
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import plotly.graph_objects as go

# ==============================================
# Page Config
# ==============================================
st.set_page_config(page_title="AI Machine Health Monitor", layout="wide", page_icon="robot")

st.markdown("""
    <h1 style='text-align: center; color: #00BFFF;'>AI Machine Health Monitor</h1>
    <h3 style='text-align: center; color: #888;'>Live Sensor Editing & Real-time Health Prediction</h3>
    <hr style='border: 3px solid #00BFFF;'>
""", unsafe_allow_html=True)

# ==============================================
# Model Architecture (مطابق للنموذج النهائي بتاعك)
# ==============================================
class FinalLSTMHealth(nn.Module):
    def __init__(self):
        super(FinalLSTMHealth, self).__init__()
        self.lstm = nn.LSTM(9, 182, num_layers=1, batch_first=True, dropout=0.0)
        self.fc = nn.Sequential(
            nn.Linear(182, 64),
            nn.ReLU(),
            nn.Dropout(0.217),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]) * 100

# ==============================================
# Sidebar Upload
# ==============================================
st.sidebar.header("Upload Files")
uploaded_model = st.sidebar.file_uploader("Model (.pth)", type="pth")
uploaded_scaler = st.sidebar.file_uploader("Scaler (.pkl)", type="pkl")
uploaded_data = st.sidebar.file_uploader("Data (.csv)", type="csv")

if not all([uploaded_model, uploaded_scaler, uploaded_data]):
    st.info("Please upload all three files.")
    st.stop()

# ==============================================
# Load Everything
# ==============================================
@st.cache_resource
def load_resources(m_file, s_file, csv_file):
    with open("temp_model.pth", "wb") as f:
        f.write(m_file.getbuffer())
    with open("temp_scaler.pkl", "wb") as f:
        f.write(s_file.getbuffer())

    model = FinalLSTMHealth()
    model.load_state_dict(torch.load("temp_model.pth", map_location='cpu'))
    model.eval()

    with open("temp_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return model, scaler, df

with st.spinner("Loading model and data..."):
    model, scaler, df = load_resources(uploaded_model, uploaded_scaler, uploaded_data)

st.success("Loaded successfully!")

# ==============================================
# Feature Preparation
# ==============================================
def prepare_features(df_subset):
    df_m = df_subset.copy()
    df_m['temp_ma'] = df_m['temperature'].rolling(5, min_periods=1).mean()
    df_m['vib_ma'] = df_m['vibration'].rolling(5, min_periods=1).mean()
    df_m['temp_roc'] = df_m['temperature'].diff().fillna(0)
    df_m['vib_roc'] = df_m['vibration'].diff().fillna(0)
    features = ['temperature','vibration','humidity','pressure','energy_consumption',
                'temp_ma','vib_ma','temp_roc','vib_roc']
    return df_m[features]

# ==============================================
# Machine Selection
# ==============================================
machine_id = st.selectbox("Select Machine ID", sorted(df['machine_id'].unique()))

machine_data = df[df['machine_id'] == machine_id].copy()

if len(machine_data) < 20:
    st.error("Not enough data for this machine")
    st.stop()

# ==============================================
# Live Sensor Editing (آخر قراءة + تعديل)
# ==============================================
st.markdown("### Live Sensor Readings (Edit values and click Predict)")
last_row = machine_data.iloc[-1]

cols = st.columns(5)
sensor_labels = ["Temperature", "Vibration", "Humidity", "Pressure", "Energy Consumption"]
sensor_keys = ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption']
inputs = {}

for col, label, key in zip(cols, sensor_labels, sensor_keys):
    with col:
        default = float(last_row[key])
        val = st.number_input(label, value=default, step=0.01, format="%.4f", key=f"input_{key}")
        inputs[key] = val

# ==============================================
# Predict Button
# ==============================================
if st.button("Predict Current Health Score", type="primary", use_container_width=True):
    with st.spinner("Predicting health..."):
        # Build sequence: last 19 historical + 1 new edited
        hist = machine_data[sensor_keys].tail(19).values
        new = np.array([[inputs[k] for k in sensor_keys]])
        full_seq = np.vstack([hist, new])

        # Feature engineering
        df_seq = pd.DataFrame(full_seq, columns=sensor_keys)
        df_seq['temp_ma'] = df_seq['temperature'].rolling(5, min_periods=1).mean()
        df_seq['vib_ma'] = df_seq['vibration'].rolling(5, min_periods=1).mean()
        df_seq['temp_roc'] = df_seq['temperature'].diff().fillna(0)
        df_seq['vib_roc'] = df_seq['vibration'].diff().fillna(0)

        feats = ['temperature','vibration','humidity','pressure','energy_consumption',
                 'temp_ma','vib_ma','temp_roc','vib_roc']

        X = df_seq[feats].values
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            health_score = model(X_tensor).item()

        # Trend on last 20 readings (real model predictions)
        last_20 = machine_data.tail(20).copy()
        last_20.loc[last_20.index[-1], sensor_keys] = [inputs[k] for k in sensor_keys]  # update last row with edited values

        trend_features = prepare_features(last_20)
        trend_scaled = scaler.transform(trend_features.values)
        trend_tensor = torch.tensor(trend_scaled, dtype=torch.float32).unsqueeze(0)

        # Predict trend for last 20 points
        trend_scores = []
        for i in range(19, len(trend_scaled)):
            seq = trend_scaled[i-19:i+1]
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                trend_scores.append(model(seq_tensor).item())

        timestamps = last_20['timestamp'].values[19:]

        # Display
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"### Machine {machine_id}")
            st.metric("Current Health Score", f"{health_score:.1f}/100", delta=f"{health_score - 85:+.1f}")
            st.progress(health_score / 100)

            if health_score >= 85:
                st.success("Excellent Condition")
            elif health_score >= 70:
                st.warning("Warning – Schedule Maintenance")
            else:
                st.error("CRITICAL – Immediate Action Required!")

        with col2:
            st.markdown("### Health Trend (Last 20 Readings)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=trend_scores,
                mode='lines+markers',
                name='Health Score',
                line=dict(color='#00BFFF', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=[timestamps[-1]],
                y=[health_score],
                mode='markers',
                marker=dict(color='red', size=15, symbol='star'),
                name='Latest Prediction'
            ))
            fig.add_hline(y=85, line_dash="dash", line_color="orange", annotation_text="Warning")
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Critical")
            fig.update_layout(height=500, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

st.caption("Edit sensor values and click Predict to see the impact on machine health in real-time")
