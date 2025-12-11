import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os

# =========================
# 1. Configuration & Constants
# =========================
RAW_FEATURES = ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption']
FEATURES = [
    'temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption',
    'temp_moving_avg', 'vib_moving_avg', 'temp_rate_change', 'vib_rate_change'
]
SEQ_LEN = 20
MOVING_WIN = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# 2. Model Architecture
# =========================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=len(FEATURES), hidden_size=100, num_layers=3, dropout=0.3):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# =========================
# 3. Helper Functions
# =========================

def init_last_raw_sequences_from_df(df, raw_features, seq_len_minus_one=SEQ_LEN-1):
    """ Initialize historical sequences for prediction """
    last_raw = {}
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    
    for m_id in df['machine_id'].unique():
        md = df[df['machine_id'] == m_id]
        arr = md[raw_features].values
        
        if arr.shape[0] >= seq_len_minus_one:
            last_raw[m_id] = arr[-seq_len_minus_one:]
        else:
            pad_count = seq_len_minus_one - arr.shape[0]
            if arr.shape[0] == 0:
                last_raw[m_id] = np.zeros((seq_len_minus_one, len(raw_features)))
            else:
                pad = np.repeat(arr[0:1, :], pad_count, axis=0)
                last_raw[m_id] = np.vstack([pad, arr])
    return last_raw

def get_machine_stats(df, machine_id):
    """
    Calculate SAFE RANGE (Min & Max) for a specific machine.
    Based strictly on 'Normal' operation history.
    """
    machine_data = df[df['machine_id'] == machine_id]
    
    # Filter for HEALTHY states only
    if 'failure_type' in df.columns:
        healthy_data = machine_data[machine_data['failure_type'] == 'Normal']
    else:
        healthy_data = machine_data 

    # Safety fallback
    if healthy_data.empty:
        healthy_data = machine_data

    stats = {}
    
    # Calculate Range [Min, Max] for sensors
    for col in RAW_FEATURES:
        if col in machine_data.columns:
            stats[col] = {
                'min_healthy': healthy_data[col].min(), # Lower Limit
                'max_healthy': healthy_data[col].max(), # Upper Limit
                'last_recorded': machine_data[col].iloc[-1]
            }
            
    # RUL stats
    if 'predicted_remaining_life' in machine_data.columns:
        stats['rul_max'] = machine_data['predicted_remaining_life'].max()
    else:
        stats['rul_max'] = 100.0

    return stats

def predict_logic(model, scaler, full_seq_raw):
    """ Features Engineering + Prediction """
    temp_df = pd.DataFrame(full_seq_raw, columns=RAW_FEATURES)
    
    temp_moving_avg = temp_df['temperature'].rolling(window=MOVING_WIN, min_periods=1).mean().values
    vib_moving_avg = temp_df['vibration'].rolling(window=MOVING_WIN, min_periods=1).mean().values
    
    temp_rate_change = np.diff(temp_df['temperature'].values, prepend=temp_df['temperature'].values[0])
    vib_rate_change = np.diff(temp_df['vibration'].values, prepend=temp_df['vibration'].values[0])
    
    final_features = np.hstack([
        full_seq_raw,
        temp_moving_avg.reshape(-1, 1),
        vib_moving_avg.reshape(-1, 1),
        temp_rate_change.reshape(-1, 1),
        vib_rate_change.reshape(-1, 1)
    ]) 
    
    seq_scaled = scaler.transform(final_features)
    seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        prediction = model(seq_tensor)
        
    return float(prediction.cpu().numpy().squeeze())

# =========================
# 4. Streamlit UI (Main App)
# =========================
def main():
    st.set_page_config(page_title="Safe-Range RUL Monitor", layout="wide")
    
    st.title("🛡️ Full Range Anomaly Monitor")
    st.markdown("### AI RUL Prediction & Out-of-Range Detection")
    st.info("ℹ️ **Logic:** Alerts trigger if input is **Higher** than historical Max OR **Lower** than historical Min (while machine was Normal).")

    # --- Sidebar ---
    st.sidebar.header("⚙️ Configuration")
    uploaded_model = st.sidebar.file_uploader("Upload Model (.pth)", type=['pth'])
    uploaded_scaler = st.sidebar.file_uploader("Upload Scaler (.pkl)", type=['pkl'])
    uploaded_csv = st.sidebar.file_uploader("Upload Data (.csv)", type=['csv'])

    model = None
    scaler = None
    
    # Loaders
    if uploaded_model and uploaded_scaler:
        try:
            with open("temp_model.pth", "wb") as f: f.write(uploaded_model.getbuffer())
            with open("temp_scaler.pkl", "wb") as f: f.write(uploaded_scaler.getbuffer())
            with open("temp_scaler.pkl", "rb") as f: scaler = pickle.load(f)
            model = LSTMRegressor()
            model.load_state_dict(torch.load("temp_model.pth", map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            st.sidebar.success("✅ Model Ready")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            st.stop()

    if uploaded_csv:
        if 'current_csv_name' not in st.session_state or st.session_state.current_csv_name != uploaded_csv.name:
            try:
                df = pd.read_csv(uploaded_csv)
                st.session_state.df_full = df 
                st.session_state.last_raw_sequences = init_last_raw_sequences_from_df(df, RAW_FEATURES)
                st.session_state.machine_ids = list(st.session_state.last_raw_sequences.keys())
                st.session_state.current_csv_name = uploaded_csv.name
                st.success(f"✅ Data Loaded")
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
    
    if not (model and scaler and 'last_raw_sequences' in st.session_state):
        st.warning("👈 Please upload files in the sidebar.")
        st.stop()

    # --- Main Interface ---
    st.markdown("---")
    
    # Machine Selection
    col_sel, col_empty = st.columns([1, 2])
    with col_sel:
        selected_machine = st.selectbox("Select Target Machine ID", st.session_state.machine_ids)
    
    stats = get_machine_stats(st.session_state.df_full, selected_machine)
    
    # Input Section with Logic
    st.subheader("📝 Live Sensor Input")
    st.caption("Values outside the [Min, Max] history of this machine will trigger an alert.")

    input_cols = st.columns(5)
    inputs = {}
    labels = ['Temperature', 'Vibration', 'Humidity', 'Pressure', 'Energy']
    keys = ['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption']
    
    warnings = [] 
    
    for i, (col, label, key) in enumerate(zip(input_cols, labels, keys)):
        with col:
            default_val = float(stats[key]['last_recorded'])
            val = st.number_input(label, value=default_val)
            inputs[key] = val
            
            # --- FULL RANGE LOGIC (Min & Max) ---
            min_limit = stats[key]['min_healthy']
            max_limit = stats[key]['max_healthy']
            
            if val > max_limit:
                st.error(f"⚠️ High!")
                st.caption(f"Max Allowed: {max_limit:.2f}")
                warnings.append(f"**{label}** is TOO HIGH (Value: {val} > Max: {max_limit:.2f})")
            
            elif val < min_limit:
                st.error(f"⚠️ Low!")
                st.caption(f"Min Allowed: {min_limit:.2f}")
                warnings.append(f"**{label}** is TOO LOW (Value: {val} < Min: {min_limit:.2f})")
                
            else:
                st.success(f"✅ Normal")
                st.caption(f"Range: {min_limit:.1f} - {max_limit:.1f}")

    # Display Alerts Summary
    if warnings:
        st.error("🚨 **ABNORMAL CONDITIONS DETECTED!**")
        for w in warnings:
            st.write(f"- {w}")

    st.markdown("---")

    # Prediction Button
    if st.button("🚀 Predict RUL"):
        try:
            prev_raw = st.session_state.last_raw_sequences[selected_machine]
            new_raw = np.array([inputs[k] for k in RAW_FEATURES]).reshape(1, -1)
            full_seq_raw = np.vstack([prev_raw, new_raw])
            
            pred_rul = predict_logic(model, scaler, full_seq_raw)
            st.session_state.last_raw_sequences[selected_machine] = full_seq_raw[1:]

            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric("🔮 Predicted RUL", f"{pred_rul:.2f} Cycles")
                
            with res_col2:
                max_rul_hist = stats['rul_max']
                if max_rul_hist > 0:
                    health_pct = min(pred_rul / max_rul_hist, 1.0)
                else: 
                    health_pct = 0.0
                
                st.write(f"**Machine Health Status:** {health_pct*100:.1f}%")
                
                if warnings:
                    st.progress(health_pct)
                    st.warning("⚠️ Prediction made with out-of-range inputs.")
                elif pred_rul < 20:
                    st.progress(health_pct)
                    st.error("🚨 CRITICAL: High failure risk detected!")
                else:
                    st.progress(health_pct)
                    st.success("✅ Machine Operating Correctly.")

            with st.expander("📊 Inspect Input Sequence (Last 20 Data Points)"):
                df_view = pd.DataFrame(full_seq_raw, columns=RAW_FEATURES)
                df_view['Type'] = ['History'] * 19 + ['🆕 New Input']
                
                def style_last_row(row):
                    if row.name == 19:
                        return ['background-color: #ffffcc; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(df_view.style.apply(style_last_row, axis=1))

        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == '__main__':
    main()