# AI-powered-Predictive-maintenance-for-industrial-equipment-
DEPI graduation project
## 📊 Dataset Overview

The Smart Manufacturing IoT-Cloud Monitoring Dataset is designed to facilitate real-time process monitoring, predictive maintenance, and anomaly detection in industrial environments. The dataset simulates IoT sensor readings collected from manufacturing machines, including parameters such as temperature, vibration, humidity, pressure, and energy consumption.

This dataset helps researchers and engineers develop machine learning models for predictive maintenance, reducing downtime and optimizing operational efficiency. It also includes anomaly flags, failure types, and a target column (maintenance_required), which indicates whether a machine requires maintenance. 

| Type | Count |
|------|-------|
Total rows | **100,000**  
Total columns | **13**  
Numeric columns | **11**  
Categorical columns | **2**  
Timestamp column | **1**  

---

## 🧾 Columns Description

| Column | Type | Description |
|--------|------|-------------|
`timestamp` | datetime | Timestamp of the machine reading (minute-by-minute log). |
`machine_id` | int | Unique identifier for each machine (1–50). |
`temperature` | float | Machine operating temperature (°C). |
`vibration` | float | Vibration intensity level. |
`humidity` | float | Humidity level (%) in the machine environment. |
`pressure` | float | Pressure reading inside the machine system. |
`energy_consumption` | float | Energy consumed (kWh). |
`machine_status` | int (0–2) | Machine state: **0 = Idle, 1 = Running, 2 = Maintenance**. |
`anomaly_flag` | binary | AI-detected anomaly (**0 = Normal, 1 = Suspicious behavior**). |
`predicted_remaining_life` | int | Estimated Remaining Useful Life (RUL) of the machine. |
`failure_type` | category | Failure category (e.g., *Normal, Vibration Issue,* etc.). |
`downtime_risk` | float | Risk score of potential downtime (0–1). |
`maintenance_required` | binary | **1 = Maintenance required, 0 = No maintenance**. |

---

## 📐 Statistical Summary — Key Insights

- **Temperature:** mean ≈ 75°C, max ≈ 122°C → High-temperature operations  
- **Vibration:** mean ≈ 50, range **–17 to 114** → Negative values = possible noise/calibration issues  
- **Humidity:** stable industrial range, **30%–80%**  
- **Energy Consumption:** mean ≈ 2.75 kWh, max ≈ 5 → Consistent usage  
- **RUL:** ranges from **1 to 499 hours**  
- **Downtime Risk:** mean ≈ 0.089 → Mostly low-risk operations  
- **Anomaly & Maintenance Flags:** rare events → **class imbalance**

---
## 📝 Data Notes

- ✅ **No missing values**
- 🕒 **Time-series data (1-minute resolution)**
- 🎯 Contains **machine health indicators + failure labels**
- Suitable for:

| Task | Use Case |
|------|---------|
Predictive Maintenance | Forecast machine failures |
Anomaly Detection | Detect abnormal behavior |
Time-Series Modeling | LSTM, ARIMA |
Classification | Failure Type prediction |
Regression | RUL prediction |

---

## 🚀 Summary

This dataset provides comprehensive industrial machine telemetry for **predictive maintenance, anomaly detection, and reliability forecasting**.  
With sensor data, machine status, and failure labels, it is highly suitable for **supervised learning and time-series analysis**.
