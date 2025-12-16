# Predictive Maintenance for Industrial Equipment
A Data-Driven Approach to Forecast Failures, Reduce Downtime, and Improve Operational Efficiency

## 🏭 Project Overview
This project builds a complete Predictive Maintenance System for industrial machines using IoT sensor data. Our goal is to detect failures early, predict machine degradation, and estimate remaining useful life (RUL) — enabling smarter maintenance decisions and preventing costly downtime.

The system integrates time-series modeling, deep learning, and classification algorithms to provide actionable insights for industrial operations.

## 📊 Business Problem
Manufacturing machines often fail without warning, leading to:
- Costly unplanned downtime  
- High emergency repair costs  
- Overloaded maintenance teams  
- Lack of visibility into machine health  
- No data-driven maintenance planning  

Predictive Maintenance shifts strategy from reactive → proactive by using real sensor data to predict failures before they happen.

## 🗂️ Dataset Summary
We analyzed 100,000+ sensor records from 50 machines, each containing:
- Temperature  
- Vibration  
- Humidity  
- Energy Consumption  
- Pressure  
- Failure Type  
- Maintenance Required  
- Timestamp  

Engineered features:
- Time-based features  
- Rolling statistics  
- Rate-of-change  
- PCA components  
- Anomaly-based Z-scores  

## 🔧 Preprocessing & Feature Engineering
Key steps:
- Timestamp cleaning & sorting  
- Rolling averages (5-step)  
- Rate-of-change (diff)  
- PCA decomposition  
- Z-score anomaly detection  
- Outlier awareness (no clipping applied)

Main insights:
- Vibration spikes & temperature drift = strongest failure indicators  
- Pressure provides moderate insight  
- Energy & humidity have limited influence  
- Maintenance machines show higher sensor averages  

## 🤖 Modeling Overview

### 1️⃣ XGBoost Classification Models

#### Maintenance Required (Binary)
Predicts whether a machine is at risk.  
- Handles missing values internally  
- Captures nonlinear behavior  

#### Failure Type (Multiclass)
Identifies what type of failure is likely.  
- Supports faster repairs  
- Uses class balancing  

### 2️⃣ LSTM Remaining Useful Life (RUL) Model
Uses 20-step time-series windows to estimate how long the machine can run before failure.

The model learns:
- Vibration spikes → sharp RUL drops  
- Temperature drift → slow long-term wear  
- ROC features → sudden degradation  

### 3️⃣ LSTM Health Score Model (Final Best Model)
Outputs a 0–100 continuous machine-health score.

Built using:
- Z-score degradation  
- Rolling smoothing  
- LSTM (182 units, optimized via Optuna)  
- Adam optimizer (lr = 0.00054)

Performance:
- R² > 0.90  
- Low MAE  
- Smooth, stable predictions  

## 🖥️ Deployment (Streamlit App)
The dashboard provides:
- Real-time sensor entry  
- Safe-range violation alerts  
- RUL predictions  
- Health score estimation  
- Last-20-step sequence preview  

Uses:
```
FINAL_BEST_HEALTH_MODEL.pth
final_scaler_health.pkl
```

## 📁 Repository Structure
```
Predictive-Maintenance-Project/
│
├── data/
│   ├── preprocessed_smart_data.csv
│   ├── smart_manufacturing_data_original.csv
│
├── deployment/
│   ├── app_RUL.txt
│   ├── app_health_score.txt
│
├── models/
│   ├── columns.pkl
│   ├── model_Maintenance_Required.pkl
│   ├── model_RUL_LSTM.pth
│   ├── model_health_score.pth
│
├── notebooks/
│   ├── 01_EDA & Preprocessing.ipynb
│   ├── 02_Modeling_RUL.ipynb
│   ├── 03_Modeling_Health_Score.ipynb
│   ├── 04_Modeling_Maintenance_Required.ipynb
│   ├── 05_Modeling_Failure_Type.ipynb
│
├── reports/
│   ├── 01_Dataset Exploration Report.pdf
│   ├── 02_Advanced Analysis Report.pdf
│   ├── 03_Model Evaluation Report.pdf
│   ├── Predictive Maintenance Project Report.pdf
│
├── ui/
│   ├── app_RUL/
│   │   ├── app.py
│   │   ├── model_RUL_LSTM.pth
│   │   ├── scaler.pkl
│   │
│   ├── app_health_score/
│       ├── app_health_score.py
│       ├── model_health_score.pth
│
└── Predictive Maintenance for Industrial Equipment Project Presentation.pdf 
│
└── README.md
│
├── requirements.txt
```

## 🚀 Key Project Outcomes
| Technology | Output | Business Value |
|-----------|--------|----------------|
| Maintenance Classification | “Is the machine at risk?” | Early alerts |
| Failure Type Classification | “What is going wrong?” | Faster repairs |
| RUL Prediction | “How long until failure?” | Maintenance scheduling |
| Health Score Model | Real-time machine condition | Continuous monitoring |

## 👩‍💻 Team Members
- Rawan Essam
- Mohamed Zakaria
- Mostafa Kamel 
- Mohamed Sobhy 
- Rawan Tarek
- Habiba Ashraf
