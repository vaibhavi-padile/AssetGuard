import streamlit as st
import pandas as pd
import numpy as np
from ingestion_validator import process_sensor_telemetry
from model import train_maintenance_model

st.set_page_config(page_title="AssetGuard IQ Terminal", layout="wide", initial_sidebar_state="collapsed")

# Custom injection of high-end dark industrial interface architecture
st.markdown("""
    <style>
        @import url('https://googleapis.com');
        
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0d0f12;
            color: #d1d5db;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
        
        .terminal-header {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 300;
            letter-spacing: -0.05em;
            color: #ffffff;
            font-size: 2rem;
            margin-bottom: 0.2rem;
        }
        .terminal-subheader {
            font-family: 'JetBrains Mono', monospace;
            color: #4b5563;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin-bottom: 3rem;
        }
        .metric-card {
            background: #13171e;
            border-left: 2px solid #d97706;
            padding: 1.5rem;
            border-radius: 2px;
        }
        .metric-label {
            font-family: 'JetBrains Mono', monospace;
            color: #6b7280;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            color: #ffffff;
            font-size: 1.75rem;
            font-weight: 400;
            margin-top: 0.5rem;
        }
        .status-nominal {
            color: #10b981;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }
        .status-critical {
            color: #ef4444;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="terminal-header">ASSETGUARD IQ</div>', unsafe_allow_html=True)
st.markdown('<div class="terminal-subheader">Predictive Telemetry Processing & Anomaly Pipeline</div>', unsafe_allow_html=True)

# Process Pipeline
df = process_sensor_telemetry("data/sensor_telemetry.csv")
model, metrics, features = train_maintenance_model(df)

df['failure_probability'] = model.predict_proba(df[features])[:, 1]

# High-End Metric Row Display
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Nodes Monitored</div><div class="metric-value">MCH-901 Active</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Model Capture Rate (Recall)</div><div class="metric-value">{metrics["recall"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Active Outage Threats</div><div class="metric-value">{int((df["failure_probability"] > 0.80).sum())}</div></div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Structured Output Display Matrix
display_df = df[['timestamp', 'temperature', 'vibration_amplitude', 'failure_probability']].tail(15).copy()
display_df['status'] = np.where(display_df['failure_probability'] > 0.80, "CRITICAL RISK DETECTED", "NOMINAL OPERATIONAL RANGE")

st.dataframe(
    display_df.style.format({'failure_probability': '{:.4f}', 'temperature': '{:.2f}', 'vibration_amplitude': '{:.3f}'}),
    use_container_width=True
)
