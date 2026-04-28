"""Streamlit dashboard home page."""
import os

import requests
import streamlit as st

SERVING_URL = os.getenv("SERVING_URL", "http://serving:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow:5000")
MONITORING_URL = os.getenv("MONITORING_URL", "http://monitoring:8002")

st.set_page_config(page_title="Fraud Detection", page_icon=":shield:", layout="wide")

st.title("Real-Time Fraud Detection Pipeline")
st.caption("End-to-end ML pipeline — IEEE-CIS Fraud Detection")

st.markdown(
    """
Use the sidebar to navigate:

- **Predictions** — submit a transaction, get fraud probability + SHAP explanation
- **Model Performance** — metrics, ROC/PR curves, confusion matrix
- **Drift Monitoring** — Evidently drift report
- **Explainability** — global SHAP feature importance
"""
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Serving API")
    try:
        r = requests.get(f"{SERVING_URL}/health", timeout=3)
        data = r.json()
        st.success("Online")
        st.write(f"Model loaded: **{data.get('model_loaded')}**")
        st.write(f"Model: **{data.get('model_name', 'n/a')}**")
    except Exception as e:
        st.error(f"Unreachable: {e}")

with col2:
    st.subheader("Monitoring")
    try:
        r = requests.get(f"{MONITORING_URL}/health", timeout=3)
        if r.ok:
            st.success("Online")
        else:
            st.warning(r.status_code)
    except Exception as e:
        st.error(f"Unreachable: {e}")

with col3:
    st.subheader("MLflow")
    st.markdown(f"[Open MLflow UI]({MLFLOW_URL})")
    st.caption("Experiment tracking + model registry")

st.divider()
st.markdown(
    """
### Quick start
1. Place IEEE-CIS CSVs in `data/raw/`
2. `docker compose up -d --build`
3. Trigger training: `curl -X POST http://localhost:8001/train`
4. Once training finishes, submit a transaction on the **Predictions** page.
"""
)
