"""Real-time prediction demo."""
import os

import pandas as pd
import requests
import streamlit as st

SERVING_URL = os.getenv("SERVING_URL", "http://serving:8000")

st.title("Real-time Prediction")
st.caption("Submit a transaction to the serving API and see the fraud probability + top contributing features.")

with st.form("txn_form"):
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("TransactionAmt (USD)", min_value=0.0, value=125.50, step=1.0)
        product = st.selectbox("ProductCD", ["W", "H", "C", "S", "R"])
        card4 = st.selectbox("Card brand (card4)", ["visa", "mastercard", "discover", "american express"])
        card6 = st.selectbox("Card type (card6)", ["debit", "credit"])
    with col2:
        p_email = st.text_input("Purchaser email domain (P_emaildomain)", "gmail.com")
        r_email = st.text_input("Recipient email domain (R_emaildomain)", "gmail.com")
        device_type = st.selectbox("DeviceType", ["desktop", "mobile", ""])
        device_info = st.text_input("DeviceInfo", "Windows")

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "TransactionAmt": amount,
        "TransactionDT": 86400 * 30,  # arbitrary
        "ProductCD": product,
        "card4": card4,
        "card6": card6,
        "P_emaildomain": p_email,
        "R_emaildomain": r_email,
        "DeviceType": device_type,
        "DeviceInfo": device_info,
    }
    try:
        r = requests.post(f"{SERVING_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        prob = data["fraud_probability"]
        st.metric("Fraud probability", f"{prob:.2%}")
        if data["is_fraud"]:
            st.error(f"FLAGGED AS FRAUD (threshold {data['threshold']:.2f})")
        else:
            st.success(f"Looks legitimate (threshold {data['threshold']:.2f})")
        st.caption(f"Latency: {data['prediction_time_ms']} ms — model: {data.get('model_name')}")
        if data["top_risk_factors"]:
            st.subheader("Top contributing features (SHAP)")
            df = pd.DataFrame(data["top_risk_factors"])
            st.bar_chart(df.set_index("feature"))
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
