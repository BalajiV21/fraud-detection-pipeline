"""Embed the latest Evidently drift report."""
import os

import requests
import streamlit as st
import streamlit.components.v1 as components

MONITORING_URL = os.getenv("MONITORING_URL", "http://monitoring:8002")

st.title("Drift Monitoring")

if st.button("Run drift check now"):
    with st.spinner("Running drift check..."):
        try:
            r = requests.post(f"{MONITORING_URL}/drift-check", timeout=60)
            st.json(r.json())
        except Exception as e:
            st.error(str(e))

st.divider()
st.subheader("Latest report")
try:
    r = requests.get(f"{MONITORING_URL}/drift-report", timeout=10)
    if r.status_code == 404:
        st.info("No drift report yet. Click 'Run drift check now' above.")
    else:
        components.html(r.text, height=900, scrolling=True)
except Exception as e:
    st.error(f"Could not fetch report: {e}")
