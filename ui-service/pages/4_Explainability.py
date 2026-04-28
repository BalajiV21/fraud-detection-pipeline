"""SHAP global summary (uses precomputed plot from training-service)."""
from pathlib import Path

import streamlit as st

st.title("Explainability — Global Feature Importance")

# These plots are saved by the training service to the shared volume.
PLOTS_DIRS = [Path("/models").glob("plots_*")]

found = False
for d in Path("/models").glob("plots_*"):
    found = True
    st.subheader(d.name)
    for img in sorted(d.glob("*.png")):
        st.image(str(img), caption=img.name, use_column_width=True)

if not found:
    st.info("No training plots found yet. Run training first.")
