"""Show metadata.json from the trained model + comparison table."""
import os

import pandas as pd
import requests
import streamlit as st

SERVING_URL = os.getenv("SERVING_URL", "http://serving:8000")

st.title("Model Performance")

try:
    r = requests.get(f"{SERVING_URL}/model/info", timeout=5)
    if r.status_code == 503:
        st.warning("Model not loaded yet. Run training first.")
        st.stop()
    r.raise_for_status()
    meta = r.json()
except Exception as e:
    st.error(f"Cannot fetch model info: {e}")
    st.stop()

st.subheader(f"Best model: `{meta.get('best_model_name')}`")
st.write(f"Tuned threshold: **{meta.get('threshold'):.4f}**")

c1, c2, c3 = st.columns(3)
test_m = meta.get("test_metrics", {})
c1.metric("Test AUC-PR", f"{test_m.get('auc_pr', 0):.3f}")
c2.metric("Test AUC-ROC", f"{test_m.get('auc_roc', 0):.3f}")
c3.metric("Test F1", f"{test_m.get('f1', 0):.3f}")

st.divider()
st.subheader("All trained models")
results = meta.get("all_results", [])
if results:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r["name"],
                "threshold": round(r["threshold"], 4),
                **{f"val_{k}": round(v, 3) for k, v in r["val_metrics"].items() if k != "threshold"},
                **{f"test_{k}": round(v, 3) for k, v in r["test_metrics"].items() if k != "threshold"},
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.caption(
    f"Trained on {meta.get('train_rows', 'n/a'):,} rows · "
    f"validated on {meta.get('val_rows', 'n/a'):,} · "
    f"tested on {meta.get('test_rows', 'n/a'):,} · "
    f"{meta.get('feature_count', 'n/a')} features"
)
