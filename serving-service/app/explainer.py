"""SHAP explanations for individual predictions."""
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_explainer_cache = {"explainer": None, "model_id": None}


def _get_explainer(model):
    mid = id(model)
    if _explainer_cache["model_id"] != mid:
        try:
            import shap

            _explainer_cache["explainer"] = shap.TreeExplainer(model)
            _explainer_cache["model_id"] = mid
        except Exception as e:
            log.warning("Could not build SHAP TreeExplainer: %s", e)
            _explainer_cache["explainer"] = None
    return _explainer_cache["explainer"]


def top_risk_factors(model, features_df: pd.DataFrame, top_k: int = 5) -> List[Tuple[str, float]]:
    """Return [(feature_name, shap_value), ...] sorted by abs contribution."""
    explainer = _get_explainer(model)
    if explainer is None:
        return []
    try:
        shap_values = explainer.shap_values(features_df)
        if isinstance(shap_values, list):
            # Some models return a list per class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        row = np.asarray(shap_values)[0]
        idx = np.argsort(-np.abs(row))[:top_k]
        return [(str(features_df.columns[i]), float(row[i])) for i in idx]
    except Exception as e:
        log.warning("SHAP failure: %s", e)
        return []
