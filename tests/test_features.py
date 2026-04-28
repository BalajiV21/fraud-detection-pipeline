"""Smoke tests for the feature engineering pipeline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training-service"))

from app.feature_eng import FeaturePipeline  # noqa: E402


def _toy_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.linspace(86400, 86400 * 30, n),
            "TransactionAmt": rng.uniform(1, 1000, n),
            "ProductCD": rng.choice(["W", "H", "C"], n),
            "card4": rng.choice(["visa", "mastercard"], n),
            "card6": rng.choice(["debit", "credit"], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
            "R_emaildomain": rng.choice(["gmail.com", "outlook.com", None], n),
            "DeviceType": rng.choice(["desktop", "mobile", None], n),
            "DeviceInfo": rng.choice(["Windows", "iOS", None], n),
            "isFraud": rng.choice([0, 1], n, p=[0.95, 0.05]),
        }
    )


def test_fit_transform_shapes():
    df = _toy_df()
    y = df["isFraud"].astype(int)
    pipe = FeaturePipeline()
    X = pipe.fit_transform(df.drop(columns=["isFraud"]), y)
    assert X.shape[0] == len(df)
    assert pipe.fitted
    # All numeric
    assert all(np.issubdtype(t, np.number) for t in X.dtypes)


def test_transform_aligns_columns():
    df = _toy_df()
    y = df["isFraud"].astype(int)
    pipe = FeaturePipeline()
    pipe.fit_transform(df.drop(columns=["isFraud"]), y)

    # Missing columns at inference time should be filled
    new = df.drop(columns=["isFraud", "DeviceInfo"]).head(5)
    X = pipe.transform(new)
    assert list(X.columns) == pipe.final_columns


def test_engineered_features_exist():
    df = _toy_df()
    y = df["isFraud"].astype(int)
    pipe = FeaturePipeline()
    X = pipe.fit_transform(df.drop(columns=["isFraud"]), y)
    for col in ["hour_of_day", "day_of_week", "amount_log", "email_match", "null_count"]:
        assert col in X.columns
