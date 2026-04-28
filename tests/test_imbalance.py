"""Tests for imbalance helpers."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training-service"))

from app.imbalance import compute_scale_pos_weight, find_threshold_for_recall  # noqa: E402


def test_scale_pos_weight_for_imbalanced():
    y = np.array([0] * 95 + [1] * 5)
    spw = compute_scale_pos_weight(y)
    assert abs(spw - 19.0) < 1e-6


def test_threshold_for_recall_returns_valid():
    rng = np.random.default_rng(0)
    y = rng.choice([0, 1], size=1000, p=[0.95, 0.05])
    proba = rng.uniform(0, 1, size=1000)
    res = find_threshold_for_recall(y, proba, target_recall=0.5)
    assert 0.0 <= res["threshold"] <= 1.0
    assert 0.0 <= res["precision"] <= 1.0
    assert 0.0 <= res["recall"] <= 1.0
