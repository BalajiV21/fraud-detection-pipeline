"""Append every prediction to a JSONL file the monitoring-service reads."""
import json
import logging
import threading
from datetime import datetime, timezone

from .config import PREDICTION_LOG_PATH

log = logging.getLogger(__name__)
_lock = threading.Lock()


def log_prediction(payload: dict, proba: float, threshold: float, latency_ms: float):
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "input": payload,
        "fraud_probability": round(float(proba), 6),
        "is_fraud": bool(proba >= threshold),
        "threshold": threshold,
        "latency_ms": round(float(latency_ms), 2),
    }
    try:
        with _lock:
            PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(PREDICTION_LOG_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
    except Exception as e:
        log.warning("Failed to log prediction: %s", e)
