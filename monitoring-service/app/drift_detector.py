"""Evidently AI drift detection.

Compares the recent window of prediction inputs (read from prediction_log.jsonl)
against the training reference data saved by training-service.
"""
import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from .cloudwatch import publish_metric
from .config import DRIFT_SHARE_ALARM_THRESHOLD, PREDICTION_LOG_PATH, REFERENCE_PATH, REPORTS_DIR

log = logging.getLogger(__name__)


def _load_reference() -> Optional[pd.DataFrame]:
    if not REFERENCE_PATH.exists():
        log.warning("No reference data at %s — train a model first.", REFERENCE_PATH)
        return None
    return pd.read_parquet(REFERENCE_PATH)


def _load_recent_predictions(n: int = 1000) -> Optional[pd.DataFrame]:
    if not PREDICTION_LOG_PATH.exists():
        log.warning("No predictions logged yet at %s", PREDICTION_LOG_PATH)
        return None
    rows = []
    with open(PREDICTION_LOG_PATH) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return None
    rows = rows[-n:]
    inputs = pd.DataFrame([r["input"] for r in rows])
    return inputs


def check_drift() -> dict:
    """Run Evidently DataDriftPreset. Save HTML report. Publish CloudWatch metric."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reference = _load_reference()
    current = _load_recent_predictions()

    if reference is None or current is None:
        return {"status": "skipped", "reason": "missing reference or current data"}

    # Align columns: only keep columns present in both
    common = [c for c in reference.columns if c in current.columns]
    if not common:
        # If serving sends raw fields, they won't match the engineered reference.
        # In that case, just compare the numeric subset of `current`.
        log.warning("No common columns between reference and current — drift report will be limited.")
        return {"status": "skipped", "reason": "no overlapping columns"}

    ref = reference[common].copy()
    cur = current[common].copy()

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)
        result = report.as_dict()

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        html_path = REPORTS_DIR / f"drift_{ts}.html"
        report.save_html(str(html_path))
        # Always overwrite "latest"
        latest_path = REPORTS_DIR / "drift_latest.html"
        report.save_html(str(latest_path))

        m = result["metrics"][0]["result"]
        drift_share = float(m.get("share_of_drifted_columns", m.get("drift_share", 0.0)))
        dataset_drift = bool(m.get("dataset_drift", False))
        n_drifted = int(m.get("number_of_drifted_columns", 0))

        publish_metric("DriftScore", drift_share, "None")
        publish_metric("FeatureDriftCount", n_drifted, "Count")
        if drift_share >= DRIFT_SHARE_ALARM_THRESHOLD:
            publish_metric("DriftAlarm", 1.0, "None")

        return {
            "status": "ok",
            "drift_share": drift_share,
            "n_drifted_columns": n_drifted,
            "dataset_drift": dataset_drift,
            "report_path": str(html_path),
            "latest_report_path": str(latest_path),
        }
    except Exception as e:
        log.exception("Drift check failed: %s", e)
        return {"status": "error", "error": str(e)}
