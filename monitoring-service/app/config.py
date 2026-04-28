import os
from pathlib import Path

REFERENCE_PATH = Path(os.getenv("REFERENCE_PATH", "/models/reference_data.parquet"))
PREDICTION_LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", "/models/prediction_log.jsonl"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/reports"))

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-2")
CW_NAMESPACE = os.getenv("CW_NAMESPACE", "FraudDetection")

DRIFT_INTERVAL_MINUTES = int(os.getenv("DRIFT_INTERVAL_MINUTES", "60"))
DRIFT_SHARE_ALARM_THRESHOLD = float(os.getenv("DRIFT_ALARM", "0.30"))
