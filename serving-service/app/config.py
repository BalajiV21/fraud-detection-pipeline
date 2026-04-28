import os
from pathlib import Path

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/best_model.joblib"))
PIPELINE_PATH = Path(os.getenv("PIPELINE_PATH", "/models/feature_pipeline.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", "/models/metadata.json"))
PREDICTION_LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", "/models/prediction_log.jsonl"))

THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
