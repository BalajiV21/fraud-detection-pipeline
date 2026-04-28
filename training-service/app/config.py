"""Centralized config for training service."""
import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

# Time-based split fractions
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # test = 0.15

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = "fraud-detection"

# Random seed
RANDOM_STATE = 42

# Models to train (toggle off any to skip)
MODELS_TO_TRAIN = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

# Optuna trials per model (reduce for fast iteration, increase for production)
N_OPTUNA_TRIALS = int(os.getenv("N_OPTUNA_TRIALS", "20"))

# Sample fraction for fast dev (set to 1.0 for full dataset)
SAMPLE_FRAC = float(os.getenv("SAMPLE_FRAC", "1.0"))
