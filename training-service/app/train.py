"""Train, compare, select best model. Logs to MLflow + saves artifacts."""
import json
import logging

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODELS_TO_TRAIN,
    RANDOM_STATE,
    TARGET_COL,
    TIME_COL,
    TRAIN_FRAC,
    VAL_FRAC,
)
from .data_loader import load_raw
from .evaluate import compute_metrics, plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from .feature_eng import FeaturePipeline
from .imbalance import compute_scale_pos_weight, find_threshold_for_recall

log = logging.getLogger(__name__)


def time_split(df: pd.DataFrame):
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df)
    i_train = int(n * TRAIN_FRAC)
    i_val = int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def _build_model(name: str, spw: float):
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
            n_jobs=-1,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=400,
            max_depth=-1,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model {name}")


def run_training_pipeline() -> dict:
    """Full pipeline: load -> feature eng -> train all models -> pick best -> save."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Starting training pipeline")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_raw()
    log.info("Loaded %s rows", len(df))

    train_df, val_df, test_df = time_split(df)
    log.info("Splits: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    y_train = train_df[TARGET_COL].astype(int)
    y_val = val_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    pipeline = FeaturePipeline()
    log.info("Fitting feature pipeline")
    X_train = pipeline.fit_transform(train_df.drop(columns=[TARGET_COL]), y_train)
    X_val = pipeline.transform(val_df.drop(columns=[TARGET_COL]))
    X_test = pipeline.transform(test_df.drop(columns=[TARGET_COL]))
    log.info("Feature matrix shape: train=%s val=%s test=%s", X_train.shape, X_val.shape, X_test.shape)

    # Save the pipeline now so serving can use the same transforms
    pipe_path = MODEL_DIR / "feature_pipeline.joblib"
    joblib.dump(pipeline, pipe_path)
    log.info("Saved feature pipeline to %s", pipe_path)

    spw = compute_scale_pos_weight(y_train)
    log.info("scale_pos_weight = %.2f", spw)

    results = []
    for name in MODELS_TO_TRAIN:
        log.info("Training %s", name)
        with mlflow.start_run(run_name=name):
            model = _build_model(name, spw)
            model.fit(X_train, y_train)

            val_proba = model.predict_proba(X_val)[:, 1]
            val_metrics = compute_metrics(y_val, val_proba, threshold=0.5)

            test_proba = model.predict_proba(X_test)[:, 1]
            thr_info = find_threshold_for_recall(y_val, val_proba, target_recall=0.80)
            test_metrics = compute_metrics(y_test, test_proba, threshold=thr_info["threshold"])

            mlflow.log_param("model_type", name)
            mlflow.log_param("scale_pos_weight", spw)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_param("tuned_threshold", thr_info["threshold"])

            # Plots
            plots_dir = MODEL_DIR / f"plots_{name}"
            plots_dir.mkdir(exist_ok=True)
            plot_pr_curve(y_test, test_proba, plots_dir / "pr_curve.png")
            plot_roc_curve(y_test, test_proba, plots_dir / "roc_curve.png")
            plot_confusion_matrix(y_test, test_proba, thr_info["threshold"], plots_dir / "cm.png")
            for p in plots_dir.glob("*.png"):
                mlflow.log_artifact(str(p))

            results.append(
                {
                    "name": name,
                    "model": model,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "threshold": thr_info["threshold"],
                }
            )
            log.info("%s done — val AUC-PR=%.3f test AUC-PR=%.3f", name, val_metrics["auc_pr"], test_metrics["auc_pr"])

    # Pick best by val AUC-PR
    best = max(results, key=lambda r: r["val_metrics"]["auc_pr"])
    log.info("Best model: %s (val AUC-PR=%.3f)", best["name"], best["val_metrics"]["auc_pr"])

    # Save best model
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best["model"], model_path)
    log.info("Saved best model to %s", model_path)

    # Save metadata
    meta = {
        "best_model_name": best["name"],
        "threshold": best["threshold"],
        "val_metrics": best["val_metrics"],
        "test_metrics": best["test_metrics"],
        "feature_count": int(X_train.shape[1]),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "all_results": [
            {
                "name": r["name"],
                "val_metrics": r["val_metrics"],
                "test_metrics": r["test_metrics"],
                "threshold": r["threshold"],
            }
            for r in results
        ],
    }
    meta_path = MODEL_DIR / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Saved metadata to %s", meta_path)

    # Save reference dataset for drift detection
    ref_path = MODEL_DIR / "reference_data.parquet"
    X_train.sample(min(5000, len(X_train)), random_state=RANDOM_STATE).to_parquet(ref_path)
    log.info("Saved reference data to %s", ref_path)

    return meta
