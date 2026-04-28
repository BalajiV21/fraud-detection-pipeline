# Project Log — Real-Time Fraud Detection Pipeline

This file tracks every meaningful change, milestone, decision, and issue across the lifecycle of the project. Append a new entry for each session/change. Format: `## YYYY-MM-DD — short title` followed by bullets.

---

## 2026-04-27 — Initial scaffold created by Claude

**Status:** Project scaffold generated. No model trained yet. No deployment yet.

### What was created
- Project root: `fraud-detection-pipeline/`
- Directory structure for all 5 services (training, serving, monitoring, ui, mlflow)
- Root config: `docker-compose.yml`, `.env.example`, `.gitignore`, `README.md`, `Makefile`
- `training-service/`: Dockerfile, requirements.txt, app/ with data_loader.py, feature_eng.py, train.py, evaluate.py, imbalance.py, main.py, config.py
- `serving-service/`: Dockerfile, requirements.txt, app/ with main.py (FastAPI), model_loader.py, feature_transform.py, explainer.py, prediction_logger.py, config.py
- `monitoring-service/`: Dockerfile, requirements.txt, app/ with drift_detector.py, cloudwatch.py, scheduler.py, main.py, config.py
- `ui-service/`: Dockerfile, requirements.txt, streamlit_app.py with multi-page dashboard
- `mlflow-service/`: Dockerfile, requirements.txt
- `tests/`: test_features.py, test_predict.py, test_drift.py
- `.github/workflows/ci.yml`: GitHub Actions CI pipeline
- `cloudwatch/cloudwatch-config.json`: CloudWatch agent config
- `scripts/setup_ec2.sh`: EC2 bootstrap script

### Decisions / notes
- Region: us-east-2 (Ohio) — matches existing compliance-auditor instance
- Instance: t3.medium with 50 GB gp3 EBS
- Threshold default: 0.5 (will be tuned post-training based on PR curve)
- Used `python:3.11-slim` base for all service images to keep size down
- MLflow uses sqlite backend store for simplicity (single-node)

### Blocked on user
Project cannot proceed past scaffolding without user-side actions:
1. Kaggle account + dataset download (IEEE-CIS, ~720 MB)
2. Docker Desktop install on Windows
3. AWS account setup + EC2 launch (when ready to deploy)
4. GitHub repo creation + push

See `USER_SETUP_INSTRUCTIONS.pdf` at project root for step-by-step.

---

## 2026-04-27 (evening) — End-to-end pipeline working locally

**Status:** All 5 containers up. Training succeeded on sample data. Serving + drift verified. Streamlit walkthrough still pending.

### Achievements
- Docker Desktop installed; `docker compose up -d --build` brings up all 5 services cleanly.
- IEEE-CIS CSVs placed in `data/raw/` (sampled to 180K rows for fast first run — `SAMPLE_FRAC < 1.0` in training config).
- First training run completed in ~85s. Leaderboard (test AUC-PR):
  - LightGBM **0.436** ⭐ (val 0.500) → saved as `best_model.joblib`
  - XGBoost 0.400 (val 0.470)
  - RandomForest 0.328 (val 0.408)
  - LogisticRegression 0.132 (val 0.101)
- All 4 runs logged in MLflow at http://localhost:5000.
- `/predict` returns sensible probabilities + SHAP top-5 risk factors.
- Drift detection working end-to-end: Evidently report at http://localhost:8002/drift-report renders correctly.
  - Synthetic load test produced expected 100% drift across 4 comparable columns (TransactionDT, TransactionAmt, addr1, addr2) — confirms alarm pipeline is wired correctly.
  - Stat test: Wasserstein distance (normed). Drift scores: TransactionDT 9.06, TransactionAmt 4.79, addr1 0.39, addr2 0.14.

### Issues + resolutions
- **MLflow SQLite "unable to open database file" loop.** Cause: `sqlite:///mlflow/mlflow.db` is a relative URI; with `WORKDIR /mlflow` it resolved to `/mlflow/mlflow/mlflow.db` (nested dir didn't exist). Fix: change to absolute URI `sqlite:////mlflow/mlflow.db` (4 slashes), wrap startup in `sh -c` to mkdir artifacts dir. Also removed obsolete `version: '3.8'` from compose file.
- **`ModuleNotFoundError: No module named 'app.feature_eng'` on serving startup.** Cause: joblib pickle stores full module path of `FeaturePipeline`; serving image had no copy of `feature_eng.py`. Fix: copied `training-service/app/feature_eng.py` to `serving-service/app/feature_eng.py` verbatim. Cleanup item: consolidate into shared `common/` package later.
- **`prediction_log.jsonl` never created** — silent write failure. Cause: `shared-data:/models:ro` (read-only mount) on serving service. The `prediction_logger.log_prediction()` exception was caught and only warned. Fix: removed `:ro` qualifier on serving's volume mount in `docker-compose.yml`. Cleanup item: separate writable log volume from the model volume so model files can stay RO.
- **`Empty column 'TransactionDT'` on first drift check.** Cause: synthetic load tests didn't include TransactionDT; pydantic dumped it as None for all rows; Evidently rejected the all-null column. Fix (immediate): re-sent synthetic predictions with realistic TransactionDT values. Cleanup item: drift_detector.py should `current = current.dropna(axis=1, how='all')` before the column-intersection step.

### Cleanup TODO (do before AWS deploy)
1. Consolidate `feature_eng.py` into a shared `common/` package — avoid two-file drift.
2. Bump `SAMPLE_FRAC` to `1.0` in `training-service/app/config.py` for full 590K-row training run.
3. Silence MLflow's `Failed to import Git` warning via `GIT_PYTHON_REFRESH=quiet` env var on training service.
4. Defensive empty-column drop in `monitoring-service/app/drift_detector.py`.
5. Separate writable `prediction-log` volume from RO `shared-data` model volume.

### Tomorrow — pick up here
- Streamlit walkthrough + screenshots: open http://localhost:8501, verify all 4 pages render (Predictions, Model Performance, Drift Monitoring, Explainability).
- Decide path: (a) full-data training run for better numbers, or (b) AWS deploy for resume bullet.
- Containers can be left running overnight (idle), or `docker compose stop` to free RAM/CPU.

---

## How to use this log going forward

When you (or Claude) make any meaningful change, add an entry like:

```
## 2026-05-04 — Trained baseline XGBoost
- Trained XGBoost on full 590K dataset, scale_pos_weight=20
- AUC-PR: 0.71, AUC-ROC: 0.93 on test set
- Saved to /models/best_model.joblib
- MLflow run id: abc123
- Issue: SMOTE caused OOM on t3.medium — disabled, used class weights only
```

Categories to log:
- **Setup/install events** (Docker installed, dataset downloaded, EC2 launched)
- **Training runs** (model type, hyperparameters, metrics, run id)
- **Deployment events** (deployed to EC2, model version, who triggered)
- **Issues + resolutions** (what broke, what fixed it)
- **Decisions** (chose X over Y because Z)
- **Drift alerts** (date, drift score, action taken)
