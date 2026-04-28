# Real-Time Fraud Detection Pipeline

End-to-end ML pipeline for detecting fraudulent financial transactions in real-time. Containerized microservices (training, serving, monitoring, UI, MLflow) deployed via Docker Compose on AWS EC2.

**Dataset:** IEEE-CIS Fraud Detection (590K transactions, 3.5% fraud rate)
**Models compared:** Logistic Regression, Random Forest, XGBoost, LightGBM
**Stack:** Python, FastAPI, Streamlit, Evidently AI, MLflow, SHAP, Docker, AWS

---

## Architecture

```
EC2 (t3.medium, Amazon Linux 2023)
└── Docker Compose
    ├── training   :8001  — Train + evaluate models, log to MLflow
    ├── serving    :8000  — FastAPI prediction endpoint + SHAP
    ├── monitoring :8002  — Evidently drift detection + CloudWatch
    ├── ui         :8501  — Streamlit dashboard
    └── mlflow     :5000  — Experiment tracking + model registry
```

## Quick start (local, after Docker Desktop installed)

```bash
# 1. Place Kaggle CSVs in data/raw/
#    train_transaction.csv, train_identity.csv

# 2. Copy env template
cp .env.example .env

# 3. Build + start everything
docker compose up -d --build

# 4. Trigger training (one-time, takes ~30-60 min)
curl -X POST http://localhost:8001/train

# 5. Open UIs
#    http://localhost:8501  — Streamlit dashboard
#    http://localhost:5000  — MLflow tracking
#    http://localhost:8000/docs  — FastAPI Swagger
```

## Convenience commands

See `Makefile`:
```bash
make build       # docker compose build
make up          # docker compose up -d
make down        # docker compose down
make logs        # tail all service logs
make train       # POST /train on training service
make test        # run pytest suite
make clean       # stop + remove volumes
```

## Project layout

```
fraud-detection-pipeline/
├── docker-compose.yml
├── .env.example
├── Makefile
├── PROJECT_LOG.md          ← changelog of every meaningful change
├── training-service/       ← data loading, feature eng, model training
├── serving-service/        ← FastAPI prediction API
├── monitoring-service/     ← Evidently drift detection
├── ui-service/             ← Streamlit dashboard
├── mlflow-service/         ← MLflow tracking server
├── data/raw/               ← Kaggle CSVs (gitignored)
├── data/processed/         ← engineered features (gitignored)
├── models/                 ← shared volume mount point
├── reports/                ← drift HTML reports
├── cloudwatch/             ← CloudWatch agent config
├── tests/                  ← pytest tests
├── scripts/setup_ec2.sh    ← EC2 bootstrap
└── .github/workflows/ci.yml
```

## Status

See `PROJECT_LOG.md` for the running changelog.

For step-by-step setup instructions (Kaggle, Docker, AWS), see `../USER_SETUP_INSTRUCTIONS.pdf` at project root.
