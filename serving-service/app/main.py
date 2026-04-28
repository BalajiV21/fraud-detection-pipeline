"""FastAPI prediction service."""
import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from .config import THRESHOLD
from .explainer import top_risk_factors
from .model_loader import bundle
from .prediction_logger import log_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("serving-service")

app = FastAPI(title="Fraud Detection — Serving API", version="0.1.0")

# Prometheus
PRED_COUNTER = Counter("fraud_predictions_total", "Total predictions", ["fraud"])
LATENCY_HIST = Histogram("fraud_prediction_latency_ms", "Prediction latency (ms)", buckets=(5, 10, 25, 50, 100, 250, 500, 1000))


class Transaction(BaseModel):
    """Loose schema — extra fields are accepted to match the 400+ raw columns."""

    TransactionID: Optional[int] = None
    TransactionDT: Optional[float] = None
    TransactionAmt: float
    ProductCD: Optional[str] = None
    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None

    # accept arbitrary extras (V1-V339, C1-C14, D1-D15, M1-M9, id_xx)
    model_config = {"extra": "allow"}


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    prediction_time_ms: float
    top_risk_factors: List[Dict[str, Any]]
    model_name: Optional[str] = None


@app.on_event("startup")
def _startup():
    bundle.load_if_needed()


@app.get("/health")
def health():
    bundle.load_if_needed()
    return {
        "status": "ok",
        "model_loaded": bundle.is_loaded(),
        "model_name": bundle.metadata.get("best_model_name"),
    }


@app.get("/model/info")
def model_info():
    bundle.load_if_needed()
    if not bundle.is_loaded():
        raise HTTPException(503, "Model not loaded yet — run training first.")
    return bundle.metadata


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _predict_one(payload: dict) -> PredictionResponse:
    bundle.load_if_needed()
    if not bundle.is_loaded():
        raise HTTPException(503, "Model not loaded. Run training first via /train on training service.")

    start = time.perf_counter()
    df = pd.DataFrame([payload])
    features = bundle.pipeline.transform(df)
    proba = float(bundle.model.predict_proba(features)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000.0

    is_fraud = proba >= THRESHOLD
    PRED_COUNTER.labels(fraud=str(is_fraud).lower()).inc()
    LATENCY_HIST.observe(latency_ms)

    factors = top_risk_factors(bundle.model, features, top_k=5)
    log_prediction(payload, proba, THRESHOLD, latency_ms)

    return PredictionResponse(
        fraud_probability=round(proba, 6),
        is_fraud=bool(is_fraud),
        threshold=THRESHOLD,
        prediction_time_ms=round(latency_ms, 2),
        top_risk_factors=[{"feature": n, "shap_value": round(v, 6)} for n, v in factors],
        model_name=bundle.metadata.get("best_model_name"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction):
    return _predict_one(txn.model_dump())


@app.post("/predict/batch")
def predict_batch(txns: List[Transaction]):
    return [_predict_one(t.model_dump()) for t in txns]
