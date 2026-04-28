"""FastAPI entrypoint for monitoring service."""
import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from .config import REPORTS_DIR
from .drift_detector import check_drift
from .scheduler import start as start_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("monitoring-service")

app = FastAPI(title="Fraud Detection — Monitoring Service", version="0.1.0")


@app.on_event("startup")
def _startup():
    start_scheduler()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/drift-check")
def drift_check_now():
    return check_drift()


@app.get("/drift-report")
def latest_report():
    path = REPORTS_DIR / "drift_latest.html"
    if not path.exists():
        return JSONResponse({"error": "No drift report yet — POST /drift-check first."}, status_code=404)
    return FileResponse(path, media_type="text/html")
