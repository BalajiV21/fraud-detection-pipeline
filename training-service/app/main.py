"""FastAPI entrypoint for training service."""
import logging
import threading
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("training-service")

app = FastAPI(title="Fraud Detection — Training Service", version="0.1.0")

_state = {"running": False, "last_result": None, "last_error": None}


@app.get("/health")
def health():
    return {"status": "ok", "running": _state["running"]}


@app.get("/status")
def status():
    return _state


def _train_in_background():
    from .train import run_training_pipeline

    _state["running"] = True
    _state["last_error"] = None
    try:
        result = run_training_pipeline()
        _state["last_result"] = result
        log.info("Training complete: best=%s", result.get("best_model_name"))
    except Exception as e:
        log.error("Training failed: %s\n%s", e, traceback.format_exc())
        _state["last_error"] = str(e)
    finally:
        _state["running"] = False


@app.post("/train")
def train(background: Optional[bool] = True):
    """Kick off training. Returns immediately if background=True."""
    if _state["running"]:
        raise HTTPException(409, "Training already running")
    if background:
        threading.Thread(target=_train_in_background, daemon=True).start()
        return {"started": True, "message": "Training started in background. Poll /status."}
    _train_in_background()
    return {"started": True, "result": _state["last_result"], "error": _state["last_error"]}
