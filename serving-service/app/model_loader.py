"""Lazy + reload-able model loader.

Serving service polls model file mtime and reloads on change so retraining
without restarting the container picks up automatically.
"""
import json
import logging
import os
from typing import Any, Optional

import joblib

from .config import METADATA_PATH, MODEL_PATH, PIPELINE_PATH

log = logging.getLogger(__name__)


class ModelBundle:
    def __init__(self):
        self.model: Optional[Any] = None
        self.pipeline: Optional[Any] = None
        self.metadata: dict = {}
        self._mtime: float = 0.0

    def _file_mtime(self) -> float:
        try:
            return os.path.getmtime(MODEL_PATH)
        except OSError:
            return 0.0

    def is_loaded(self) -> bool:
        return self.model is not None and self.pipeline is not None

    def load_if_needed(self) -> bool:
        """Load or reload if file is newer. Returns True if a (re)load happened."""
        if not MODEL_PATH.exists() or not PIPELINE_PATH.exists():
            return False
        cur = self._file_mtime()
        if self.is_loaded() and cur <= self._mtime:
            return False
        log.info("Loading model from %s", MODEL_PATH)
        self.model = joblib.load(MODEL_PATH)
        self.pipeline = joblib.load(PIPELINE_PATH)
        if METADATA_PATH.exists():
            try:
                self.metadata = json.loads(METADATA_PATH.read_text())
            except Exception:
                self.metadata = {}
        self._mtime = cur
        log.info("Model loaded. metadata keys: %s", list(self.metadata.keys()))
        return True


bundle = ModelBundle()
