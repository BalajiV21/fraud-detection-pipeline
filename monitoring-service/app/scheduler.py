"""Background scheduler that runs drift checks periodically."""
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from .config import DRIFT_INTERVAL_MINUTES
from .drift_detector import check_drift

log = logging.getLogger(__name__)
_scheduler = None


def start():
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        lambda: log.info("Scheduled drift result: %s", check_drift()),
        "interval",
        minutes=DRIFT_INTERVAL_MINUTES,
        id="drift_check",
    )
    _scheduler.start()
    log.info("Scheduler started — drift check every %d minutes", DRIFT_INTERVAL_MINUTES)


def stop():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
