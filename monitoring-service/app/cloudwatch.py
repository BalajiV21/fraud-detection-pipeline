"""Publish custom metrics to CloudWatch. No-op if boto3 cannot find creds."""
import logging

from .config import AWS_REGION, CW_NAMESPACE

log = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            import boto3

            _client = boto3.client("cloudwatch", region_name=AWS_REGION)
        except Exception as e:
            log.warning("CloudWatch client unavailable: %s", e)
            _client = False
    return _client if _client else None


def publish_metric(name: str, value: float, unit: str = "None", dimensions=None):
    cw = _get_client()
    if cw is None:
        log.info("(noop) %s=%s %s", name, value, unit)
        return
    try:
        cw.put_metric_data(
            Namespace=CW_NAMESPACE,
            MetricData=[
                {
                    "MetricName": name,
                    "Value": float(value),
                    "Unit": unit,
                    "Dimensions": dimensions or [],
                }
            ],
        )
    except Exception as e:
        log.warning("Failed to publish %s: %s", name, e)
