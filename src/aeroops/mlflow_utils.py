"""MLflow tracking defaults for AeroOps.

MLflow 3.x rejects the legacy ``./mlruns`` file store. We pin ``mlflow>=2.12,<3``
in requirements.txt and default the tracking URI to a SQLite store so a fresh
``pip install -r requirements.txt`` + train works without extra env setup.
"""
from __future__ import annotations

import os

import mlflow

DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"


def configure_mlflow(tracking_uri: str | None = None) -> str:
    """Set tracking URI from arg / ``MLFLOW_TRACKING_URI`` / SQLite default.

    Returns the URI actually configured.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    return uri
