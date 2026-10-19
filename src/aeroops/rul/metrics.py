"""Evaluation metrics for RUL prediction."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric score: late predictions (d > 0) are penalised more than early ones."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))
