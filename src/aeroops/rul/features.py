"""Feature engineering for tabular models and sequence windows for the LSTM."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import USED_SENSORS

ROLL_WINDOWS = (5, 15)


def tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row features: raw sensors + rolling mean/std + cycle (no future leakage)."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    feats = df[["unit", "cycle", *USED_SENSORS]].copy()
    g = df.groupby("unit")[USED_SENSORS]
    for w in ROLL_WINDOWS:
        mean = g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        std = g.rolling(w, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
        feats[[f"{c}_mean{w}" for c in USED_SENSORS]] = mean[USED_SENSORS].to_numpy()
        feats[[f"{c}_std{w}" for c in USED_SENSORS]] = std[USED_SENSORS].to_numpy()
    if "rul" in df.columns:
        feats["rul"] = df["rul"].to_numpy()
    return feats


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in ("unit", "rul")]


def last_rows(feats: pd.DataFrame) -> pd.DataFrame:
    """Last observed cycle of each unit (used for test-set evaluation)."""
    return feats.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)


class MinMax:
    def fit(self, x: np.ndarray) -> "MinMax":
        self.lo = x.min(axis=0)
        self.span = np.where(x.max(axis=0) - self.lo == 0, 1.0, x.max(axis=0) - self.lo)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lo) / self.span


def sequence_windows(df: pd.DataFrame, scaler: MinMax, window: int = 30, last_only: bool = False):
    """Sliding windows of shape (n, window, n_sensors). Short units are left-padded
    with their first observation. Returns (X, y or None)."""
    xs, ys = [], []
    for _, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        arr = scaler.transform(g[USED_SENSORS].to_numpy(dtype=float))
        if len(arr) < window:
            arr = np.vstack([np.repeat(arr[:1], window - len(arr), axis=0), arr])
            rul = None if "rul" not in g else np.concatenate([np.repeat(g["rul"].to_numpy()[:1], window - len(g)), g["rul"].to_numpy()])
        else:
            rul = g["rul"].to_numpy() if "rul" in g else None
        ends = [len(arr)] if last_only else range(window, len(arr) + 1)
        for e in ends:
            xs.append(arr[e - window:e])
            if rul is not None:
                ys.append(rul[e - 1])
    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32) if ys else None
    return X, y
