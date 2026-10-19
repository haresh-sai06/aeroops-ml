"""Loading and labelling the C-MAPSS turbofan data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SETTINGS = [f"setting_{i}" for i in range(1, 4)]
SENSORS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]
# Sensors that are (near-)constant on FD001 and carry no degradation signal.
DROP_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
USED_SENSORS = [s for s in SENSORS if s not in DROP_SENSORS]
RUL_CAP = 125  # piece-wise linear RUL target, standard in the C-MAPSS literature


def read_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")
    return df


def add_train_rul(df: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """RUL = (last cycle of that unit) - current cycle, optionally capped."""
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    rul = max_cycle - out["cycle"]
    out["rul"] = rul.clip(upper=cap) if cap is not None else rul
    return out


def load_fd(data_dir: str | Path, fd: str = "FD001"):
    """Return (train_df with rul, test_df, true RUL per test unit as np.ndarray)."""
    data_dir = Path(data_dir)
    train = add_train_rul(read_raw(data_dir / f"train_{fd}.txt"))
    test = read_raw(data_dir / f"test_{fd}.txt")
    rul_true = pd.read_csv(data_dir / f"RUL_{fd}.txt", header=None).iloc[:, 0].to_numpy(dtype=float)
    return train, test, rul_true


def validate(df: pd.DataFrame) -> None:
    """Basic data-quality checks; raises ValueError on failure."""
    missing = [c for c in ["unit", "cycle", *USED_SENSORS] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df[["unit", "cycle", *USED_SENSORS]].isna().any().any():
        raise ValueError("null values in sensor data")
    for _, g in df.groupby("unit"):
        if not np.all(np.diff(g["cycle"].to_numpy()) == 1):
            raise ValueError("cycles must be contiguous and increasing within each unit")
