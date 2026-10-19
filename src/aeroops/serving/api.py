"""FastAPI model-serving layer for both AeroOps models.

  POST /predict/delay  - probability a flight arrives >=15 min late (XGBoost, BTS features)
  POST /predict/rul    - remaining useful life of a turbofan engine (XGBoost, C-MAPSS)

History features for /predict/delay come from an "online feature store" snapshot
(models/online_features.json) produced by the same Spark job that builds the offline
training features, so training/serving definitions stay in sync.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier, XGBRegressor

from ..flights.schema import HISTORY_SPECS
from ..rul.data import USED_SENSORS
from ..rul.features import last_rows, tabular_features

MODEL_DIR = Path(os.getenv("AEROOPS_MODEL_DIR", "models"))
FD = os.getenv("AEROOPS_RUL_FD", "FD001")

app = FastAPI(title="AeroOps ML API", version="0.1.0")


# ---------- delay model ----------
class FlightRequest(BaseModel):
    carrier: str = Field(..., examples=["WN"])
    origin: str = Field(..., examples=["ATL"])
    dest: str = Field(..., examples=["LGA"])
    flight_date: dt.date = Field(..., examples=["2024-04-02"])
    dep_hour: int = Field(..., ge=0, le=23)
    distance: float = Field(..., gt=0)
    prev_leg_arr_delay: float | None = Field(None, description="Inbound aircraft's arrival delay (min), if known")
    prev_leg_dep_delay: float | None = None
    leg_of_day: int | None = Field(None, ge=1)


class DelayResponse(BaseModel):
    delay_probability: float
    predicted_delayed: bool
    threshold: float
    history_as_of: str
    unknown_categories: list[str]


@lru_cache(maxsize=1)
def delay_model():
    m = XGBClassifier()
    m.load_model(MODEL_DIR / "xgb_delay.json")
    meta = json.loads((MODEL_DIR / "xgb_delay_meta.json").read_text())
    snap = json.loads((MODEL_DIR / "online_features.json").read_text())
    return m, meta, snap


def delay_frame(req: FlightRequest, meta: dict, snap: dict) -> tuple[pd.DataFrame, list[str]]:
    row = {"carrier": req.carrier, "origin": req.origin, "dest": req.dest, "dep_hour": req.dep_hour,
           "day_of_week": (req.flight_date.isoweekday() % 7) + 1,  # Spark dayofweek: Sun=1..Sat=7
           "day_of_month": req.flight_date.day, "distance": req.distance,
           "prev_leg_arr_delay": req.prev_leg_arr_delay, "prev_leg_dep_delay": req.prev_leg_dep_delay,
           "leg_of_day": req.leg_of_day}
    for name, (keys, _) in HISTORY_SPECS.items():
        rate, n = snap["tables"][name].get("|".join(str(row[k]) for k in keys), [np.nan, 0])
        row[f"{name}_delay_rate"], row[f"{name}_flights"] = rate, n
    df = pd.DataFrame([row])[meta["features"]]
    unknown = []
    for c, levels in meta["categories"].items():
        if row[c] not in levels:
            unknown.append(c)  # unseen category -> treated as missing by XGBoost
            df[c] = None
        df[c] = pd.Categorical(df[c], categories=levels)
    for c in df.columns:
        if c not in meta["categories"]:
            df[c] = df[c].astype(float)
    return df, unknown


@app.post("/predict/delay", response_model=DelayResponse)
def predict_delay(req: FlightRequest):
    model, meta, snap = delay_model()
    x, unknown = delay_frame(req, meta, snap)
    p = float(model.predict_proba(x)[0, 1])
    return DelayResponse(delay_probability=round(p, 4), predicted_delayed=p >= meta["threshold"],
                         threshold=round(meta["threshold"], 3), history_as_of=snap["as_of"], unknown_categories=unknown)


# ---------- RUL model ----------
class Cycle(BaseModel):
    cycle: int = Field(..., ge=1)
    sensors: dict[str, float] = Field(..., description=f"Sensor readings keyed by name: {USED_SENSORS}")


class RULRequest(BaseModel):
    unit_id: int = 1
    cycles: list[Cycle] = Field(..., min_length=1)


class RULResponse(BaseModel):
    unit_id: int
    last_cycle: int
    predicted_rul_cycles: float


@lru_cache(maxsize=1)
def rul_model():
    m = XGBRegressor()
    m.load_model(MODEL_DIR / f"xgb_rul_{FD}.json")
    cols = json.loads((MODEL_DIR / f"xgb_rul_{FD}_features.json").read_text())
    return m, cols


@app.post("/predict/rul", response_model=RULResponse)
def predict_rul(req: RULRequest):
    rows = []
    for c in req.cycles:
        missing = [s for s in USED_SENSORS if s not in c.sensors]
        if missing:
            raise HTTPException(status_code=422, detail=f"cycle {c.cycle} missing sensors: {missing}")
        rows.append({"unit": req.unit_id, "cycle": c.cycle, **{s: c.sensors[s] for s in USED_SENSORS}})
    df = pd.DataFrame(rows).sort_values("cycle")
    model, cols = rul_model()
    x = last_rows(tabular_features(df))[cols]
    return RULResponse(unit_id=req.unit_id, last_cycle=int(df["cycle"].max()),
                       predicted_rul_cycles=round(max(0.0, float(model.predict(x.to_numpy())[0])), 1))


@app.get("/health")
def health():
    return {"status": "ok", "models": {"delay": (MODEL_DIR / "xgb_delay.json").exists(),
                                       "rul": (MODEL_DIR / f"xgb_rul_{FD}.json").exists()}}
