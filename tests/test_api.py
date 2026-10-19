import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier, XGBRegressor

from aeroops.flights.schema import HISTORY_SPECS
from aeroops.rul.data import USED_SENSORS
from aeroops.rul.features import feature_columns, tabular_features


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from aeroops.serving import api

    rng = np.random.default_rng(0)
    # tiny RUL model
    rul_df = pd.DataFrame([{"unit": 1, "cycle": c, **{s: rng.normal() for s in USED_SENSORS}} for c in range(1, 60)])
    ft = tabular_features(rul_df)
    cols = feature_columns(ft)
    XGBRegressor(n_estimators=5).fit(ft[cols], rng.uniform(0, 125, len(ft))).save_model(tmp_path / "xgb_rul_FD001.json")
    (tmp_path / "xgb_rul_FD001_features.json").write_text(json.dumps(cols))
    # tiny delay model
    meta_feats = ["carrier", "origin", "dest", "dep_hour", "day_of_week", "day_of_month", "distance",
                  *[f"{n}_delay_rate" for n in HISTORY_SPECS], *[f"{n}_flights" for n in HISTORY_SPECS],
                  "prev_leg_arr_delay", "prev_leg_dep_delay", "leg_of_day"]
    cats = {"carrier": ["AA", "DL"], "origin": ["JFK", "LAX"], "dest": ["JFK", "LAX"]}
    n = 200
    X = pd.DataFrame({c: rng.normal(size=n) for c in meta_feats})
    for c, lv in cats.items():
        X[c] = pd.Categorical(rng.choice(lv, n), categories=lv)
    XGBClassifier(n_estimators=5, enable_categorical=True, tree_method="hist").fit(X[meta_feats], rng.integers(0, 2, n)) \
        .save_model(tmp_path / "xgb_delay.json")
    (tmp_path / "xgb_delay_meta.json").write_text(json.dumps({"features": meta_feats, "categories": cats, "threshold": 0.3}))
    snap = {"as_of": "2024-03-31", "tables": {k: {} for k in HISTORY_SPECS}}
    snap["tables"]["origin_7d"]["JFK"] = [0.2, 1000]
    (tmp_path / "online_features.json").write_text(json.dumps(snap))

    monkeypatch.setattr(api, "MODEL_DIR", tmp_path)
    api.delay_model.cache_clear()
    api.rul_model.cache_clear()
    yield TestClient(api.app)
    api.delay_model.cache_clear()
    api.rul_model.cache_clear()


def test_health(client):
    assert client.get("/health").json()["models"] == {"delay": True, "rul": True}


def test_predict_delay(client):
    r = client.post("/predict/delay", json={"carrier": "AA", "origin": "JFK", "dest": "LAX", "flight_date": "2024-04-02",
                                            "dep_hour": 17, "distance": 2475, "prev_leg_arr_delay": 30})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["delay_probability"] <= 1 and body["history_as_of"] == "2024-03-31"
    assert body["unknown_categories"] == []


def test_predict_delay_flags_unknown_airport(client):
    r = client.post("/predict/delay", json={"carrier": "AA", "origin": "XYZ", "dest": "LAX", "flight_date": "2024-04-02",
                                            "dep_hour": 9, "distance": 500})
    assert r.status_code == 200 and r.json()["unknown_categories"] == ["origin"]


def test_predict_rul(client):
    cycles = [{"cycle": i, "sensors": {s: 0.1 * i for s in USED_SENSORS}} for i in range(1, 25)]
    r = client.post("/predict/rul", json={"unit_id": 7, "cycles": cycles})
    assert r.status_code == 200, r.text
    assert r.json()["last_cycle"] == 24 and r.json()["predicted_rul_cycles"] >= 0


def test_predict_rul_rejects_missing_sensor(client):
    r = client.post("/predict/rul", json={"cycles": [{"cycle": 1, "sensors": {"s_2": 1.0}}]})
    assert r.status_code == 422
