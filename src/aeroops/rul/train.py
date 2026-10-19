"""Engine remaining-useful-life (RUL) models on NASA C-MAPSS with unit-based validation.

Validation design:
  - 5-fold GroupKFold over ENGINE UNITS on the training file (an engine's cycles never appear
    in both train and validation; random row splits leak because adjacent cycles are near-identical)
  - Official held-out test set (test_FD00x + RUL_FD00x), evaluated at each engine's last cycle
Models: mean-RUL baseline, Ridge, RandomForest, XGBoost (+ feature ablation), PyTorch LSTM.

Usage: python -m aeroops.rul.train --data-dir data/raw/cmapss --fd FD001
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from .data import RUL_CAP, USED_SENSORS, load_fd, validate
from .features import MinMax, feature_columns, last_rows, sequence_windows, tabular_features
from .lstm import predict_lstm, train_lstm
from .metrics import nasa_score, rmse

SEED = 42
RF_PARAMS = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5}
XGB_PARAMS = {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}


def make_tabular(name: str, threads: int):
    if name == "Ridge regression":
        return Ridge(alpha=1.0)
    if name == "Random Forest":
        return RandomForestRegressor(**RF_PARAMS, n_jobs=threads, random_state=SEED)
    return XGBRegressor(**XGB_PARAMS, n_jobs=threads, random_state=SEED)


def cv_last_cycle_eval(ftr: pd.DataFrame, cols, name, threads, k=5):
    """GroupKFold by unit. Validation engines are truncated at a random cycle (like the test set)
    and scored at that cycle, so CV mimics the official test protocol."""
    rng = np.random.default_rng(SEED)
    units = ftr["unit"].to_numpy()
    fold_rmse = []
    for tr_idx, va_idx in GroupKFold(n_splits=k).split(ftr, groups=units):
        tr, va = ftr.iloc[tr_idx], ftr.iloc[va_idx]
        cut = {u: rng.integers(31, g["cycle"].max() + 1) for u, g in va.groupby("unit")}
        va_last = va[va["cycle"] == va["unit"].map(cut)]
        if name == "Baseline (mean RUL)":
            pred = np.full(len(va_last), tr["rul"].mean())
        else:
            m = make_tabular(name, threads).fit(tr[cols].to_numpy(), tr["rul"].to_numpy())
            pred = m.predict(va_last[cols].to_numpy())
        fold_rmse.append(rmse(va_last["rul"], np.clip(pred, 0, None)))
    return float(np.mean(fold_rmse)), float(np.std(fold_rmse))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/cmapss")
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--lstm-epochs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    out, md = Path(a.out), Path(a.models_dir)
    out.mkdir(parents=True, exist_ok=True)
    md.mkdir(parents=True, exist_ok=True)

    train, test, rul_true = load_fd(a.data_dir, a.fd)
    validate(train)
    validate(test)
    y_test = np.minimum(rul_true, RUL_CAP)  # truth capped like the training target; uncapped also reported

    ftr = tabular_features(train)
    fte = last_rows(tabular_features(test))
    all_cols = feature_columns(ftr)
    raw_cols = ["cycle", *USED_SENSORS]

    mlflow.set_experiment(f"rul-{a.fd}")
    results = []

    def record(name, cols, pred, params, seconds, cv=(None, None)):
        pred = np.clip(pred, 0, None)
        r = {"model": name, "features": "raw" if cols == raw_cols else ("rolling" if cols else "-"),
             "cv_rmse_mean": None if cv[0] is None else round(cv[0], 2),
             "cv_rmse_std": None if cv[1] is None else round(cv[1], 2),
             "test_rmse": round(rmse(y_test, pred), 2), "test_rmse_uncapped": round(rmse(rul_true, pred), 2),
             "test_nasa_score": round(nasa_score(y_test, pred), 1), "fit_seconds": round(seconds, 1)}
        with mlflow.start_run(run_name=f"{name} [{r['features']}]"):
            mlflow.log_params({"fd": a.fd, "rul_cap": RUL_CAP, "n_features": len(cols), **params})
            mlflow.log_metrics({k: v for k, v in r.items() if isinstance(v, (int, float)) and v is not None})
        results.append(r)
        print(r, flush=True)
        return pred

    # Baseline
    cv = cv_last_cycle_eval(ftr, all_cols, "Baseline (mean RUL)", a.threads)
    record("Baseline (mean RUL)", [], np.full(len(fte), ftr["rul"].mean()), {}, 0.0, cv)

    for name, params in [("Ridge regression", {"alpha": 1.0}), ("Random Forest", RF_PARAMS), ("XGBoost", XGB_PARAMS)]:
        for cols in ([raw_cols, all_cols] if name == "XGBoost" else [all_cols]):  # ablation: raw vs rolling features
            cv = cv_last_cycle_eval(ftr, cols, name, a.threads)
            t = time.time()
            m = make_tabular(name, a.threads).fit(ftr[cols].to_numpy(), ftr["rul"].to_numpy())
            sec = time.time() - t
            record(name, cols, m.predict(fte[cols].to_numpy()), params, sec, cv)
            if name == "XGBoost" and cols == all_cols:
                m.save_model(md / f"xgb_rul_{a.fd}.json")
                (md / f"xgb_rul_{a.fd}_features.json").write_text(json.dumps(cols))

    # LSTM (no CV: test-only, fixed hyper-parameters chosen a priori; documented in README)
    scaler = MinMax().fit(train[USED_SENSORS].to_numpy(dtype=float))
    Xs, ys = sequence_windows(train, scaler, window=30)
    Xs_te, _ = sequence_windows(test, scaler, window=30, last_only=True)
    lp = {"window": 30, "hidden": 64, "layers": 2, "epochs": a.lstm_epochs, "lr": 1e-3, "batch_size": 256}
    t = time.time()
    model = train_lstm(Xs, ys, epochs=a.lstm_epochs, seed=SEED,
                       log=lambda ep, loss: print(f"  lstm epoch {ep + 1}: mse {loss:.1f}", flush=True))
    record("LSTM (PyTorch)", USED_SENSORS, predict_lstm(model, Xs_te), lp, time.time() - t)
    results[-1]["features"] = "30-cycle windows"
    torch.save(model.state_dict(), md / f"lstm_rul_{a.fd}.pt")

    meta = {"dataset": f"NASA C-MAPSS {a.fd}", "train_engines": int(train.unit.nunique()),
            "test_engines": int(test.unit.nunique()), "train_rows": len(train), "rul_cap": RUL_CAP,
            "cpu_threads": a.threads, "results": results}
    (out / f"rul_results_{a.fd}.json").write_text(json.dumps(meta, indent=2))
    lines = ["| Model | Features | CV RMSE (5-fold by engine) | Test RMSE | Test RMSE (uncapped truth) | NASA score | Fit (s) |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        cvs = "-" if r["cv_rmse_mean"] is None else f"{r['cv_rmse_mean']} ± {r['cv_rmse_std']}"
        lines.append(f"| {r['model']} | {r['features']} | {cvs} | {r['test_rmse']} | {r['test_rmse_uncapped']} | {r['test_nasa_score']} | {r['fit_seconds']} |")
    (out / f"rul_results_{a.fd}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
