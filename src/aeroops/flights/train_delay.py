"""Train / validate the arrival-delay classifier with time-based splits and feature-group ablations.

Validation design (no random shuffling - flights are a time series):
  CV fold 1: train Jan 1-31          -> validate Feb 1-14
  CV fold 2: train Jan 1 - Feb 14    -> validate Feb 15-29
  Final:     train Jan 1 - Feb 29    -> TEST on March (touched once, after model selection)

Usage: python -m aeroops.flights.train_delay --lake-dir data/lake
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .schema import CATEGORICAL, HISTORY, OPERATIONAL, SCHEDULE, TARGET

SEED = 42
XGB_PARAMS = dict(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, eval_metric="logloss")
ABLATIONS = {
    "XGB: schedule only": SCHEDULE,
    "XGB: schedule + history": SCHEDULE + HISTORY,
    "XGB: schedule + history + operational": SCHEDULE + HISTORY + OPERATIONAL,
}
FOLDS = [("2024-01-01", "2024-02-01", "2024-02-15"), ("2024-01-01", "2024-02-15", "2024-03-01")]
TEST = ("2024-01-01", "2024-03-01", "2024-04-01")


def load(lake_dir: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(Path(lake_dir) / "features" / "flight_delay")
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    cats = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        df[c] = pd.Categorical(df[c], categories=cats[c])
    return df.sort_values("flight_date").reset_index(drop=True), cats


def split(df, start, cut, end):
    tr = df[(df.flight_date >= start) & (df.flight_date < cut)]
    te = df[(df.flight_date >= cut) & (df.flight_date < end)]
    return tr, te


def scores(y, p, thr=None) -> dict:
    out = {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}
    if thr is not None:
        yhat = (p >= thr).astype(int)
        out.update(precision=precision_score(y, yhat), recall=recall_score(y, yhat), f1=f1_score(y, yhat))
    return {k: round(float(v), 4) for k, v in out.items()}


def best_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    return round(float(grid[int(np.argmax([f1_score(y, (p >= t).astype(int)) for t in grid]))]), 2)


def fit_predict(name, cols, tr, te, threads):
    if name.startswith("Prior"):
        return np.full(len(te), tr[TARGET].mean()), None
    if name.startswith("LogReg"):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500))
        m.fit(tr[cols], tr[TARGET])
        return m.predict_proba(te[cols])[:, 1], m
    m = XGBClassifier(**XGB_PARAMS, n_jobs=threads, random_state=SEED)
    m.fit(tr[cols], tr[TARGET])
    return m.predict_proba(te[cols])[:, 1], m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake-dir", default="data/lake")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    out, md = Path(a.out), Path(a.models_dir)
    out.mkdir(exist_ok=True)
    md.mkdir(exist_ok=True)
    df, cats = load(a.lake_dir)

    numeric_lr = ["dep_hour", "day_of_week", "distance", *HISTORY]
    models = {"Prior (base rate)": [], "LogReg: schedule + history (numeric)": numeric_lr, **ABLATIONS}
    mlflow.set_experiment("flight-delay")
    rows = []
    for name, cols in models.items():
        cv = []
        for i, (s, c, e) in enumerate(FOLDS):
            tr, va = split(df, s, c, e)
            p, _ = fit_predict(name, cols, tr, va, a.threads)
            cv.append(scores(va[TARGET], p))
            print(f"{name} fold{i + 1}: {cv[-1]}", flush=True)
        # threshold picked on the last CV fold, then applied unchanged to the test month
        thr = best_threshold(va[TARGET], p) if cols else None
        tr, te = split(df, *TEST)
        t0 = time.time()
        p, model = fit_predict(name, cols, tr, te, a.threads)
        fit_s = time.time() - t0
        test = scores(te[TARGET], p, thr)
        row = {"model": name, "n_features": len(cols),
               "cv_roc_auc_mean": round(float(np.mean([f["roc_auc"] for f in cv])), 4),
               "cv_roc_auc_std": round(float(np.std([f["roc_auc"] for f in cv])), 4),
               "cv_pr_auc_mean": round(float(np.mean([f["pr_auc"] for f in cv])), 4),
               **{f"test_{k}": v for k, v in test.items()}, "threshold": thr, "fit_seconds": round(fit_s, 1)}
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"features": ",".join(cols) or "none", "split": "time-based", **(XGB_PARAMS if name.startswith("XGB") else {})})
            mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float)) and v is not None})
        rows.append(row)
        print(row, flush=True)
        if name == "XGB: schedule + history + operational":
            model.save_model(md / "xgb_delay.json")
            (md / "xgb_delay_meta.json").write_text(json.dumps({"features": cols, "categories": cats, "threshold": thr,
                                                                 "train_window": [TEST[0], TEST[1]]}))
            imp = sorted(zip(cols, model.feature_importances_.tolist()), key=lambda t: -t[1])
            (out / "delay_feature_importance.json").write_text(json.dumps([{"feature": f, "gain_share": round(v, 4)} for f, v in imp], indent=2))

    tr, te = split(df, *TEST)
    meta = {"dataset": "BTS Reporting Carrier On-Time Performance, Jan-Mar 2024 (operated flights)",
            "train_rows": len(tr), "test_rows": len(te), "test_positive_rate": round(float(te[TARGET].mean()), 4),
            "results": rows}
    (out / "delay_results.json").write_text(json.dumps(meta, indent=2))
    lines = ["| Model | CV ROC-AUC (2 folds) | Test ROC-AUC | Test PR-AUC | Test Brier | Test F1 @thr |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['cv_roc_auc_mean']} ± {r['cv_roc_auc_std']} | {r['test_roc_auc']} | {r['test_pr_auc']} | {r['test_brier']} | {r.get('test_f1', '-')} |")
    (out / "delay_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
