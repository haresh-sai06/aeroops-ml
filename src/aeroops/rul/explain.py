"""SHAP explanations for the trained XGBoost model.

Usage: python -m pdm.explain --data-dir data/raw/cmapss --fd FD001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from .data import load_fd  # noqa: E402
from .features import last_rows, tabular_features  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/cmapss")
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    md, out = Path(args.models_dir), Path(args.out)

    model = XGBRegressor()
    model.load_model(md / f"xgb_rul_{args.fd}.json")
    cols = json.loads((md / f"xgb_rul_{args.fd}_features.json").read_text())
    _, test, _ = load_fd(args.data_dir, args.fd)
    X = last_rows(tabular_features(test))[cols]

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    shap.summary_plot(values, X, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out / f"shap_summary_{args.fd}.png", dpi=120)
    plt.close()

    importance = sorted(zip(cols, np.abs(values).mean(axis=0)), key=lambda t: -t[1])[:10]
    top = [{"feature": f, "mean_abs_shap": round(float(v), 3)} for f, v in importance]
    (out / f"shap_top_features_{args.fd}.json").write_text(json.dumps(top, indent=2))
    for t in top:
        print(t)


if __name__ == "__main__":
    main()
