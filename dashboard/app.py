"""AeroOps Streamlit dashboard: network KPIs (DuckDB over gold parquet) + model results.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aeroops.kpis import connect, run_all  # noqa: E402

LAKE = Path(os.getenv("AEROOPS_LAKE_DIR", ROOT / "data" / "lake"))
REPORTS = ROOT / "reports"

st.set_page_config(page_title="AeroOps", layout="wide")
st.title("AeroOps: US flight operations & predictive models")
st.caption("BTS Reporting Carrier On-Time Performance, Jan-Mar 2024 · NASA C-MAPSS FD001")


@st.cache_resource
def con():
    return connect(LAKE)


@st.cache_data
def q(sql: str) -> pd.DataFrame:
    return con().execute(sql).df()


tab_ops, tab_airports, tab_models = st.tabs(["Network KPIs", "Airports & causes", "Models"])

with tab_ops:
    carriers = q("SELECT DISTINCT carrier FROM gold.carrier_monthly ORDER BY 1")["carrier"].tolist()
    daily = q("SELECT * FROM gold.daily_kpis ORDER BY flight_date")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flights", f"{int(daily.flights.sum()):,}")
    c2.metric("On-time (arr <15 min)", f"{100 * (1 - daily.arr_delayed_15.sum() / daily.operated.sum()):.1f}%")
    c3.metric("Cancelled", f"{100 * daily.cancelled.sum() / daily.flights.sum():.2f}%")
    c4.metric("Avg arrival delay", f"{(daily.avg_arr_delay_min * daily.operated).sum() / daily.operated.sum():.1f} min")
    st.plotly_chart(px.line(daily, x="flight_date", y=["otp_pct", "cancel_pct"], title="Daily on-time % and cancellation %"),
                    use_container_width=True)
    pick = st.multiselect("Carriers", carriers, default=["AA", "DL", "UA", "WN"])
    cm = q("SELECT * FROM gold.carrier_monthly")
    cm = cm[cm.carrier.isin(pick)]
    st.plotly_chart(px.bar(cm, x="month", y="otp_pct", color="carrier", barmode="group", title="Monthly on-time % by carrier"),
                    use_container_width=True)
    hp = q("SELECT dep_hour, SUM(arr_delayed_15)*100.0/SUM(operated) AS delayed_pct FROM gold.hourly_profile GROUP BY 1 ORDER BY 1")
    st.plotly_chart(px.area(hp, x="dep_hour", y="delayed_pct", title="Delay share by scheduled departure hour (delays propagate through the day)"),
                    use_container_width=True)

with tab_airports:
    res = run_all(LAKE, ROOT / "sql")
    left, right = st.columns(2)
    left.subheader("Busiest-50 airports with the highest delay share")
    left.dataframe(res["03_worst_airports"], hide_index=True)
    right.plotly_chart(px.pie(res["04_delay_cause_mix"], names="cause", values="pct_of_delay_minutes",
                              title="Delay minutes by root cause"), use_container_width=True)
    st.subheader("Carrier ranking")
    st.dataframe(res["02_carrier_ranking"], hide_index=True)

with tab_models:
    f = REPORTS / "delay_results.json"
    if f.exists():
        d = json.loads(f.read_text())
        st.subheader("Arrival-delay classifier (time-based split, test = March 2024)")
        st.dataframe(pd.DataFrame(d["results"])[["model", "cv_roc_auc_mean", "test_roc_auc", "test_pr_auc", "test_brier", "test_f1"]],
                     hide_index=True)
    fi = REPORTS / "delay_feature_importance.json"
    if fi.exists():
        st.plotly_chart(px.bar(pd.DataFrame(json.loads(fi.read_text())).head(12), x="gain_share", y="feature",
                               orientation="h", title="Delay model: feature importance (gain share)"), use_container_width=True)
    r = REPORTS / "rul_results_FD001.json"
    if r.exists():
        st.subheader("Engine RUL regression (C-MAPSS FD001, official test set)")
        st.dataframe(pd.DataFrame(json.loads(r.read_text())["results"]), hide_index=True)
    shap_png = REPORTS / "shap_summary_FD001.png"
    if shap_png.exists():
        st.image(str(shap_png), caption="SHAP summary for the XGBoost RUL model")
