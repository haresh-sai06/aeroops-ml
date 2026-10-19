"""Run the SQL KPI layer (DuckDB over the gold parquet tables) and write reports/kpis.md.

Usage: python -m aeroops.kpis --lake-dir data/lake
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

GOLD_TABLES = ["daily_kpis", "carrier_monthly", "airport_daily", "route_stats", "hourly_profile", "delay_causes"]


def connect(lake_dir: str | Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE SCHEMA gold")
    for t in GOLD_TABLES:
        con.execute(f"CREATE VIEW gold.{t} AS SELECT * FROM read_parquet('{Path(lake_dir) / 'gold' / t}/*.parquet')")
    return con


def run_all(lake_dir, sql_dir="sql"):
    con = connect(lake_dir)
    out = {}
    for f in sorted(Path(sql_dir).glob("*.sql")):
        out[f.stem] = con.execute(f.read_text()).df()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake-dir", default="data/lake")
    ap.add_argument("--sql-dir", default="sql")
    ap.add_argument("--out", default="reports/kpis.md")
    a = ap.parse_args()
    parts = []
    for name, df in run_all(a.lake_dir, a.sql_dir).items():
        parts.append(f"### {name}\n\n{df.to_markdown(index=False)}\n")
        print(name)
        print(df.to_string(index=False))
    Path(a.out).write_text("# SQL KPI results (BTS on-time data, Jan-Mar 2024)\n\n" + "\n".join(parts))
