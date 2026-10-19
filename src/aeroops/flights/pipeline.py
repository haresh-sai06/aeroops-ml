"""End-to-end pipeline: raw CSV -> bronze parquet -> silver parquet (+quarantine) -> gold parquet.

Usage: python -m aeroops.pipeline --raw-dir data/raw --lake-dir data/lake
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..spark import get_spark
from .quality import run_checks, summarize
from .transform import gold_tables, to_bronze, to_silver


def run(raw_dir: str, lake_dir: str, report_dir: str = "reports") -> dict:
    spark = get_spark()
    lake, reports = Path(lake_dir), Path(report_dir)
    reports.mkdir(parents=True, exist_ok=True)
    timings = {}

    t = time.time()
    raw = spark.read.option("header", True).option("inferSchema", False).csv(f"{raw_dir}/*.csv")
    bronze = to_bronze(raw)
    bronze.write.mode("overwrite").parquet(str(lake / "bronze" / "ontime"))
    bronze = spark.read.parquet(str(lake / "bronze" / "ontime"))
    bronze_rows = bronze.count()
    timings["bronze_s"] = round(time.time() - t, 1)

    t = time.time()
    silver, quarantined = to_silver(bronze)
    silver.write.mode("overwrite").partitionBy("year", "month").parquet(str(lake / "silver" / "flights"))
    quarantined.write.mode("overwrite").parquet(str(lake / "quarantine" / "flights"))
    silver = spark.read.parquet(str(lake / "silver" / "flights"))
    q_rows = spark.read.parquet(str(lake / "quarantine" / "flights")).count()
    silver_rows = silver.count()
    timings["silver_s"] = round(time.time() - t, 1)

    t = time.time()
    dq = summarize(run_checks(silver, quarantined_rows=q_rows, bronze_rows=bronze_rows))
    timings["quality_s"] = round(time.time() - t, 1)
    (reports / "data_quality.json").write_text(json.dumps(dq, indent=2))
    if not dq["passed"]:
        raise SystemExit("data quality checks failed, see reports/data_quality.json")

    t = time.time()
    gold_counts = {}
    for name, df in gold_tables(silver).items():
        df.coalesce(1).write.mode("overwrite").parquet(str(lake / "gold" / name))
        gold_counts[name] = spark.read.parquet(str(lake / "gold" / name)).count()
    timings["gold_s"] = round(time.time() - t, 1)

    summary = {
        "bronze_rows": bronze_rows,
        "silver_rows": silver_rows,
        "quarantined_rows": q_rows,
        "duplicates_removed": bronze_rows - silver_rows - q_rows,
        "gold_tables": gold_counts,
        "timings": timings,
    }
    (reports / "pipeline_run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    spark.stop()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--lake-dir", default="data/lake")
    ap.add_argument("--report-dir", default="reports")
    a = ap.parse_args()
    run(a.raw_dir, a.lake_dir, a.report_dir)
