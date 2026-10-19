"""Leakage-safe Spark feature pipeline for arrival-delay (>=15 min) prediction.

Feature groups (used for ablations):
  schedule    - known when the flight is scheduled (carrier, airports, hour, weekday, distance)
  history     - rolling delay rates for origin / dest / carrier / route computed ONLY from days
                strictly before the flight date (as-of D-1), so no same-day leakage
  operational - inbound-aircraft signal: the same tail number's previous leg that day
                (its arrival delay is known before this flight departs)

Usage: python -m aeroops.flights.features --lake-dir data/lake
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ..spark import get_spark
from .schema import CATEGORICAL, FEATURE_GROUPS, HISTORY, HISTORY_SPECS, OPERATIONAL, SCHEDULE, TARGET  # noqa: F401


def _day_index(col: str = "flight_date"):
    return F.datediff(F.col(col), F.lit("1970-01-01"))


def history_features(operated: DataFrame, keys: list[str], days: int, name: str) -> DataFrame:
    """Per (keys, date): delay rate over the window [D-days, D-1]."""
    daily = operated.groupBy(*keys, "flight_date").agg(
        F.count("*").alias("n"), F.sum(F.col(TARGET).cast("int")).alias("d")
    ).withColumn("_day", _day_index())
    w = Window.partitionBy(*keys).orderBy("_day").rangeBetween(-days, -1)
    return daily.select(
        *keys, "flight_date",
        (F.sum("d").over(w) / F.sum("n").over(w)).alias(f"{name}_delay_rate"),
        F.coalesce(F.sum("n").over(w), F.lit(0)).alias(f"{name}_flights"),
    )


def operational_features(df: DataFrame) -> DataFrame:
    w = Window.partitionBy("tail_number", "flight_date").orderBy("crs_dep_time")
    return (
        df.withColumn("prev_leg_arr_delay", F.when(F.col("tail_number").isNotNull(), F.lag("arr_delay").over(w)))
        .withColumn("prev_leg_dep_delay", F.when(F.col("tail_number").isNotNull(), F.lag("dep_delay").over(w)))
        .withColumn("leg_of_day", F.when(F.col("tail_number").isNotNull(), F.row_number().over(w)))
    )


def build_features(silver: DataFrame) -> DataFrame:
    # Operational lag uses ALL scheduled legs (incl. cancelled) to get the right ordering.
    df = operational_features(silver.withColumn("day_of_month", F.dayofmonth("flight_date")))
    # Labelled population: flights that operated (not cancelled / diverted).
    operated = df.filter(~F.col("cancelled") & ~F.col("diverted") & F.col("arr_delay").isNotNull())
    out = operated
    for name, (keys, days) in HISTORY_SPECS.items():
        out = out.join(history_features(operated, keys, days, name), on=[*keys, "flight_date"], how="left")
    return out.select("flight_date", "year", "month", *SCHEDULE, *HISTORY, *OPERATIONAL,
                      F.col(TARGET).cast("int").alias(TARGET), "arr_delay")


def online_snapshot(silver: DataFrame) -> dict:
    """'Online feature store' snapshot: history features as of the last date in the lake,
    served by the API so online features match the offline definition (window ends at D-1)."""
    operated = silver.filter(~F.col("cancelled") & ~F.col("diverted") & F.col("arr_delay").isNotNull())
    last_day = operated.agg(F.max("flight_date")).collect()[0][0]
    snap = {"as_of": str(last_day), "tables": {}}
    for name, (keys, days) in HISTORY_SPECS.items():
        rows = (operated.filter(F.col("flight_date") > F.date_sub(F.lit(last_day), days))
                .groupBy(*keys).agg(F.count("*").alias("n"), F.avg(F.col(TARGET).cast("double")).alias("rate"))
                .collect())
        snap["tables"][name] = {"|".join(str(r[k]) for k in keys): [round(r["rate"], 5), int(r["n"])] for r in rows}
    return snap


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake-dir", default="data/lake")
    ap.add_argument("--models-dir", default="models")
    a = ap.parse_args(argv)
    spark = get_spark("aeroops-features")
    lake = Path(a.lake_dir)
    silver = spark.read.parquet(str(lake / "silver" / "flights"))
    feats = build_features(silver)
    feats.write.mode("overwrite").partitionBy("year", "month").parquet(str(lake / "features" / "flight_delay"))
    n = spark.read.parquet(str(lake / "features" / "flight_delay")).count()
    Path(a.models_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.models_dir) / "online_features.json").write_text(json.dumps(online_snapshot(silver)))
    print(f"feature rows: {n}")
    spark.stop()


if __name__ == "__main__":
    main()
