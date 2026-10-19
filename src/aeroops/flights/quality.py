"""Lightweight data-quality checks on the silver layer (no external DQ framework needed)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass
class CheckResult:
    name: str
    severity: str  # "error" fails the pipeline, "warn" is reported only
    failed_rows: int
    total_rows: int
    passed: bool

    def to_dict(self):
        return asdict(self)


def run_checks(silver: DataFrame, quarantined_rows: int = 0, bronze_rows: int | None = None) -> list[CheckResult]:
    total = silver.count()
    rules = {
        # name: (severity, condition that marks a row as FAILING)
        "keys_not_null": ("error", F.col("flight_date").isNull() | F.col("carrier").isNull()
                          | F.col("origin").isNull() | F.col("dest").isNull()),
        "distance_positive": ("error", F.col("distance") <= 0),
        "dep_hour_in_range": ("error", ~F.col("dep_hour").between(0, 23)),
        "operated_flights_have_arr_delay": ("warn", ~F.col("cancelled") & ~F.col("diverted") & F.col("arr_delay").isNull()),
        "cancelled_flights_have_no_air_time": ("warn", F.col("cancelled") & F.col("air_time").isNotNull()),
        "arr_delay_plausible_range": ("warn", (F.col("arr_delay") < -120) | (F.col("arr_delay") > 2880)),
        "cause_minutes_non_negative": ("error", (F.coalesce(F.col("carrier_delay"), F.lit(0)) < 0)
                                       | (F.coalesce(F.col("weather_delay"), F.lit(0)) < 0)
                                       | (F.coalesce(F.col("nas_delay"), F.lit(0)) < 0)),
    }
    aggs = [F.sum(F.when(cond, 1).otherwise(0)).alias(name) for name, (_, cond) in rules.items()]
    dup = total - silver.select("flight_date", "carrier", "flight_number", "origin", "dest", "crs_dep_time").distinct().count()
    row = silver.agg(*aggs).collect()[0].asDict()
    results = [CheckResult(n, sev, int(row[n] or 0), total, int(row[n] or 0) == 0) for n, (sev, _) in rules.items()]
    results.append(CheckResult("unique_flight_key", "error", dup, total, dup == 0))
    if bronze_rows:
        q_ratio_bad = quarantined_rows / bronze_rows > 0.01
        results.append(CheckResult("quarantine_ratio_below_1pct", "error", quarantined_rows, bronze_rows, not q_ratio_bad))
    if total == 0:
        results.append(CheckResult("silver_not_empty", "error", 1, 0, False))
    return results


def summarize(results: list[CheckResult]) -> dict:
    return {
        "passed": all(r.passed for r in results if r.severity == "error"),
        "checks": [r.to_dict() for r in results],
    }
