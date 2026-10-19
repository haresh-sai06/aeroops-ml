"""Bronze -> silver -> gold transformations (pure functions on DataFrames, easy to test)."""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Raw BTS column -> silver column name
RAW_COLUMNS = {
    "FlightDate": "flight_date",
    "Reporting_Airline": "carrier",
    "Tail_Number": "tail_number",
    "Flight_Number_Reporting_Airline": "flight_number",
    "Origin": "origin",
    "OriginCityName": "origin_city",
    "OriginState": "origin_state",
    "Dest": "dest",
    "DestCityName": "dest_city",
    "CRSDepTime": "crs_dep_time",
    "DepDelay": "dep_delay",
    "DepDel15": "dep_del15",
    "TaxiOut": "taxi_out",
    "TaxiIn": "taxi_in",
    "ArrDelay": "arr_delay",
    "ArrDel15": "arr_del15",
    "Cancelled": "cancelled",
    "CancellationCode": "cancellation_code",
    "Diverted": "diverted",
    "AirTime": "air_time",
    "Distance": "distance",
    "CarrierDelay": "carrier_delay",
    "WeatherDelay": "weather_delay",
    "NASDelay": "nas_delay",
    "SecurityDelay": "security_delay",
    "LateAircraftDelay": "late_aircraft_delay",
}
NUMERIC = ["dep_delay", "dep_del15", "taxi_out", "taxi_in", "arr_delay", "arr_del15", "cancelled", "diverted",
           "air_time", "distance", "carrier_delay", "weather_delay", "nas_delay", "security_delay",
           "late_aircraft_delay"]
CAUSES = ["carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]


def to_bronze(raw: DataFrame) -> DataFrame:
    """Select the useful raw columns (as strings) and add lineage metadata."""
    cols = [F.col(f"`{src}`").cast("string").alias(dst) for src, dst in RAW_COLUMNS.items()]
    return raw.select(*cols).withColumn("_ingested_at", F.current_timestamp()) \
              .withColumn("_source_file", F.input_file_name())


def to_silver(bronze: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Type-cast, derive columns, dedupe. Returns (clean, quarantined) DataFrames."""
    df = bronze
    for c in NUMERIC:
        df = df.withColumn(c, F.col(c).cast(T.DoubleType()))
    df = (
        df.withColumn("flight_date", F.to_date("flight_date"))
        .withColumn("crs_dep_time", F.lpad(F.col("crs_dep_time"), 4, "0"))
        .withColumn("dep_hour", (F.col("crs_dep_time").substr(1, 2).cast("int") % 24))
        .withColumn("cancelled", F.col("cancelled").cast("int") == 1)
        .withColumn("diverted", F.col("diverted").cast("int") == 1)
        .withColumn("is_arr_delayed", F.coalesce(F.col("arr_del15"), F.lit(0.0)) == 1.0)
        .withColumn("route", F.concat_ws("-", "origin", "dest"))
        .withColumn("year", F.year("flight_date"))
        .withColumn("month", F.month("flight_date"))
        .withColumn("day_of_week", F.dayofweek("flight_date"))
        .drop("_source_file")
    )
    valid = (
        F.col("flight_date").isNotNull()
        & F.col("carrier").isNotNull()
        & F.col("origin").rlike("^[A-Z0-9]{3}$")
        & F.col("dest").rlike("^[A-Z0-9]{3}$")
        & (F.col("origin") != F.col("dest"))
        & F.col("distance").isNotNull() & (F.col("distance") > 0)
        & F.col("dep_hour").between(0, 23)
    )
    quarantined = df.filter(~valid | valid.isNull())
    clean = df.filter(valid).dropDuplicates(["flight_date", "carrier", "flight_number", "origin", "dest", "crs_dep_time"])
    return clean, quarantined


def _flight_kpis():
    operated = ~F.col("cancelled") & ~F.col("diverted")
    return [
        F.count("*").alias("flights"),
        F.sum(F.col("cancelled").cast("int")).alias("cancelled"),
        F.sum(F.col("diverted").cast("int")).alias("diverted"),
        F.sum(F.when(operated & F.col("is_arr_delayed"), 1).otherwise(0)).alias("arr_delayed_15"),
        F.sum(F.when(operated, 1).otherwise(0)).alias("operated"),
        F.avg(F.when(operated, F.col("arr_delay"))).alias("avg_arr_delay_min"),
        F.avg(F.when(operated, F.col("dep_delay"))).alias("avg_dep_delay_min"),
        F.avg("taxi_out").alias("avg_taxi_out_min"),
    ]


def _with_rates(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("otp_pct", F.round(100 * (1 - F.col("arr_delayed_15") / F.col("operated")), 2))
        .withColumn("cancel_pct", F.round(100 * F.col("cancelled") / F.col("flights"), 2))
    )


def gold_tables(silver: DataFrame) -> dict[str, DataFrame]:
    daily = _with_rates(silver.groupBy("flight_date").agg(*_flight_kpis()))
    carrier_month = _with_rates(silver.groupBy("year", "month", "carrier").agg(*_flight_kpis()))
    airport_daily = _with_rates(
        silver.groupBy("flight_date", F.col("origin").alias("airport"), F.col("origin_state").alias("state"))
        .agg(*_flight_kpis())
    )
    route = _with_rates(silver.groupBy("route", "origin", "dest").agg(*_flight_kpis(), F.avg("distance").alias("distance")))
    hourly = _with_rates(silver.groupBy("dep_hour", "day_of_week").agg(*_flight_kpis()))
    causes = silver.filter(F.col("is_arr_delayed")).groupBy("year", "month", "carrier").agg(
        *[F.sum(F.coalesce(F.col(c), F.lit(0.0))).alias(f"{c}_min") for c in CAUSES]
    )
    return {
        "daily_kpis": daily,
        "carrier_monthly": carrier_month,
        "airport_daily": airport_daily,
        "route_stats": route,
        "hourly_profile": hourly,
        "delay_causes": causes,
    }
