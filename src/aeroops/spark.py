from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession


def get_spark(app: str = "aeroops", cores: str | None = None) -> SparkSession:
    # make Python workers use the same interpreter as the driver
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    cores = cores or os.getenv("AEROOPS_SPARK_CORES", "4")
    return (
        SparkSession.builder.appName(app)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", os.getenv("AEROOPS_DRIVER_MEMORY", "1g"))
        .config("spark.sql.shuffle.partitions", os.getenv("AEROOPS_SHUFFLE_PARTITIONS", "8"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
