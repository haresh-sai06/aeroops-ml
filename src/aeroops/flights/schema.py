"""Feature definitions shared by the Spark job (offline) and the API (online). No Spark import here."""

SCHEDULE = ["carrier", "origin", "dest", "dep_hour", "day_of_week", "day_of_month", "distance"]
HISTORY_SPECS = {  # name: (grouping keys, lookback days)
    "origin_7d": (["origin"], 7),
    "dest_7d": (["dest"], 7),
    "carrier_7d": (["carrier"], 7),
    "route_28d": (["origin", "dest"], 28),
    "origin_hour_28d": (["origin", "dep_hour"], 28),
}
HISTORY = [f"{n}_delay_rate" for n in HISTORY_SPECS] + [f"{n}_flights" for n in HISTORY_SPECS]
OPERATIONAL = ["prev_leg_arr_delay", "prev_leg_dep_delay", "leg_of_day"]
CATEGORICAL = ["carrier", "origin", "dest"]
TARGET = "is_arr_delayed"
FEATURE_GROUPS = {"schedule": SCHEDULE, "history": HISTORY, "operational": OPERATIONAL}
