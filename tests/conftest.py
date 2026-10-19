import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def spark():
    from aeroops.spark import get_spark

    s = get_spark("aeroops-tests", cores="2")
    yield s
    s.stop()


def raw_row(**kw):
    base = {"FlightDate": "2024-01-02", "Reporting_Airline": "AA", "Tail_Number": "N1", "Flight_Number_Reporting_Airline": "100",
            "Origin": "JFK", "OriginCityName": "New York, NY", "OriginState": "NY", "Dest": "LAX", "DestCityName": "Los Angeles, CA",
            "CRSDepTime": "0830", "DepDelay": "5", "DepDel15": "0", "TaxiOut": "12", "TaxiIn": "6", "ArrDelay": "20",
            "ArrDel15": "1", "Cancelled": "0", "CancellationCode": None, "Diverted": "0", "AirTime": "330", "Distance": "2475",
            "CarrierDelay": "20", "WeatherDelay": "0", "NASDelay": "0", "SecurityDelay": "0", "LateAircraftDelay": "0"}
    base.update(kw)
    return base


@pytest.fixture()
def raw_df(spark):
    rows = [
        raw_row(),
        raw_row(),  # exact duplicate -> removed in silver
        raw_row(FlightDate="2024-01-03", ArrDelay="-3", ArrDel15="0", CarrierDelay=None, CRSDepTime="0700"),
        raw_row(FlightDate="2024-01-03", Flight_Number_Reporting_Airline="101", CRSDepTime="1300", Origin="LAX", Dest="JFK",
                ArrDelay="45", ArrDel15="1"),
        raw_row(FlightDate="2024-01-04", Cancelled="1", ArrDelay=None, ArrDel15=None, AirTime=None, CancellationCode="B"),
        raw_row(Origin="JFK", Dest="JFK", Flight_Number_Reporting_Airline="999"),  # invalid -> quarantine
        raw_row(Distance="0", Flight_Number_Reporting_Airline="998"),  # invalid -> quarantine
    ]
    return spark.createDataFrame(rows)
