from aeroops.flights.transform import gold_tables, to_bronze, to_silver
from aeroops.kpis import GOLD_TABLES, run_all


def test_sql_kpis_run_on_gold(raw_df, tmp_path):
    silver, _ = to_silver(to_bronze(raw_df))
    for name, df in gold_tables(silver).items():
        df.coalesce(1).write.parquet(str(tmp_path / "gold" / name))
    assert set(GOLD_TABLES) == {p.name for p in (tmp_path / "gold").iterdir()}
    res = run_all(tmp_path, "sql")
    assert res["01_network_kpis"]["flights"].sum() == 4
    assert abs(res["04_delay_cause_mix"]["pct_of_delay_minutes"].sum() - 100) < 0.1
