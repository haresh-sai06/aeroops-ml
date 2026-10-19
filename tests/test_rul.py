import numpy as np
import pandas as pd
import pytest

from aeroops.rul.data import USED_SENSORS, add_train_rul, validate
from aeroops.rul.features import MinMax, last_rows, sequence_windows, tabular_features
from aeroops.rul.metrics import nasa_score, rmse


def synthetic(units=3, cycles=40):
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, units + 1):
        for c in range(1, cycles + 1):
            rows.append({"unit": u, "cycle": c, **{s: 100 + c * 0.1 + rng.normal() for s in USED_SENSORS}})
    return pd.DataFrame(rows)


def test_metrics():
    assert rmse([10, 20], [10, 20]) == 0
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    # late predictions are penalised harder than early ones
    assert nasa_score([50], [60]) > nasa_score([50], [40]) > 0


def test_rul_label_and_cap():
    df = add_train_rul(synthetic(units=1, cycles=200), cap=125)
    assert df["rul"].iloc[-1] == 0 and df["rul"].max() == 125


def test_rolling_features_no_future_leakage():
    df = synthetic(units=1, cycles=20)
    full = tabular_features(df)
    partial = tabular_features(df.iloc[:10])
    pd.testing.assert_series_equal(full.iloc[9][partial.columns], partial.iloc[9], check_names=False)
    assert len(last_rows(full)) == 1


def test_sequence_windows_pad_short_units():
    df = add_train_rul(synthetic(units=2, cycles=10))
    sc = MinMax().fit(df[USED_SENSORS].to_numpy())
    X, y = sequence_windows(df, sc, window=30, last_only=True)
    assert X.shape == (2, 30, len(USED_SENSORS)) and y.shape == (2,)


def test_validate_rejects_gaps():
    df = synthetic(units=1, cycles=5).drop(index=2)
    with pytest.raises(ValueError):
        validate(df)
