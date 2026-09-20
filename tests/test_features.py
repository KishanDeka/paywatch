import numpy as np
import pytest
from pydantic import ValidationError
from paywatch.features import Scaler, raw_features, windows, normal_window_mask
from paywatch.schema import Transaction
from paywatch.data import split_chronologically
import pandas as pd


def event(**kwargs):
    return Transaction(**({"event_id": "a", "time": 1, "amount": 0, "v": [0] * 28} | kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amount": -1},
        {"time": -1},
        {"amount": float("nan")},
        {"time": float("inf")},
        {"v": [0] * 27},
        {"v": [0] * 29},
        {"v": [float("inf")] * 28},
        {"v": [float("nan")] * 28},
        {"event_id": ""},
        {"Class": 1},
        {"v": [1e7] * 28},
    ],
)
def test_schema_rejects_bad_events(kwargs):
    with pytest.raises(ValidationError):
        event(**kwargs)


def test_zero_amount_and_tied_timestamps():
    x = raw_features(event(), previous_time=1)
    assert x[-1] == x[-2] == 0


def test_late_event_rejected():
    with pytest.raises(ValueError, match="out_of_order"):
        raw_features(event(), previous_time=2)


def test_constant_feature_and_train_only_scaling():
    scaler = Scaler.fit(np.ones((10, 30)))
    assert np.isfinite(scaler.transform(np.full((1, 30), 1000))).all()
    assert np.all(scaler.mean == 1)
    assert np.all(scaler.scale == 1)


def test_windows_do_not_remove_fraud_and_join_unrelated_rows():
    x = np.arange(18).reshape(9, 2)
    labels = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
    selected = windows(x, 3)[normal_window_mask(labels, 3)]
    np.testing.assert_array_equal(selected[:, 0, 0], [0, 8, 10, 12])


def test_short_window():
    with pytest.raises(ValueError):
        windows(np.zeros((2, 30)), 10)


def test_split_never_shares_timestamps():
    df = pd.DataFrame({"Time": np.repeat(np.arange(200), 3), "Class": np.tile([0, 0, 1], 200)})
    parts = list(split_chronologically(df).values())
    for left, right in zip(parts[:-1], parts[1:]):
        assert left.Time.max() < right.Time.min()


def test_single_class_partition_fails_loudly():
    df = pd.DataFrame({"Time": np.arange(200), "Class": np.zeros(200)})
    with pytest.raises(ValueError, match="both classes"):
        split_chronologically(df)
