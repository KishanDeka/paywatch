import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .features import raw_features
from .schema import Transaction


def load_csv(path):
    path = Path(path)
    df = pd.read_csv(path)
    required = ["Time", "Amount", *[f"V{i}" for i in range(1, 29)], "Class"]
    if set(df.columns) != set(required):
        raise ValueError("expected Time, Amount, V1..V28, Class only")
    if len(df) < 100 or not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError("dataset too short or nonfinite")
    if not df.Class.isin([0, 1]).all():
        raise ValueError("labels must be binary")
    if (df.Time < 0).any() or (df.Amount < 0).any():
        raise ValueError("negative time or amount")
    dataset_id = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    df["event_id"] = [f"{dataset_id}:{i}" for i in range(len(df))]
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def split_chronologically(df):
    times = df.Time.to_numpy()
    cuts = (
        [0]
        + [
            int(np.searchsorted(times, times[int(len(df) * f)], side="left"))
            for f in (0.6, 0.7, 0.8)
        ]
        + [len(df)]
    )
    parts = [df.iloc[a:b].copy() for a, b in zip(cuts[:-1], cuts[1:])]
    for part in parts:
        if len(part) < 10 or part.Class.nunique() != 2:
            raise ValueError("each chronological partition needs >=10 rows and both classes")
    return dict(zip(["train", "calibration", "policy", "test"], parts))


def row_event(row):
    return Transaction(
        event_id=row.event_id,
        time=row.Time,
        amount=row.Amount,
        v=[getattr(row, f"V{i}") for i in range(1, 29)],
    )


def feature_matrix(df):
    rows, previous = [], None
    for row in df.itertuples(index=False):
        event = row_event(row)
        rows.append(raw_features(event, previous))
        previous = event.time
    return np.stack(rows)
