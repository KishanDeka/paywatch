from dataclasses import dataclass
import numpy as np

FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "log_gap"]


def raw_features(event, previous_time=None):
    if previous_time is not None and event.time < previous_time:
        raise ValueError("out_of_order")
    gap = 0.0 if previous_time is None else event.time - previous_time
    return np.asarray([*event.v, np.log1p(event.amount), np.log1p(gap)], dtype=np.float32)


@dataclass
class Scaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("invalid training features")
        scale = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(scale < 1e-8, 1.0, scale))

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("invalid inference features")
        return ((x - self.mean) / self.scale).astype(np.float32)

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            mean, scale = data["mean"], data["scale"]
        if (
            mean.shape != (30,)
            or scale.shape != (30,)
            or not np.isfinite([mean, scale]).all()
            or (scale <= 0).any()
        ):
            raise ValueError("invalid scaler")
        return cls(mean, scale)


def windows(x, length=10):
    if length < 1 or len(x) < length:
        raise ValueError("not enough rows for sequence length")
    return np.lib.stride_tricks.sliding_window_view(x, length, axis=0).transpose(0, 2, 1)


def normal_window_mask(labels, length):
    return np.convolve(np.asarray(labels), np.ones(length, dtype=int), mode="valid") == 0
