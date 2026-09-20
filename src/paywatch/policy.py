from dataclasses import dataclass
import hashlib
import numpy as np


def sentinel(event_id, fraction=0.05):
    if not 0 <= fraction <= 1:
        raise ValueError("invalid sentinel fraction")
    value = int.from_bytes(hashlib.blake2b(event_id.encode(), digest_size=8).digest(), "big")
    return value / 2**64 < fraction


def calibrate(probability, coefficient, intercept):
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    z = np.clip(coefficient * np.log(p / (1 - p)) + intercept, -40, 40)
    return 1 / (1 + np.exp(-z))


def route_mask(loss, threshold, sentinels):
    return np.isnan(loss) | (loss >= threshold) | sentinels


def choose_gate(loss, y, sentinels, target_recall=0.98):
    if not 0 < target_recall <= 1 or np.sum(y) == 0:
        raise ValueError("gate selection requires fraud labels and a valid recall target")
    fraud = y == 1
    required = int(np.ceil(target_recall * fraud.sum() - 1e-12))
    forced = np.isnan(loss) | sentinels
    remaining = max(0, required - int((fraud & forced).sum()))
    if remaining == 0:
        finite = loss[np.isfinite(loss)]
        return float(np.nextafter(finite.max(), np.inf)) if len(finite) else 0.0
    fraud_losses = np.sort(loss[fraud & ~forced])[::-1]
    return float(fraud_losses[remaining - 1])


def choose_threshold(probability, y, amounts, routed, review_cost=2.0, loss_fraction=1.0):
    if review_cost < 0 or not 0 <= loss_fraction <= 1:
        raise ValueError("invalid business costs")
    indices = np.flatnonzero(routed)
    indices = indices[np.argsort(-probability[indices], kind="stable")]
    base = float(loss_fraction * amounts[y == 1].sum())
    if len(indices) == 0:
        return 1.000001
    change = review_cost - loss_fraction * amounts[indices] * (y[indices] == 1)
    costs = base + np.cumsum(change)
    probabilities = probability[indices]
    ends = np.r_[np.flatnonzero(probabilities[:-1] != probabilities[1:]), len(indices) - 1]
    options = [(base, 0, 1.000001)]
    options.extend((float(costs[i]), int(i + 1), float(probabilities[i])) for i in ends)
    return min(options)[2]


@dataclass(frozen=True)
class Policy:
    gate_threshold: float
    fraud_threshold: float
    sentinel_fraction: float = 0.05
    coefficient: float = 1.0
    intercept: float = 0.0

    def __post_init__(self):
        if not np.isfinite(
            [self.gate_threshold, self.fraud_threshold, self.coefficient, self.intercept]
        ).all():
            raise ValueError("nonfinite policy")
        if (
            self.gate_threshold < 0
            or not 0 <= self.fraud_threshold <= 1.000001
            or not 0 <= self.sentinel_fraction <= 1
        ):
            raise ValueError("invalid policy bounds")
