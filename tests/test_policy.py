import numpy as np
import pytest
from paywatch.policy import sentinel, choose_gate, route_mask, choose_threshold, calibrate
from paywatch.evaluation import metrics, population_stability, drift_reference


def test_sentinel_stable_and_fraction_endpoints():
    assert sentinel("event", 0.05) == sentinel("event", 0.05)
    assert not sentinel("event", 0)
    assert sentinel("event", 1)
    rate = np.mean([sentinel(str(i)) for i in range(10000)])
    assert 0.04 < rate < 0.06


def test_gate_recall_and_warmup():
    loss = np.array([np.nan, 0.01, 0.2, 0.8, 1.0])
    y = np.array([1, 1, 0, 1, 0])
    sampled = np.zeros(5, bool)
    gate = choose_gate(loss, y, sampled, 1.0)
    assert gate == 0.01
    assert route_mask(loss, gate, sampled)[y == 1].all()


def test_gate_cannot_be_fitted_without_fraud():
    with pytest.raises(ValueError):
        choose_gate(np.arange(3), np.zeros(3), np.zeros(3, bool))


def test_cost_threshold_accounts_for_missed_amount():
    p = np.array([0.1, 0.4, 0.9])
    y = np.array([0, 1, 1])
    amount = np.array([10, 500, 5])
    assert choose_threshold(p, y, amount, np.ones(3, bool)) == 0.4


def test_high_review_cost_can_choose_no_alerts():
    assert (
        choose_threshold(
            np.array([0.1, 0.9]), np.array([0, 1]), np.array([0, 1]), np.ones(2, bool), 100
        )
        > 0.9
    )


def test_calibration_extremes_are_finite():
    assert np.isfinite(calibrate(np.array([0, 1]), 2, -5)).all()


def test_metrics_count_fraud_discarded_by_gate():
    result = metrics(
        np.array([0, 1, 1]),
        np.array([0, 0, 0.9]),
        np.array([0, 0, 1], bool),
        np.array([0, 0, 1], bool),
        np.array([1, 100, 5]),
    )
    assert result["recall"] == result["gate_recall"] == 0.5
    assert result["missed_fraud_amount"] == 100


def test_drift_detects_shift_and_handles_constant_columns():
    x = np.column_stack([np.arange(100), np.ones(100)])
    ref = drift_reference(x)
    assert max(population_stability(x, ref)) == 0
    assert population_stability(x + 1000, ref)[0] > 1


def test_sorted_cost_search_matches_exhaustive_thresholds():
    rng = np.random.default_rng(42)
    for _ in range(50):
        p = np.round(rng.random(40), 1)
        y = rng.integers(0, 2, 40)
        amounts = rng.uniform(0, 100, 40)
        routed = rng.random(40) < 0.6
        selected = choose_threshold(p, y, amounts, routed)
        candidates = np.unique(np.r_[0.0, p[routed], 1.000001])

        def cost(t):
            alert = routed & (p >= t)
            return 2 * alert.sum() + amounts[(y == 1) & ~alert].sum()

        assert cost(selected) == pytest.approx(min(cost(t) for t in candidates))
