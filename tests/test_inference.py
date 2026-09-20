import json
import os
from pathlib import Path
import numpy as np
import pytest
from paywatch.inference import Engine
from paywatch.schema import Transaction
from paywatch.features import raw_features, windows


@pytest.fixture
def engine():
    path = Path(os.getenv("PAYWATCH_TEST_MODELS", "models/demo"))
    if not (path / "manifest.json").exists():
        pytest.skip("train the synthetic demo to enable ONNX tests")
    return Engine(path)


def event(i):
    return Transaction(event_id=f"test:{i}", time=i, amount=i + 1, v=[0.01 * i] * 28)


def test_warmup_and_restart_equivalence(engine):
    state = None
    for i in range(engine.length - 1):
        decision, state = engine.score(event(i), state)
        assert decision["route_reason"] == "warmup"
        assert decision["fraud_probability"] is not None
    before = json.dumps(state)
    a, next_a = engine.score(event(engine.length), state)
    b, next_b = engine.score(event(engine.length), json.loads(before))
    assert a["anomaly_loss"] == b["anomaly_loss"]
    assert next_a == next_b
    assert json.dumps(state) == before


def test_offline_online_feature_and_loss_parity(engine):
    events = [event(i) for i in range(30)]
    x = engine.scaler.transform(
        np.stack([raw_features(e, events[i - 1].time if i else None) for i, e in enumerate(events)])
    )
    batch = windows(x, engine.length).copy()
    expected = np.mean(
        (engine.ae.run(None, {"sequence": batch})[0][:, -1] - batch[:, -1]) ** 2, axis=-1
    )
    state, actual = None, []
    for e in events:
        decision, state = engine.score(e, state)
        if decision["anomaly_loss"] is not None:
            actual.append(decision["anomaly_loss"])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_late_event_does_not_mutate_state(engine):
    _, state = engine.score(event(5))
    original = json.dumps(state)
    with pytest.raises(ValueError, match="out_of_order"):
        engine.score(event(4), state)
    assert json.dumps(state) == original


def test_changed_model_cannot_reuse_old_state(engine):
    with pytest.raises(ValueError, match="another model"):
        engine.score(event(1), {"rows": [], "last_time": None, "model_version": "other"})


def test_artifact_tampering_detected(engine, tmp_path):
    import shutil

    source = Path(os.getenv("PAYWATCH_TEST_MODELS", "models/demo"))
    for name in ["manifest.json", "scaler.npz", "autoencoder.onnx", "xgboost.onnx"]:
        shutil.copy(source / name, tmp_path / name)
    with (tmp_path / "autoencoder.onnx").open("ab") as f:
        f.write(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        Engine(tmp_path)


def test_explicit_routing_branches(engine):
    from dataclasses import replace

    state = None
    for i in range(engine.length):
        _, state = engine.score(event(i), state)
    engine.policy = replace(engine.policy, gate_threshold=1e12, sentinel_fraction=0)
    decision, _ = engine.score(event(20), state)
    assert decision["route_reason"] == "gate_pass"
    assert decision["fraud_probability"] is None
    engine.policy = replace(engine.policy, sentinel_fraction=1)
    decision, _ = engine.score(event(20), state)
    assert decision["route_reason"] == "sentinel"
    assert decision["fraud_probability"] is not None
    engine.policy = replace(engine.policy, gate_threshold=0)
    decision, _ = engine.score(event(20), state)
    assert decision["route_reason"] == "anomaly"


def test_export_parity_extreme_input_and_batch_one(engine):
    import torch
    from paywatch.lstm import LSTMAutoencoder

    path = Path(os.getenv("PAYWATCH_TEST_MODELS", "models/demo"))
    model = LSTMAutoencoder(hidden=engine.manifest["hidden"], length=engine.length)
    model.load_state_dict(torch.load(path / "autoencoder.pt", weights_only=True))
    model.eval()
    for value in (0.0, 1e4, -1e4):
        x = np.full((1, engine.length, 30), value, np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        actual = engine.ae.run(None, {"sequence": x})[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
